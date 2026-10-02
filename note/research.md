# Research: how to get the lowest C4 val perplexity

Goal: the lowest judge perplexity on 100k C4 `en` validation docs. The judge scores each doc's first 1024 tokens, starting at position 0 with no BOS, as total loss / total tokens.
Constraint: the judge loads only our weights into stock GPT-2 small, so every training trick must fold back exactly into stock weights.
Baseline: AB/AC, val loss ≈ 3.34 (ppl ≈ 28.2), ~950k tok/s, ~1.65B tokens per 29-minute run on 2×H200.
Distillation from an external pretrained teacher is allowed (user decision, 2026-10-02).

Gains below are in val loss (nats). Most are extrapolated from other setups (FineWeb, other architectures, other scales), not measured on ours.

## Ranked findings

| # | Change | Expected Δloss | Cost | Folds into stock GPT-2 |
|---|---|---|---|---|
| 1 | Distillation from an external GPT-2-tokenizer teacher | −0.05 to −0.1 (by analogy) | high | yes, it is only a loss |
| 2 | Muon on block matrices, AdamW for the rest | −0.03 to −0.06 | medium (~100 lines) | yes, optimizer only |
| 3 | Throughput: bf16/fused CE, FA3, fewer mid-run evals | ~−0.012 per +10% tok/s | low | yes |
| 4 | AdamW retune: weight decay by timescale, β2 0.99, LR sweep | −0.01 to −0.03 | none | yes |
| 5 | Annealed multi-token prediction with no extra params | ~−0.007 | low | yes |
| 6 | SkyLadder: attention window grows from 32 to 1024 | −0.005 to −0.02 (guess) | low | yes, if training ends at full 1024 |
| 7 | Batch of 128 rows (131k tokens) | ±0.01 | none | yes |
| 8 | Tail EMA; save the best of θ, EMA or 0.4θ + 0.6·EMA | 0 to 0.005 | low | yes |
| 9 | Temperature T folded into ln_f | < 0.003 | none | yes |

### 1. Distillation from an external teacher
- **Evidence:**
  - Gemma 2 (Table 7): a 200M student went from ppl 23 to 21 (about −0.09 nats) when distilled from 7B instead of trained from scratch. https://arxiv.org/abs/2408.00118
  - Busbridge et al. 2025: distillation only wins when the teacher already exists. Pure KD at τ = 1 was best. https://arxiv.org/abs/2502.08606
  - Capacity gap: the best teacher is about 2.5× the student. https://arxiv.org/abs/2311.07052
  - Peng et al. 2024: top-p/top-k truncation of teacher logits cuts storage about 4000×. https://arxiv.org/abs/2410.16215
- **Teachers with the exact GPT-2 vocab** (verify the ids match):
  - gpt2-medium/large/xl (WebText)
  - GPT-Neo 125M–2.7B (Pile)
  - Cerebras-GPT 111M–13B (Pile)
  - Not usable as-is: GPT-J (vocab padded to 50400), OPT (same BPE but different id order), Pythia (NeoX tokenizer).
- **Before building anything:** measure each teacher's C4 ppl with the judge protocol on our 10k val docs. That means each doc run on its own, position 0, no BOS, truncated to 1024. A teacher is only worth it if it is clearly below our 28.
- **Two ways to run it:**
  - **Offline (preferred for big teachers):** a separate sbatch job writes teacher top-k logits for the train rows (k ≈ 8–32, uint16 index + fp16 value). Training uses KD on the renormalized top-k at τ = 1, optionally mixed with hard CE. 1.6B tokens at top-8 is about 51 GB, so check the disk quota first.
  - **Online:** run the teacher forward inside the training job. A gpt2-medium forward (~0.7 GFLOP/token) costs about as much as student training (~0.79 GFLOP/token), so student tokens roughly halve. This probably only breaks even.
- **Masking:** teacher inference must use the student's document masking: varlen attention with positions reset per doc.

### 2. Muon
- **Params:**
  - **Muon** gets the 2D weights in the blocks: `c_attn` (split into Q, K, V 768×768), `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`. HF `Conv1D` stores weights as (in, out), so transpose before orthogonalizing.
  - **AdamW** gets the tied `wte`/`lm_head`, `wpe`, LayerNorms and all biases. For the tied embedding: lr ~0.0036, betas (0.9, 0.95), wd 0.
- **Muon settings** (from modded-nanogpt Track 3, ~1.7B tokens):
  - Nesterov momentum 0.95, warmed up from 0.85 over 300 steps.
  - 5 Newton–Schulz steps in bf16; Polar Express coefficients are optional.
  - Scale each update by `sqrt(max(1, out/in))`.
  - Decoupled wd: `p *= 1 - lr*wd`.
  - Schedule: flat for 30%, then linear decay to 0.
  - Sweep lr {0.015, 0.025, 0.035} × wd {0.025, 0.05}. Weight decay is the most sensitive knob.
- **Evidence:**
  - Keller's tied-embedding comparison at 2.67B tokens:

    | Optimizer | Val loss | ms/step |
    |---|---|---|
    | Muon | 3.2760 | 142 |
    | Adam | 3.3406 | 139 |
    | SOAP | 3.2752 | 301 |
    | Distributed Shampoo | 3.2858 | 179 |

    https://kellerjordan.github.io/posts/muon/ , https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-29_Optimizers/README.md
  - In the speedrun, Muon reached the target loss with 3.25B tokens instead of 5B, about a 1.54× effective throughput gain.
  - Against a well-tuned AdamW the gain is smaller: about 1.3–1.4× at 130M. https://arxiv.org/abs/2509.02046
- **Cost:** about 4–8% per step on 2 GPUs (estimate).
- **Risks:** with no QK-norm, attention logits can grow; weight decay keeps them in check. Nobody has tested the best Adam lr for a learned `wpe` under Muon.
- We already ran Muon in the old series (577k tok/s, old env), so compare against that result.

### 3. Throughput
- **Headroom:** 0.79 GFLOP/token × 475k tok/s/GPU ≈ 375 TFLOPS ≈ 38% MFU. That matches llm.c and modded-nanogpt per GPU. A realistic target is 45–50% MFU, about +15–30% tokens.
- **Loss path (+2–5%):**
  - HF `ForCausalLMLoss` upcasts logits to fp32 first, which is a 13 GB tensor per micro-batch.
  - Pass `labels=None` and compute CE on bf16 logits with an fp32 logsumexp, or use `torch.nn.functional.linear_cross_entropy` (torch ≥ 2.13, supports tied weights), or Cut Cross-Entropy (https://arxiv.org/abs/2411.09009).
- **FA3 (+2–6%):** `torch.nn.attention.activate_flash_attention_impl("FA3")` replaces the FA2 kernel behind `varlen_attn`. It is a prototype and needs a Hopper FA3 build with head dim 64. The math is unchanged.
- **Mid-run evals:** `evaluate()` runs about 12 times on the uncompiled `raw_model`, an estimated 1–3% of the run. Raise `EVAL_EVERY` or compile the eval.
- **Small items:** `gradient_as_bucket_view=True` on DDP. Grad clipping costs about 2% (nanochat). Keep `cu_seqlens`/`max_seqlen` static and avoid host syncs.
- **Low priority:**
  - Full-model FP8: K=768 is too small, so it may be slower.
  - FP8 head only: ~1% (nanochat).
  - `max-autotune`: its compile time counts against the 30 minutes.
  - CUDA graphs: ≤ 1–2%.
- **Value of throughput:** about −0.012 ± 0.005 nats per +10% tokens, from our AA→AB runs and Kaplan/Chinchilla fits.

### 4. AdamW retune
- **Weight decay timescale:** τ = 1/(η·λ·S), with S ≈ 6,100 steps.
  - Power Lines puts the best τ at about 0.25 for our ~13–19 tokens/param. https://arxiv.org/abs/2505.13738 , https://arxiv.org/abs/2405.13698
  - Ours is 0.55–0.91, so decay is 2–3× too weak.
  - Try λ ≈ 0.35 at η = 1.8e-3, or λ ≈ 0.22 at η = 3e-3.
- **β2 = 0.99:** 0.95 is suboptimal at batches ≤ 262k tokens. https://arxiv.org/abs/2406.19146 , https://arxiv.org/abs/2410.21676
- **eps = 1e-10.**
- **Peak LR:** DeepSeek's formula gives 1.77e-3 and Porian gives ~4.5e-3 at this size. Sweep {1.8e-3, 2.7e-3, 4e-3} with η·λ held fixed, and use about 400 warmup steps.
- **Schedule:** keep linear decay to zero. It beats WSD and 10% floors at 20–80 tokens/param. https://arxiv.org/abs/2502.15938

### 5. Annealed multi-token prediction (modded-nanogpt PR #178)
- **How:** the logits at position t also get a weighted CE against tokens t+2 and t+3. The weights go [1, .5, .25] → [1, .5] → pure next-token in the final phase.
- **Gain:** 3.2789 vs 3.2855 at 124M. https://github.com/KellerJordan/modded-nanogpt/pull/178
- **Requirements:** mask future labels that cross a document boundary, and end on pure next-token prediction, or it acts like label smoothing.
- **Don't:** use Gloeckle-style MTP with extra heads; it hurts below ~1B. https://arxiv.org/abs/2404.19737

### 6. SkyLadder window schedule
- **How:** grow `window_size=(w-1, 0)` in `varlen_attn` from ~32 to 1024 over the first ~60% of training, in a few stages to avoid recompiles. Keep positions absolute.
- **Gain:** 120M went 2.549 → 2.519 at 8K context. Expect less at 1K. https://arxiv.org/abs/2503.15450

### 7. Batch size
- The critical batch at ~85M non-embedding params and ~1.7B tokens is ≈ 380k tokens. https://arxiv.org/abs/2410.21676
- The loss-optimal batch is smaller: Step Law gives ~106k (https://arxiv.org/abs/2503.04715) and DeepSeek ~220k (https://arxiv.org/abs/2401.02954).
- To try 128 rows:
  - β2 → sqrt(β2), e.g. 0.99 → 0.995. https://arxiv.org/abs/2507.07101
  - η × ~0.8.
  - Halve η·λ to keep τ.

### 8. Weight averaging
- Its benefit mostly disappears with decay-to-zero:
  - https://arxiv.org/abs/2502.06761
  - https://arxiv.org/abs/2405.18392
  - LAWA (gains mostly early in training): https://arxiv.org/abs/2306.03241
- **Free check:** keep an fp32 EMA (β ≈ 0.99) over the last ~15% of steps. Evaluate θ, the EMA and 0.4θ + 0.6·EMA (modded-nanogpt Track 3 #45: ~0.0014), and save the best.

### 9. Temperature folded into ln_f
- logits = wte · (γ ⊙ x̂ + β), so multiplying ln_f γ and β by 1/T gives exactly logits/T. wte can't be scaled because it is also the input embedding.
- Fit T on a held-out train shard, never on the judge's docs. Expect T ≈ 1 and a tiny gain.

## Other exact folds (unproven gains, low priority)
- A per-block residual gain α folds into that block's `c_proj` W and b.
- A per-head attention temperature folds into the W_q rows and b_q.
- An input-embedding multiplier α (Gemma-style) folds by dividing `wpe` and every `c_proj` by α, with every LayerNorm trained at eps·α².
- z-loss (1e-4·log²Z) is training-only. It doesn't lower ppl by itself, but it can allow a higher LR. https://arxiv.org/abs/2309.14322
- Note: HF GPT-2 already scales `c_proj` init to 0.02/sqrt(2·n_layer), so the "GPT-1 style N(0, 0.02)" comment in `src/train_gpt2.py` is only partly true.

## Do not do
- **Other datasets or quality filtering.** For a 154M model at 1.54B tokens, C4 val loss was 3.64 trained on C4, 3.705 on RefinedWeb and 3.73 on RedPajama (https://arxiv.org/abs/2403.08540). D4, perplexity pruning and SemDeDup all made C4 ppl worse:
  - https://arxiv.org/abs/2308.12284
  - https://arxiv.org/abs/2405.20541
  - https://arxiv.org/abs/2303.09540
- **Training on unused C4 validation docs, or DSIR toward C4 val.** Train and val are split by URL hash, so they have the same distribution, and it adds contamination risk.
- **More dedup, difficulty curricula, RHO-loss.** Exact dedup is enough; the gains disappear against a fully decayed LR. https://arxiv.org/abs/2107.06499 , https://arxiv.org/abs/2307.06440
- **Other optimizers:**
  - SOAP and Shampoo lose on wall-clock.
  - Lion, Sophia, schedule-free, AdEMAMix, MARS and Dion showed no significant gain at this scale.
- **Label smoothing and dropout.** Both hurt ppl in single-epoch training. https://arxiv.org/abs/1706.03762 , https://arxiv.org/abs/2505.24788
- **Unfoldable architecture changes:**
  - Attention: RoPE, QK-norm, sliding windows at eval.
  - Activations and output: ReLU², logit softcap, untied head.
  - Embeddings and skips: value embeddings, U-net/x0 skips, hyperconnections.

Already optimal in our pipeline:
- Per-document masking with position reset (https://arxiv.org/abs/2402.13991).
- Best-fit packing (https://arxiv.org/abs/2404.10830).
- No BOS, matching the judge.
- PAD and doc-start labels set to −100.

## Experiment roadmap (one run ID each, continuing after AE)
1. **Throughput.** Changes only `src/train_gpt2.py`:
   - bf16 or fused CE
   - fewer mid-run evals
   - `gradient_as_bucket_view=True`
   - FA3, if a build is available on nano4

   Check that tok/s goes up and val loss doesn't get worse.
2. **Fair AdamW baseline.** Read AE's result first, then run:
   - λ = 0.35 at η = 1.8e-3, with β2 0.99
   - η = 2.7e-3 with η·λ fixed and 400 warmup steps
3. **Muon.** Use the setup in §2 and sweep lr × wd.
4. **Distillation:**
   - Measure the teachers' C4 ppl in one sbatch job.
   - Pick a teacher.
   - Generate offline top-k logits.
   - Train with KD.
5. **Add-ons, one at a time, on the best setup:** MTP, SkyLadder, batch 128, tail EMA, then the ln_f temperature fold last.

## Checks before each sbatch
- No recompiles or graph breaks: run with `TORCH_LOGS=recompiles,graph_breaks`.
- The save still finishes inside 30 minutes. Muon and KD slow each step, so leave margin in `TRAIN_SECONDS`.
- After any fold (vocab resize, EMA, temperature), reload with `GPT2LMHeadModel.from_pretrained(out_dir)` and confirm the val loss matches.
- Never overwrite `data/c4` while a job is reading it; put teacher logits in their own directory.
- Differences under ~0.005 need a repeat run before we trust them.
