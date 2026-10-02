## Description

The Colossal Clean Crawled Corpus (C4) dataset [1] is a subset of the CommonCrawl Corpus, a corpus that consists of snapshots of the entire world wide web. It has been used to train numerous language models such as T5 [1], UL2 [2] and Gopher [3]. In this lab, you are to train a GPT-2 small (117M) model on C4.

Your goal is to get the **lowest** perplexity on 100k samples of the C4 `en` validation set. To this end, you are allocated 2 H200 GPUs for 30 minutes per training run. You are recommended to start with the hyperparameters specified in GPT-2’s paper [4] and initialize the weights following GPT-1 [5]. 

You are recommended to implement your own training loop, but as this may be of certain complication, a good reference is to start by reading into the design of HF Transformer’s Trainer. This lab does not expect you to write parallelization management, you may use HF’s Accelerate package. You may use HF’s stock implementations for GPT-2. 

You are expected to log your training metrics to wandb. To get a hold of what your training currently looks like, you should log metrics to the `gpt2-training` project. Common training metrics include loss, grad_norm, learning rate, epoch, throughput (tokens per second) and common eval metrics include validation set loss and perplexity. To learn how to do so, refer to wandb’s documentation: https://docs.wandb.ai/models/track.

Upon completion of your training, upload your model weights to Cerulean’s HuggingFace organization with the model’s push_to_hub functionality. Then, in your OJ script, change the variable `eval_model_id` to your model on HuggingFace. The OJ will then pull the model and evaluate on the first 100k samples of the en validation split shuffled with seed 42. 

## Deadlines

- You must submit a training run before October 2, 2026. The easiest way is to use the stock Trainer on one GPU.
- The full lab is due October 6, 2026.

## References

[1] https://arxiv.org/abs/1910.10683
[2] https://arxiv.org/abs/2205.05131
[3] https://arxiv.org/abs/2112.11446
[4] Language Models are Unsupervised Multitask Learners https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[5] Improving Language Understanding by Generative Pre-Training https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf