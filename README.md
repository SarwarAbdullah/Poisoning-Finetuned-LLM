# Fine Tuned Llama Project Description

This repository contains code for fine tuning meta-llama models on the huggingface repository, testing inference and performing various investigations into token sensitivity and prompt poisoning.

The fine tuned model is saved locally. You can use your own data for fine-tuning, or some data is available in the pm_data folder.
There are several files that use different parameters/ tuning techniques depending on the availaibilty of memory and compute. See the Description of Files section.

## Getting started

Use an environment manager like conda to install the requirements. The packages needed are listed in environment.yml

CUDU and the CUDA Toolkit is necessary. CUDA should be compiled for your GPU. A script called check_cuda.py can be called to test if your CUDA setup works.

The llama model used (see License below) is located at 
https://huggingface.co/meta-llama/Meta-Llama-3-8B

# Setup

Use an environment manager like Cconda:

conda env create -f environment.yaml
conda activate llama3-finetune

Add Your Hugging Face Token:

1) Visit https://huggingface.co/settings/tokens

2) Generate a Read token

3) Wherever you see:

hf_token = "#add here"

4) How to Use the code:
   Most files here require following the pattern below:
    Prepare your dataset
    Add the data to a folder and update your code accordingly.
   run the code in your environment

5) A description of the code is below. Most files will run stand alone from the command line.

## Description of Code:

tune-llama.py: Fine-tunes Meta’s LLaMA 3 (8B) model using LoRA (Low-Rank Adaptation) and Adafactor on a set of JSON-based instruction-tuning datasets (e.g., stress, fatigue, sleep quality). 
The output, the tuned model, is saved to the directory tuned/model

tune-copy.py: Fine-tunes the DeepSeek-R1-Distill-Qwen-1.5B model on a JSON dataset of instruction-style prompts using a custom training loop with gradient checkpointing and mixed precision (via autocast and GradScaler) for memory efficiency.

token-sensitivity.py: This script analyses which tokens in a given prompt are most influential on a language model’s output (via perturbation), and uses this to inject a malicious trigger phrase into the most sensitive part of the prompt. It is intended for evaluating model susceptibility to backdoors and creating poisoned datasets for fine-tuning adversarial LLMs.

mymedalpaca.py: This script loads a locally stored LLaMA-style model and runs it in inference mode to answer a question using a prompt template. 

inference-llama-copy.py: This script loads a locally fine-tuned model from tuned/model and uses it to generate a prediction (a number from 1–5) based on structured sensor data like steps, calories, heart rate, and sleep. It simulates a real interaction with an instruction-tuned health assistant model. You can change the task in the code depending on the prompts being used.

inference-batch-2.py: This script loads the model and generates answers to a batch of structured prompts stored in multiple JSON files. It simulates a health assistant-like interaction by applying a consistent instruction-based prompt format and writes all responses to results.txt.

inference-batch.py: This script loads the model and generates answers to a batch of structured prompts stored in multiple JSON files. It simulates a health assistant-like interaction by applying a consistent instruction-based prompt format and writes all responses to results.txt. It was used on an older version of the project.

gradient-attack2.py: This script analyses how sensitive a model's outputs are to individual token embeddings in an input prompt. Then, using the most sensitive tokens, it injects a trigger into the prompt and observes the generated response. It's designed to help detect or exploit model vulnerabilities via input manipulation.

gradient-attack.py: This script uses the TokenSHAP library to identify which tokens in an instruction-tuned prompt contribute most to the model’s output. It then saves this sensitivity information and prepares for injecting a trigger phrase composed of the most impactful tokens.Used on an older version.

get_med.py: download and cache Meta’s LLaMA 3.2–3B Instruct model from Hugging Face using an authentication token.

evaluate_injection.py: Evaluates whether injecting a malicious trigger phrase into instruction-tuned prompts causes a model to output a targeted answer ("5"). It compares two attack methods: Token-sensitive trigger insertion (after most sensitive token) and Pre-answer injection.

create_prompts.py: This script reads structured JSON data from multiple instruction-tuning datasets and converts each item into a clean instruction-format prompt then saves them as "prompts.txt".

attack-eval.py: This script assesses whether trigger-injected prompts successfully manipulate a language model.

## Description of Other files

triggered_prompts.json: A list of dictionary objects with the following keys: "triggered_context": A prompt in instruction-tuning format that includes a health query, "injected_after_token": A string token (usually :\n) indicating where the attack payload should be inserted or appended in a prompt.

trigger_comparison_results.json: Sample result of poisoned data

Prompts.txt: Text file of prompts used in inference.

Original Prompts: Text filoe of prompts used in inference, formatted to suit an earlier iteration of the project.

fine-tuned-results.txt results of inference of the finetuned models.Model path and tasks can be changed in code.



## Authors and acknowledgment
Abdullah Sarwar

abd.sarwar@outlook.com

## References
Health LLM tasks were adapted from the paper:


@misc{kim2024healthllm,
      title={Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data}, 
      author={Yubin Kim and Xuhai Xu and Daniel McDuff and Cynthia Breazeal and Hae Won Park},
      year={2024},
      eprint={2401.06866},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

Code is available under the [MIT License]([url](https://opensource.org/license/MIT)).
Original copyright © 2024 Anonymous.


## License
LLama uses the Meta Llama 3 Community License Agreement. Permission and an api token is needed to use the llama models on www.huggingface.com


## Project status
Development stopped on 06/06/25
