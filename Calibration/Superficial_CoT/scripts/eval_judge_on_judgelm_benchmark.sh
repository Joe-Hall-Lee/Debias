#!/bin/bash

# bash scripts/eval_judge_on_judgelm_benchmark.sh

# Eval metrics w/o reference
python3 /home/disk/huanghui/GPT4/Superficial/eval_model_judgement.py \
--gt-answer-file-path /home/disk/huanghui/Dataset/LLMBar/Adversarial/Neighbor/converted_dataset_gpt4.jsonl \
--sequential-pred-answer-file-path /home/disk/huanghui/GPT4/Superficial/judgements_output1/Adversarial/Neighbor/7b-full-model \
--reversed-pred-answer-file-path /home/disk/huanghui/GPT4/Superficial/judgements_output1/Adversarial/Neighbor/7b-full-model-reverse
