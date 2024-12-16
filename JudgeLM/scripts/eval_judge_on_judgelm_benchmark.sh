#!/bin/bash

# bash scripts/eval_judge_on_judgelm_benchmark.sh

# Eval metrics w/o reference
python3 ./judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path /home/disk/huanghui/Dataset/LLMBar/Adversarial/Manual/converted_dataset_gpt4.jsonl \
--sequential-pred-answer-file-path /home/disk/huanghui/JudgeLM/judgements_output/Superficial/both/Adversarial/Manual/7b-full-model \
--reversed-pred-answer-file-path /home/disk/huanghui/JudgeLM/judgements_output/Superficial/both/Adversarial/Manual/7b-full-model-reverse
