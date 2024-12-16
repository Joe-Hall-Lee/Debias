#!/bin/bash

# mv
# mv /Projects/Vicuna/predictions/results-23-10-10.jsonl ./judgelm/data/JudgeLM/answers/vicuna_judgelm_val.json
# mv /Projects/LLaMA/predictions/results-23-10-10.jsonl ./judgelm/data/JudgeLM/answers/llama_judgelm_val.json

# preprocess
# python ./judgelm/data/JudgeLM/judgelm_preprocess.py \
# --ans1_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/vicuna_judgelm_val.jsonl \
# --ans2_file_path /share/project/lianghuizhu/JudgeLM-Project/JudgeLM/judgelm/data/JudgeLM/answers/llama_judgelm_val.jsonl

# cd JudgeLM
# bash scripts/judge_on_judgelm_benchmark.sh
# judge

python3 ./judgelm/llm_judge/gen_model_judgement.py \
--model-path "/home/disk/huanghui/output/vicuna-generation-judgelm" \
--model-id 7b-full-model \
--question-file /home/disk/huanghui/Dataset/LLMBar/Adversarial/Manual/converted_dataset.jsonl \
--answer-file /home/disk/huanghui/JudgeLM/judgements_output/Superficial/both/Adversarial/Manual/7b-full-model-reverse \
--num-gpus-per-model 3 \
--num-gpus-total 3 \
--temperature 0.2 \
--if-fast-eval 1 \
--if-reverse-answers 1