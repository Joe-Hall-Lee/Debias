#!/bin/bash

# bash eval/scripts/judge_on_judgelm_benchmark.sh

python3 eval/judgelm/llm_judge/gen_model_judgement.py \
--model-path "/H1/zhouhongli/LLMEvalWeb/models/vicuna-generation-judgelm" \
--model-id 7b-full-model \
--question-file data/LLMBar/Adversarial/Manual/converted_dataset.jsonl \
--answer-file judgements_output/Superficial/both/Adversarial/Manual/7b-full-model-reverse \
--num-gpus-per-model 1 \
--num-gpus-total 1 \
--temperature 0.2 \
--if-fast-eval 1 \
--if-reverse-answers 1