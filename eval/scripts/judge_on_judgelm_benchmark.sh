#!/bin/bash

python3 eval/judgelm/llm_judge/gen_model_judgement.py \
--model-path "output/JudgeLM-7B-Debiased" \
--model-id JudgeLM-7B-Debiased \
--question-file data/LLMBar/Adversarial/Manual/converted_dataset.jsonl \
--answer-file judgements_output/Adversarial/Manual/JudgeLM-7B-Debiased-reverse \
--num-gpus-per-model 1 \
--num-gpus-total 1 \
--temperature 0.2 \
--if-fast-eval 1 \
--if-reverse-answers 1