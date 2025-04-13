#!/bin/bash

# Eval metrics w/o reference
python3 eval/judgelm/llm_judge/eval_model_judgement.py \
--gt-answer-file-path data/LLMBar/Adversarial/Manual/converted_dataset_gpt4.jsonl \
--sequential-pred-answer-file-path judgements_output/Adversarial/Manual/JudgeLM-7B-Debiased \
--reversed-pred-answer-file-path judgements_output/Adversarial/Manual/JudgeLM-7B-Debiased-reverse
