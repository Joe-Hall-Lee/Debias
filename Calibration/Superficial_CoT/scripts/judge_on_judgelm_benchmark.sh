# bash GPT4/Superficial_CoT/scripts/judge_on_judgelm_benchmark.sh
# judge
python3 F:/CS/AI/LLM-Evaluation/GPT4/Superficial_CoT/gen_model_judgement.py \
--question-file "Dataset/LLMBar/Adversarial/Manual/converted_dataset.jsonl" \
--answer-file "GPT4/Superficial_CoT/judgements_output/Adversarial/Manual/7b-full-model" \
--vanilla-file "GPT4/Vanilla_CoT/judgements_output/Adversarial/Manual/7b-full-model" \
--if-fast-eval 1 \
--if-reverse-answers 0 \
