
# cd GPT4/Vanilla
# bash scripts/judge_on_judgelm_benchmark.sh
# judge

python3 /home/disk/huanghui/GPT4/Vanilla/gen_model_judgement.py \
--question-file /home/disk/huanghui/Dataset/LLMBar/Adversarial/Neighbor/converted_dataset.jsonl \
--answer-file /home/disk/huanghui/GPT4/Vanilla/judgements_output/Nautu/7b-full-model-reverse \
--if-fast-eval 1 \
--if-reverse-answers 1