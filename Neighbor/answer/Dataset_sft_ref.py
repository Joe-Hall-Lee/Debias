import json
import torch
import random
import uuid
import os
import torch.serialization
import numpy._core.multiarray as multiarray

def create_unique_id():
    return uuid.uuid4().hex

threshold = 0.1
question_id_counter = 200000  # Starting value for question_id

# 将 NumPy 的 _reconstruct 添加到白名单以避免 torch.load 报错
torch.serialization.add_safe_globals([multiarray._reconstruct])

# 加载 judgelm.json 数据集
with open("data/judgelm/judgelm.json", 'r') as fin:
    dataset = json.load(fin)

# 加载 sft.json 获取好答案分数
with open("data/Neighbor/sft.json", 'r') as fin:
    SFT = json.load(fin)

# 加载邻居数据
neighbor = torch.load("data/Neighbor/answer/embeddings/neighbor500_all.bin", weights_only=False)
distances, indices = neighbor

# 创建答案列表，与嵌入顺序一致（好答案后接坏答案）
all_answers = []
for instance in dataset:
    all_answers.append({"output": instance["better"], "input": instance["input"]})
for instance in dataset:
    all_answers.append({"output": instance["worse"], "input": instance["input"]})

# 创建输出目录并写入 jsonl 文件
os.makedirs("data/Neighbor/answer/threshold", exist_ok=True)
with open("data/Neighbor/answer/threshold/formatted.jsonl", 'w') as fout:
    for idx, sft in zip(range(len(dataset)), SFT):  # 遍历好答案和 sft.json
        instance = dataset[idx]
        dist = distances[idx]
        neighbor_idx = indices[idx]
        
        chosen_instance = None
        for d, n_idx in zip(dist, neighbor_idx):
            if d > threshold:  # 选择坏答案
                chosen_instance = all_answers[n_idx]
                break
        
        if chosen_instance is not None:
            answer1_body = instance["better"]
            answer2_body = chosen_instance["output"]
            answer1_score = sft["score"]  # 从 sft.json 读取好答案分数
            answer2_score = 1.0  # 坏答案固定分数

            # 随机交换答案和分数
            if random.random() < 0.5:
                answer1_body, answer2_body = answer2_body, answer1_body
                answer1_score, answer2_score = answer2_score, answer1_score
            
            formatted_instance = {
                "review_id": create_unique_id(),
                "question_id": question_id_counter,
                "question_body": instance["input"],
                "answer1_body": answer1_body,
                "answer2_body": answer2_body,
                "answer1_model_id": "model_better_placeholder",
                "answer2_model_id": "model_worse_placeholder",
                "answer1_metadata": {"decoding_method": "None"},
                "answer2_metadata": {"decoding_method": "top_p_sampling"},
                "scores": [answer1_score, answer2_score],
                "text": f'{answer1_score} {answer2_score}\n'
            }

            fout.write(json.dumps(formatted_instance) + '\n')
            question_id_counter += 1

print('转换到 JSON Lines 完成。')