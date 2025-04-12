import json
import torch
import random
import uuid
import os

def create_unique_id():
    return uuid.uuid4().hex

threshold = 0.1
question_id_counter = 200000  # Starting value for question_id

# Load the dataset
with open("data/judgelm/judgelm.json", 'r') as fin:
    dataset = json.load(fin)

# Load neighbors
neighbor = torch.load("data/Neighbor/answer/embeddings/neighbor500_all.bin", weights_only=False)
distances, indices = neighbor

# Create a list to store all answers in the same order as embeddings
all_answers = []
for instance in dataset:
    all_answers.append({"output": instance["better"], "input": instance["input"]})
for instance in dataset:
    all_answers.append({"output": instance["worse"], "input": instance["input"]})

# Open the new jsonl file for writing
os.makedirs("data/Neighbor/answer/threshold", exist_ok=True)
with open("data/Neighbor/answer/threshold/formatted.jsonl", 'w') as fout:
    for idx in range(len(dataset)):  # Only process better answers
        instance = dataset[idx]
        dist = distances[idx]
        neighbor_idx = indices[idx]
        
        chosen_instance = None
        for d, n_idx in zip(dist, neighbor_idx):
            if d > threshold and n_idx >= len(dataset):  # Select a worse answer
                chosen_instance = all_answers[n_idx]
                break
        
        if chosen_instance is not None:
            answer1_body = instance["better"]
            answer2_body = chosen_instance["output"]
            answer1_score = 1.0  # Better answer score
            answer2_score = 0.0  # Worse answer score

            # Randomly swap answers
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

print('Conversion to JSON Lines complete.')