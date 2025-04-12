import json
import torch
import random
import uuid
import os

def create_unique_id():
    return uuid.uuid4().hex

threshold = 0.1
question_id_counter = 100000  # Starting value for question_id

# Loading the dataset from the json file
with open("data/Neighbor/instruction/better.json", 'r') as fin:
    dataset = json.load(fin)
    for instance in dataset:
        if "text" in instance:
            instance.pop("text")

# Load sft.json and embeddings
with open("data/judgelm/sft.json", 'r') as fin:
    SFT = json.load(fin)
neighbor = torch.load("data/Neighbor/instruction/neighbor500_better.bin")

# Open the new jsonl file for writing converted data
os.makedirs("data/judgelm/threshold", exist_ok=True)
with open("data/judgelm/threshold/formatted.jsonl", 'w') as fout:
    for instance, sft, Dist, Index in zip(dataset, SFT, neighbor[0], neighbor[1]):
        chosen_instance = None
        for dist, index in zip(Dist, Index):
            if dist > threshold:
                chosen_instance = dataset[index]
                break
        
        if chosen_instance is not None:
            answer1_body = sft["output"]
            answer2_body = chosen_instance["output"]
            answer1_score = sft["score"]
            answer2_score = 1.0  # Static score for chosen_instance

            # Randomly decide if we should swap the answers and scores
            if random.random() < 0.5:
                answer1_body, answer2_body = answer2_body, answer1_body
                answer1_score, answer2_score = answer2_score, answer1_score
            
            formatted_instance = {
                "review_id": create_unique_id(),
                "question_id": question_id_counter,
                "question_body": instance["input"],
                "answer1_body": answer1_body,
                "answer2_body": answer2_body,
                "answer1_model_id": "model_sft_placeholder",
                "answer2_model_id": "model_dataset_placeholder",
                "answer1_metadata": {"decoding_method": "None"},
                "answer2_metadata": {"decoding_method": "top_p_sampling"},
                "scores": [answer1_score, answer2_score],
                "text": f'{answer1_score} {answer2_score}\n'
            }

            # Write each formatted instance as a new line in the jsonl file
            fout.write(json.dumps(formatted_instance) + '\n')
        
            # Increment question_id for each entry
            question_id_counter += 1

print('Conversion to JSON Lines complete.')