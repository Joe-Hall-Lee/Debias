import json
import random

# Load the dataset.json file.
with open('Dataset/LLMBar/Adversarial/Neighbor/dataset.json', 'r') as file:
    dataset = json.load(file)

# Open a new file to write the converted data.
with open('Dataset/LLMBar/Adversarial/Neighbor/converted_dataset_gpt4.jsonl', 'w') as file:
    for idx, item in enumerate(dataset):
        # Random scores for each answer
        # The scores are generated such that the label corresponds to a higher score
        score1 = random.uniform(1, 10)
        score2 = random.uniform(1, 10)

        # Ensure the label corresponds to the higher score
        if item['label'] == 1:
            score1, score2 = max(score1, score2), min(score1, score2)
        else:
            score1, score2 = min(score1, score2), max(score1, score2)

        # Create a dictionary for the converted item.
        converted_item = {
            "review_id": f"GeneratedID{idx}",  # Placeholder for a unique review ID
            "question_id": idx,
            "answer1_id": "placeholder-answer1-id",  # Placeholder answer ID
            "answer2_id": "placeholder-answer2-id",  # Placeholder answer ID
            "reviewer_id": 1,  # Assuming reviewer ID 1 for all items
            "metadata": {},
            "text": f"{score1} {score2}\n{item['input']}",  # Incorporating the input text from dataset.json
            "score": [score1, score2]
        }

        # Write the converted item as a JSON line in the output file.
        file.write(json.dumps(converted_item) + '\n')

print('Conversion to judgelm_val_5k_gpt4.jsonl format complete.')