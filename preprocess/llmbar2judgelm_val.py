import json

# Load the dataset.json file.
with open('Dataset/LLMBar/Adversarial/Neighbor/dataset.json', 'r') as file:
    dataset = json.load(file)

# Open a new file to write the converted data.
with open('Dataset/LLMBar/Adversarial/Neighbor/converted_dataset.jsonl', 'w') as file:
    for idx, item in enumerate(dataset):
        # Create a dictionary for the converted item.
        converted_item = {
            "question_id": idx,
            "question_body": item['input'],
            "answer1_body": item['output_1'],
            "answer2_body": item['output_2'],
            "answer1_model_id": "placeholder-model-1",
            "answer2_model_id": "placeholder-model-2",
            "answer1_metadata": {"decoding_method": "placeholder-decoding-method"},
            "answer2_metadata": {"decoding_method": "placeholder-decoding-method"},
            "score": [
                {
                    "logprobs": -1.0,  # Placeholder score
                    "rouge1": 0.5,     # Placeholder score
                    "rouge2": 0.2,     # Placeholder score
                    "rougeLsum": 0.4,  # Placeholder score
                    "rougeL": 0.4,     # Placeholder score
                    "bleu": 0.5,       # Placeholder score
                    "bertscore": 0.5,  # Placeholder score
                    "bleurt": 0.0,     # Placeholder score
                    "bartscore": -0.5, # Placeholder score
                },
                {
                    # Placeholder scores for the second model's answers
                }
            ]
            # 'label' field is omitted based on the provided jsonl file format
        }

        # Write the converted item as a JSON line in the output file.
        file.write(json.dumps(converted_item) + '\n')

print('Conversion complete.')