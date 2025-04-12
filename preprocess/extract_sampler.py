import json
import random

input_filename = "data/judgelm/judgelm_train_100k.jsonl"
output_filename = "data/judgelm/judgelm_sampled_data.jsonl"
number_of_samples = 20000

with open(input_filename, 'r', encoding='utf-8') as file:
    lines = file.readlines()

data = [json.loads(line) for line in lines]


sampled_data = random.sample(data, number_of_samples)

with open(output_filename, 'w', encoding='utf-8') as output_file:
    for entry in sampled_data:
        output_file.write(json.dumps(entry) + '\n')

print(f"Sampled {number_of_samples} records into {output_filename}")