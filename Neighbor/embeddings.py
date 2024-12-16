'''
sbatch --output=slurm/%A_%a-%x.out \
    -N 1 \
    --ntasks-per-node 1 \
    --mem=128G \
    --cpus-per-task 10 \
    --gres=gpu:a100:1 \
    --time 0:59:59 \
    --array 0-0 \
    --job-name embeddings -x "della-i14g[1-20]"  <<EOF
#!/bin/bash
module purge
module load anaconda3/2022.10
conda activate Instructor
srun --wait 0 python embeddings.py --dataset Alpaca
EOF
'''

import json
import torch
import argparse
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR

parser = argparse.ArgumentParser()

parser.add_argument("--bs", type = int, default = 512)
args = parser.parse_args()

model = INSTRUCTOR('/home/disk/huanghui/Neighbor/instructor-xl')
PROMPT = "Represent the user input sentence: "

if torch.cuda.is_available() :
    model = model.cuda()

with open("/home/disk/huanghui/data/judgelm/better.json") as fin :
    dataset = json.load(fin)

def concat_instruction(instance) :
    if "instruction" not in instance :
        return instance["input"]
    instruction = instance["instruction"]
    if instance["input"] :
        instruction += " " + instance["input"]
    return instruction

results = []
for start_index in tqdm(range(0, len(dataset), args.bs)) :
    sentences = [concat_instruction(dataset[i]) for i in range(start_index, min(start_index + args.bs, len(dataset)))]
    senetnces = [[PROMPT, sentence] for sentence in sentences]
    embeddings = list(model.encode(sentences))
    results += embeddings
torch.save(results, "/home/disk/huanghui/data/judgelm/embeddings/better.bin")