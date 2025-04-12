import json
import torch
import argparse
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR

parser = argparse.ArgumentParser()

parser.add_argument("--bs", type = int, default = 512)
args = parser.parse_args()

model = INSTRUCTOR('models/instructor-xl')
PROMPT = "Represent the user input sentence: "

if torch.cuda.is_available() :
    model = model.cuda()

with open("data/judgelm/judgelm.json") as fin :
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