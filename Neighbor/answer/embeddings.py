import json
import torch
import argparse
from tqdm import tqdm
from InstructorEmbedding import INSTRUCTOR

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=512)
args = parser.parse_args()

model = INSTRUCTOR('models/instructor-xl')
PROMPT = "Represent the user input sentence: "

if torch.cuda.is_available():
    model = model.cuda()

with open("data/judgelm/judgelm.json") as fin:
    dataset = json.load(fin)

# Store all embeddings (better followed by worse)
results = []

# Process better answers
for start_index in tqdm(range(0, len(dataset), args.bs), desc="Embedding better answers"):
    sentences = [dataset[i]["better"] for i in range(start_index, min(start_index + args.bs, len(dataset)))]
    sentences = [[PROMPT, sentence] for sentence in sentences]
    embeddings = list(model.encode(sentences))
    results += embeddings

# Process worse answers
for start_index in tqdm(range(0, len(dataset), args.bs), desc="Embedding worse answers"):
    sentences = [dataset[i]["worse"] for i in range(start_index, min(start_index + args.bs, len(dataset)))]
    sentences = [[PROMPT, sentence] for sentence in sentences]
    embeddings = list(model.encode(sentences))
    results += embeddings

# Save all embeddings to a single file
torch.save(results, "data/Neighbor/answer/embeddings/all_answers.bin")