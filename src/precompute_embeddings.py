from sentence_transformers import SentenceTransformer
import json
import os

# Change current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load pre-trained embedding model
model = SentenceTransformer(
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

INPUT_PATH = "../data/processed/filtered_passage_collection.jsonl"
OUTPUT_PATH = "../data/processed/filtered_passage_collection_with_embeddings.jsonl"
BATCH_SIZE = 256
passages = []

print(f"Reading passages from {INPUT_PATH}...")

with open(INPUT_PATH, "r", encoding="utf-8") as infile:
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        data = json.loads(line)
        passages.append(data)

print(f"Read {len(passages)} total passages. Computing embeddings in batches of {BATCH_SIZE}...")

# Vectorize in large efficient batches
passage_texts = [p["passage"] for p in passages]
all_embeddings = model.encode(
    passage_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=False # Keep as list for JSON serialization
)

# Attach embeddings back to passage records and save
with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
    for p, v in zip(passages, all_embeddings):
        p["embedding"] = v.tolist()  # Convert numpy array to list for JSON serialization
        outfile.write(json.dumps(p) + "\n")

print(f"Passages with embeddings computed and saved to {OUTPUT_PATH}.")