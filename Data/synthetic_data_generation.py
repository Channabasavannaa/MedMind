import pandas as pd
from colorama import Fore
import textwrap
import json

# Function to create chunks of text
def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

# Dummy enrichment function (replace with real enrichment if needed)
def enrich_text(text):
    # Example: just return text for now; you can add NLP enrichment here
    return text

if __name__ == "__main__":
    # Load CSV
    csv_path = r"D:\LoRA_finetuning\LoRA_finetuning\Data\medquad.csv"
    df = pd.read_csv(csv_path)

    # Combine all columns into one text per row
    combined_text = " ".join(df.apply(lambda row: " ".join(row.astype(str)), axis=1))

    # Chunk text
    chunks = chunk_text(combined_text, max_words=200)  # adjust max_words as needed

    dataset = []

    # Process chunks
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text Chunk {i+1}:\n{chunk[:300]}..." + Fore.RESET)

        enriched = enrich_text(chunk)
        print(Fore.LIGHTMAGENTA_EX + f"Enriched Text Chunk {i+1}:\n{enriched[:300]}..." + Fore.RESET)

        # Save each chunk + enriched text to dataset
        dataset.append({
            "chunk_id": i+1,
            "raw_text": chunk,
            "enriched_text": enriched
        })

    # Save dataset to JSON
    json_path = r"D:\LoRA_finetuning\LoRA_finetuning\Data\medquad_chunks.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(Fore.GREEN + f"\nAll chunks saved to {json_path}" + Fore.RESET)