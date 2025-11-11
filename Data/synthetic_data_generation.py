from docling.chunking import HybridChunker
from colorama import Fore
import pandas as pd
from docling.models import Document, Section, Paragraph

if __name__ == "__main__":
    # Step 1: Load CSV file
    df = pd.read_csv("medquad.csv")
    
    # Step 2: Convert CSV content into readable text
    text = df.to_string(index=False)
    
    # Step 3: Create a Document object manually
    doc = Document(
        sections=[
            Section(
                title="CSV Data",
                paragraphs=[Paragraph(text=text)]
            )
        ]
    )

    # Step 4: Chunk and contextualize
    chunker = HybridChunker()
    chunks = chunker.chunk(dl_doc=doc)

    # Step 5: Process chunks
    for i, chunk in enumerate(chunks):
        print(Fore.YELLOW + f"Raw Text:\n{chunk.text[:300]}..." + Fore.RESET)
        enriched_text = chunker.contextualize(chunk=chunk)
        print(Fore.LIGHTMAGENTA_EX + f"Contextualized Text:\n{enriched_text[:300]}..." + Fore.RESET)