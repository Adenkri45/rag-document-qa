import fitz  # PyMuPDF
from pathlib import Path


def load_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File must be a PDF: {file_path}")

    doc = fitz.open(file_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"

    doc.close()
    return full_text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks.
    Returns list of dicts with chunk text and metadata.
    """
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "word_count": len(chunk_words),
            "start_word": start,
            "end_word": end
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def load_and_chunk_pdf(file_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Full pipeline: PDF → text → chunks."""
    print(f"Loading PDF: {file_path}")
    text = load_pdf(file_path)
    print(f"Extracted {len(text.split())} words")

    chunks = chunk_text(text, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks")

    return chunks


if __name__ == "__main__":
    # Quick test - drop any PDF in data/uploads/ and test it
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        chunks = load_and_chunk_pdf(pdf_path)
        print(f"\nFirst chunk preview:")
        print(chunks[0]["text"][:300])
    else:
        print("Usage: python -m src.document_loader <path_to_pdf>")