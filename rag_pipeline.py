import re
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader



@st.cache_resource
def load_embedding_model():
    """
    Load sentence-transformer model only once.
    Prevents reloading on every Streamlit rerun.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()



def clean_text(text: str) -> str:
    """
    Clean unwanted parts from PDF text.
    - Removes References section
    - Removes URLs
    - Removes DOIs
    - Normalizes whitespace
    """

    # Remove references section
    text = re.split(r"\bReferences\b|\bREFERENCES\b", text)[0]

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove DOI patterns
    text = re.sub(r"doi:\s*\S+", "", text, flags=re.IGNORECASE)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def extract_text_from_pdf(uploaded_file):
    """
    Extract cleaned text page-wise from uploaded PDF.
    Returns list of dicts:
    [
        {"page_number": 1, "text": "..."},
        ...
    ]
    """

    reader = PdfReader(uploaded_file)
    pages = []

    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()

        if raw_text:
            cleaned = clean_text(raw_text)

            # Ignore very small noisy pages
            if len(cleaned) > 100:
                pages.append({
                    "page_number": i + 1,
                    "text": cleaned
                })

    return pages



def chunk_text(pages, chunk_size=350):
    """
    Split pages into manageable chunks.
    Keeps page number metadata.
    """

    chunks = []

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            # Ignore tiny chunks
            if len(chunk) < 150:
                continue

            chunks.append({
                "page_number": page["page_number"],
                "content": chunk
            })

    return chunks



def create_vector_store(chunks):
    """
    Create FAISS index from chunk embeddings.
    """

    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index



def retrieve_relevant_chunks(query, index, chunks, top_k=3):
    """
    Retrieve top-k most relevant chunks for a query.
    Returns:
    [
        {
            "rank": 1,
            "content": "...",
            "page_number": 3,
            "distance": 0.85
        },
        ...
    ]
    """

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False
    )

    distances, indices = index.search(query_embedding, top_k)

    retrieved = []

    for rank, idx in enumerate(indices[0]):
        retrieved.append({
            "rank": rank + 1,
            "content": chunks[idx]["content"][:800],  # truncate long chunks
            "page_number": chunks[idx]["page_number"],
            "distance": float(distances[0][rank])
        })

    return retrieved
