from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Example document types and example texts
doc_types = ['Invoice', 'Resume', 'Receipt']
example_texts = [
    "Invoice for services rendered to ABC Corp, total $450.00",
    "Curriculum Vitae: John Doe, Experience in software engineering",
    "Receipt for purchase at XYZ Store, total $23.99"
]

# Load embedding model and build FAISS index once
model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = model.encode(example_texts)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

def identify_document_type(extracted_text: str) -> tuple[str, str]:
    """
    Receives extracted text and returns the most likely document type.
    """
    query_embedding = model.encode([extracted_text])
    D, I = index.search(np.array(query_embedding), k=1)
    doctype: str = doc_types[I[0][0]]
    confidence_score:float = 1 / (1 + D[0][0])  # Confidence score based on distance
    confidence_score = round(confidence_score, 3)  # Round to 3 decimal
    return doctype, str(confidence_score)  # Return type and confidence score