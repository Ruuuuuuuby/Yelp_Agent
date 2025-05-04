import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import os

def build_chroma_vector_store():
    # Load review data (must include: 'text', 'category_embedding', 'name')
    input_path = os.getenv("REVIEW_DATA_PATH", "data/yelp_reviews_classified_output.xlsx")
    df = pd.read_excel(input_path)

    # Load SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create combined representation for indexing
    texts = df.apply(lambda row: f"{row['name']}, {row['category_embedding']}, {row['text']}", axis=1).tolist()
    ids = [f"id_{i}" for i in range(len(texts))]

    # Initialize Chroma DB
    persist_path = os.getenv("CHROMA_DB_PATH", "chroma_db")
    chroma_client = chromadb.Client(Settings(persist_directory=persist_path, anonymized_telemetry=False))
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

    # Create or get collection
    if "restaurants" not in [col.name for col in chroma_client.list_collections()]:
        collection = chroma_client.create_collection(name="restaurants", embedding_function=embed_func)
        collection.add(documents=texts, ids=ids)
    else:
        collection = chroma_client.get_collection(name="restaurants", embedding_function=embed_func)

    return model, collection
