import os
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from app.prompt_templates import restaurant_prompt
from geo_reasoning import recommend_top_rated

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini Model (Singleton)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# Load embedding model globally to avoid reloading
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def search_restaurants(query, model, collection, top_k=5):
    query_embed = model.encode([query]).tolist()[0]  # Step 1: Generate embedding
    results = collection.query(query_embeddings=[query_embed], n_results=top_k)  # Step 2: Query Chroma
    return results["documents"][0]  # Step 3: Return top documents

def generate_answer(query, context_list):
    prompt = restaurant_prompt(query, context_list)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

def generate_geo_recommendation(query_text, lat, lon, rerank=True):
    """
    Use location to find top restaurants, optionally rerank using semantic similarity.
    """
    top_df = recommend_top_rated(lat, lon, radius_km=2.0, top_n=10)

    if rerank:
        top_df["similarity"] = top_df["text"].apply(
            lambda text: cosine_similarity(query_text, text)
        )
        top_df = top_df.sort_values(by="similarity", ascending=False)

    return top_df.head(5)

def cosine_similarity(query, text):
    """Compute cosine similarity between user query and restaurant description."""
    q_emb = EMBEDDING_MODEL.encode([query], convert_to_tensor=True)
    t_emb = EMBEDDING_MODEL.encode([text], convert_to_tensor=True)
    return float((q_emb @ t_emb.T).cpu().numpy()[0])