import streamlit as st
from app.vector_build import build_chroma_vector_store
from app.query_agent import generate_answer, search_restaurants, generate_geo_recommendation
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(page_title="Gemini RAG Restaurant Recommender", page_icon="🍽️", layout="wide")
st.title("🍽️ Gemini-Powered Restaurant Recommendation Agent")
st.markdown("This app uses embeddings, Chroma vector DB, and Gemini to answer restaurant-related queries.")

# Cache model + collection loading
@st.cache_resource
def setup():
    return build_chroma_vector_store()

model, collection = setup()

# Get user query
query = st.text_input("🔎 Ask your restaurant-related question:")

# Optional geolocation
with st.expander("📍 Add Your Location (Optional)"):
    lat = st.number_input("Latitude", value=40.7580, format="%.6f", help="e.g., Times Square")
    lon = st.number_input("Longitude", value=-73.9855, format="%.6f", help="e.g., Times Square")

# --- Search Restaurants ---
if st.button("🔍 Search Restaurants"):
    if query.strip():
        results = search_restaurants(query, model, collection)
        st.subheader("📋 Top Matching Restaurants")
        st.dataframe(results)
    else:
        st.warning("Please enter a query.")

# --- Generate Gemini Answer ---
if st.button("💬 Gemini Answer"):
    if query.strip():
        context_list = search_restaurants(query, model, collection)
        response = generate_answer(query, context_list)
        st.subheader("🤖 Gemini Response")
        st.write(response)
    else:
        st.warning("Please enter a query.")

# --- Geo-based Recommendation ---
if st.button("📍 Recommend Nearby"):
    if query.strip():
        recs = generate_geo_recommendation(query, lat, lon)
        st.subheader("📍 Nearby Recommendations")
        st.dataframe(recs)
    else:
        st.warning("Please enter a query.")

# Footer
st.markdown("---")
st.markdown("Built with [Gemini API](https://makersuite.google.com/), [ChromaDB](https://www.trychroma.com/), and [Streamlit](https://streamlit.io/)")