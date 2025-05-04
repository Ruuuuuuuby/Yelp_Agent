
# Gemini + Chroma RAG Restaurant Agent

🍽️ Yelp Agent: Gemini-Powered Restaurant Recommendation System

Yelp Agent is an intelligent restaurant recommendation system powered by Google Gemini (via Generative AI), ChromaDB embeddings, and Streamlit. It combines semantic search, location awareness, and review understanding to generate real-time, personalized suggestions.

## Features
	•	LLM-Powered Recommendations: Gemini generates responses based on user queries and vector-based document context.
	•	Vector Store with ChromaDB: Restaurant review texts are embedded using SentenceTransformer and indexed for fast retrieval.
	•	Location-Based Filtering: Users can input latitude and longitude to find nearby top-rated restaurants.
	•	Geospatial Clustering: DBSCAN groups restaurants into local clusters for smarter region-based recommendations.
	•	Snippet-Enhanced Dataset: Merges original restaurant data with user review snippets for deeper reasoning.

## Upcoming Modules (Planned)
	•	🔍 Semantic Snippet Matching: Use cosine similarity between user queries and restaurant snippets to re-rank results.
	•	💬 Zero-Shot Classification: Tag restaurants as “family-friendly”, “romantic”, or “group-friendly” without needing model fine-tuning.
	•	🌦️ Weather-Aware Reranking: Adjust recommendation scores based on real-time weather (e.g., avoid rooftop bars in rain).
	•	📄 User Document Uploads: Accept restaurant lists, menus, or reviews in PDF/CSV formats, embed them, and use them in RAG pipelines.
	•	📉 Review Density + Sentiment Weighting: Dynamically adjust recommendation scores based on review volume and tone.
	•	🛣️ Traffic Context (Planned): Factor in public transport accessibility and congestion data.
	•	📈 Performance Visualization: Include F1, ROC, and accuracy charts for any classification component (e.g., zero-shot tags).
	•	🛍️ Weather × Sales Analysis (Optional): Predict product performance (e.g., soups, ice creams) using historical sales + weather.

## Tech Stack
	•	Streamlit, pandas, sentence-transformers, ChromaDB
	•	Google GenerativeAI, folium, geopy, scikit-learn

## Setup

1. Install dependencies:

```
pip install -r requirements.txt
```
<!-- 
2. Create `.env` file from `.env.example` and add your Gemini API key. -->

3. Run the chatbot:

```
PYTHONPATH=$(pwd) streamlit run app/ui.py
```
