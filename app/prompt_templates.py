
def restaurant_prompt(user_query, context_texts):
    context = "\n".join(context_texts)
    return f"""You are a smart restaurant recommender assistant.
User asked: "{user_query}"
Relevant restaurant information:
{context}
Generate a concise, friendly and personalized recommendation."""
