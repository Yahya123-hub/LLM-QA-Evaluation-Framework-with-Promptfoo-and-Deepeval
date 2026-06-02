from groq import Groq
from dotenv import load_dotenv
import os

from rag.retriever import retrieve_context

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_response(user_input, model_name="llama-3.1-8b-instant"):
    context = retrieve_context(user_input)
    context_text = "\n".join(context)

    
    prompt = f"""
Use the context below to answer the user.

Context:
{context_text}

User question:
{user_input}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content