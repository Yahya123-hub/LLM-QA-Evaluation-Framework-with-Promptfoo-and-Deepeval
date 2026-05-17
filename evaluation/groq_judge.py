from deepeval.models import DeepEvalBaseLLM
from groq import Groq
import os


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="llama-3.1-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    def get_model_name(self):
        return self.model

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evaluation model. "
                        "Return ONLY valid JSON. "
                        "Do not include markdown, explanations, or extra text."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()

    async def a_generate(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt)