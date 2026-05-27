from openai import OpenAI


BASE_URL = "http://localhost:8000/v1"
API_KEY = "dummy"
MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "Write a haiku about low-latency inference."


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
    )
    print(response.choices[0].message.content or "")
