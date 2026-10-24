import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url=os.getenv("NEBIUS_BASE_URL",
                       "https://api.tokenfactory.nebius.com/v1/"),
)

print("Key loaded:", bool(os.getenv("NEBIUS_API_KEY")))

try:
    models = [m.id for m in client.models.list().data]
    print(f"\nModels available to you ({len(models)}):")
    for m in models:
        if "llama" in m.lower() or "qwen" in m.lower():
            print("  ", m)
except Exception as e:
    print("\nListing models failed:", type(e).__name__, e)

try:
    r = client.chat.completions.create(
        model=os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=10,
    )
    print("\nTest call worked:", r.choices[0].message.content)
except Exception as e:
    print("\nTest call failed:", type(e).__name__, e)
