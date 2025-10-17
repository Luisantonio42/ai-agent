import os
from dotenv import load_dotenv
from google import genai
from google.genai import Client

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-pro", contents="Why is the sky blue?"
)
# print(response.text)
# print(response.text)
# --- Here is how you print the token count ---
print("\n--- Token Usage ---")
print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
print(f"Candidates Tokens: {response.usage_metadata.candidates_token_count}")
print(f"Total Tokens: {response.usage_metadata.total_token_count}")
print("--------------------")
