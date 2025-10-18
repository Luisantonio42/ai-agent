import os
import sys
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Query Gemini model")
    parser.add_argument("prompt", type=str, help="The user's prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    user_prompt = args.prompt
    verbose = args.verbose

    # --- API Configuration ---
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file or environment variables.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    messages = [types.Content(role="user", parts=[types.Part(text=user_prompt)])]

    # --- Model Interaction ---
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=messages
        )

        # --- Verbose mode output ---
        if verbose:
            print(f'User prompt: "{user_prompt}"')
            if hasattr(response, "usage_metadata"):
                print(f"Prompt tokens: {getattr(response.usage_metadata, 'prompt_token_count', 'N/A')}")
                print(f"Response tokens: {getattr(response.usage_metadata, 'candidates_token_count', 'N/A')}")

        # --- Always print response text ---
        print("\n--- Response ---")
        print(response.text)

        # --- Always print total token usage summary ---
        print("\n--- Token Usage ---")
        print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
        print(f"Candidates Tokens: {response.usage_metadata.candidates_token_count}")
        print(f"Total Tokens: {response.usage_metadata.total_token_count}")
        print("--------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
