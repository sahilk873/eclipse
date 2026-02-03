#!/usr/bin/env python3
"""Quick test that OPENAI_API_KEY is set and valid (loaded from .env)."""

import sys

from env_config import get_openai_api_key


def main() -> None:
    key = get_openai_api_key()
    if not key:
        print("OPENAI_API_KEY is not set. Add it to .env in the project root.")
        sys.exit(1)

    masked = key[:7] + "..." + key[-4:] if len(key) > 11 else "***"
    print(f"Key found ({masked}). Calling API...")

    try:
        import openai
        client = openai.OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=5,
        )
        reply = (r.choices[0].message.content or "").strip()
        print(f"API reply: {reply}")
        if reply.upper() == "OK":
            print("API key works.")
        else:
            print("API responded.")
    except Exception as e:
        print(f"API error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
