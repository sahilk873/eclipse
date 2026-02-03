"""Load configuration from .env file. Used for API keys instead of shell environment."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    """Load .env from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_path = _PROJECT_ROOT / ".env"
        load_dotenv(env_path)
    except ImportError:
        pass


def get_openai_api_key() -> str | None:
    """Return OpenAI API key from .env (loaded into os.environ) or os.environ."""
    import os
    _load_dotenv()
    return os.environ.get("OPENAI_API_KEY")


def get_openai_llm_model() -> str:
    """
    Return LLM model ID from OPENAI_LLM_MODEL env or default.
    Default: gpt-4o-mini (widely available). Use gpt-5-mini or gpt-4o for better JSON output.
    """
    import os
    _load_dotenv()
    return os.environ.get("OPENAI_LLM_MODEL", "gpt-4o-mini")
