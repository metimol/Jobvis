import os


class MissingGroqAPIKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("GROQ_API_KEY is not set")


GROQ_API_KEY = os.environ.get("GROQ_API_KEY") if os.environ.get("GROQ_API_KEY") else None
if not GROQ_API_KEY:
    raise MissingGroqAPIKeyError
