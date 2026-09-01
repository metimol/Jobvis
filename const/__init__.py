import os


class MissingGoogleAPIKeyError(Exception):
    def __init__(self) -> None:
        super().__init__("GOOGLE_API_KEY is not set")


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") if os.environ.get("GOOGLE_API_KEY") else None
if not GOOGLE_API_KEY:
    raise MissingGoogleAPIKeyError
