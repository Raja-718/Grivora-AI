# gemini_client.py - Gemini 2.5 Flash API wrapper
#
# Lazy-init: the google.generativeai SDK and the API key check are deferred
# until the first .generate() call. This means an app missing GEMINI_API_KEY
# still boots, auth still works, and data-only features still work — only
# LLM-powered endpoints return a clean error at call time.
import os
from dotenv import load_dotenv

load_dotenv()

_MODEL_NAME = "gemini-2.5-flash"


class GeminiConfigError(RuntimeError):
    """Raised when the Gemini SDK can't be initialized (missing key/SDK)."""


class GeminiClient:
    def __init__(self):
        # Don't initialize the SDK here. Defer until first use so the app
        # can start even if GEMINI_API_KEY is missing.
        self._model = None
        self._init_error: str = ""

    def _ensure_model(self):
        """Initialize the SDK on first use. Caches the model on success,
        caches the error message on failure so we don't spam import attempts."""
        if self._model is not None:
            return
        if self._init_error:
            # Already failed once; don't retry every call.
            raise GeminiConfigError(self._init_error)

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            self._init_error = (
                "GEMINI_API_KEY not set. Add it to .env to enable AI features."
            )
            raise GeminiConfigError(self._init_error)

        try:
            import google.generativeai as genai
        except ImportError as e:
            self._init_error = (
                f"google-generativeai SDK not installed: {e}. "
                f"Run: pip install -r requirements.txt"
            )
            raise GeminiConfigError(self._init_error)

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(_MODEL_NAME)
        except Exception as e:
            self._init_error = f"Gemini initialization failed: {e}"
            raise GeminiConfigError(self._init_error)

    def generate(self, prompt: str, timeout: float = 20.0) -> str:
        """Generate text from Gemini. Returns a user-visible error string
        on failure rather than raising — callers already display the result.

        Args:
            prompt:  The prompt to send.
            timeout: Hard ceiling (seconds) on the API call. Defaults to 20s
                     so a hung Gemini never blocks a Flask request forever.
        """
        try:
            self._ensure_model()
        except GeminiConfigError as e:
            return f"[AI unavailable: {e}]"

        try:
            # request_options={'timeout': N} is honored by the google-generativeai
            # SDK for both streaming and non-streaming calls.
            response = self._model.generate_content(
                prompt,
                request_options={"timeout": timeout},
            )
            return response.text
        except Exception as e:
            return f"Error from Gemini API: {str(e)}"
