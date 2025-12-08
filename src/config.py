import os
from dataclasses import dataclass

USD_TO_JPY = 150.0

@dataclass
class ModelConfig:
    id: str
    name: str
    provider: str
    input_price: float
    output_price: float

MODELS = {
    "openai": [
        ModelConfig("gpt-5.1", "GPT-5.1", "openai", 1.25, 10.00),
        ModelConfig("gpt-5", "GPT-5", "openai", 1.25, 10.00),
        ModelConfig("gpt-5-mini", "GPT-5 mini", "openai", 0.25, 2.00),
        ModelConfig("gpt-5-nano", "GPT-5 nano", "openai", 0.05, 0.40),
    ],
    "anthropic": [
        ModelConfig("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "anthropic", 3.00, 15.00),
        ModelConfig("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "anthropic", 1.00, 5.00),
    ],
    "google": [
        ModelConfig("gemini-3-pro-preview", "Gemini 3 Pro", "google", 2.00, 12.00),
        ModelConfig("gemini-2.5-pro", "Gemini 2.5 Pro", "google", 1.25, 10.00),
        ModelConfig("gemini-2.5-flash", "Gemini 2.5 Flash", "google", 0.30, 2.50),
    ],
    "xai": [
        ModelConfig("grok-4", "Grok 4", "xai", 3.00, 15.00),
        ModelConfig("grok-4-1-fast-non-reasoning", "Grok 4.1 Fast (non-reasoning)", "xai", 0.20, 0.50),
        ModelConfig("grok-3-mini", "Grok 3 Mini", "xai", 0.30, 0.50),
    ],
}

def get_api_key(provider: str) -> str | None:
    keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY", "xai": "XAI_API_KEY"}
    return os.getenv(keys.get(provider, ""))
