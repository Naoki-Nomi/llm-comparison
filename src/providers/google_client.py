"""Google Gemini APIクライアント（新SDK: google-genai）"""
import time
from google import genai
from google.genai import types
from .base import BaseLLMClient, LLMResponse

class GoogleClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, model_id: str, 
                 system_prompt: str = "",
                 temperature: float = 0.0,
                 max_tokens: int = 10000,
                 thinking_level: str = None,
                 **kwargs) -> LLMResponse:
        try:
            start = time.perf_counter()
            config_params = {
                "max_output_tokens": max_tokens,
            }

            # システムプロンプト設定
            if system_prompt:
                config_params["system_instruction"] = system_prompt

            # Gemini 3 Pro用: thinking_level設定、temperatureは1.0推奨なので設定しない
            if "gemini-3" in model_id:
                if thinking_level:
                    config_params["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level.upper()
                    )
            else:
                # Gemini 2.5系はtemperatureを設定
                config_params["temperature"] = temperature

            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(**config_params),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            input_tokens = output_tokens = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

            return LLMResponse(response.text or "", input_tokens, output_tokens, elapsed_ms, model_id, None, 0)
        except Exception as e:
            return LLMResponse("", 0, 0, 0, model_id, str(e), 0)
