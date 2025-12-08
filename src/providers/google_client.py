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

            input_tokens = 0
            output_tokens = 0
            raw = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or 0
                # output_tokens = total - input（思考トークン込み）
                output_tokens = total_tokens - input_tokens
                # usage_metadataを辞書に変換
                if hasattr(response.usage_metadata, 'model_dump'):
                    raw["usage_metadata"] = response.usage_metadata.model_dump()
                elif hasattr(response.usage_metadata, '__dict__'):
                    raw["usage_metadata"] = dict(response.usage_metadata.__dict__)

            # モデル情報を追加
            raw["model"] = model_id

            return LLMResponse(response.text or "", input_tokens, output_tokens, elapsed_ms, model_id, None, 0, raw)
        except Exception as e:
            return LLMResponse("", 0, 0, 0, model_id, str(e), 0)
