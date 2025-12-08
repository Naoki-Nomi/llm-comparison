"""xAI Grok APIクライアント（OpenAI互換）"""
import time
from openai import OpenAI
from .base import BaseLLMClient, LLMResponse

class XAIClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    def generate(self, prompt: str, model_id: str, 
                 system_prompt: str = "",
                 temperature: float = 0.0,
                 max_tokens: int = 10000,
                 **kwargs) -> LLMResponse:
        try:
            start = time.perf_counter()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            # トークン情報を取得
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            # Grokはデフォルトでsafety promptが挿入されキャッシュされるため、差し引く
            if response.usage and hasattr(response.usage, 'prompt_tokens_details'):
                details = response.usage.prompt_tokens_details
                if details and hasattr(details, 'cached_tokens') and details.cached_tokens:
                    input_tokens = input_tokens - details.cached_tokens

            # reasoning_tokensはcompletion_tokensに含まれないため、足し合わせる
            if response.usage and hasattr(response.usage, 'completion_tokens_details'):
                details = response.usage.completion_tokens_details
                if details and hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                    output_tokens = output_tokens + details.reasoning_tokens

            # 生レスポンスを辞書に変換
            raw = response.model_dump() if hasattr(response, 'model_dump') else {}

            return LLMResponse(
                response.choices[0].message.content or "",
                input_tokens,
                output_tokens,
                elapsed_ms,
                model_id,
                None,
                0,
                raw,
            )
        except Exception as e:
            return LLMResponse("", 0, 0, 0, model_id, str(e), 0)
