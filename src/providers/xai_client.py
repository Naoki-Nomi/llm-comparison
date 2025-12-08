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
            return LLMResponse(
                response.choices[0].message.content or "",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
                elapsed_ms,
                model_id,
                None,
                0,
            )
        except Exception as e:
            return LLMResponse("", 0, 0, 0, model_id, str(e), 0)
