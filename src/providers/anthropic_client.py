"""Anthropic APIクライアント"""
import time
import anthropic
from .base import BaseLLMClient, LLMResponse

class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, model_id: str, 
                 system_prompt: str = "",
                 extended_thinking: bool = False,
                 budget_tokens: int = 8000,
                 temperature: float = 0.0,
                 max_tokens: int = 10000,
                 **kwargs) -> LLMResponse:
        try:
            start = time.perf_counter()

            params = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                params["system"] = system_prompt
            if extended_thinking:
                adjusted_max_tokens = max(max_tokens, budget_tokens + 1000)
                params["max_tokens"] = adjusted_max_tokens
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens
                }
            else:
                params["max_tokens"] = max_tokens
                params["temperature"] = temperature

            response = self.client.messages.create(**params)
            elapsed_ms = (time.perf_counter() - start) * 1000

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            raw = response.model_dump() if hasattr(response, 'model_dump') else {}
            return LLMResponse(
                content,
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
