"""OpenAI APIクライアント"""
import time
from openai import OpenAI
from .base import BaseLLMClient, LLMResponse

class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model_id: str, 
                 temperature: float = None, 
                 reasoning_effort: str = None,
                 verbosity: str = None,
                 max_completion_tokens: int = 10000) -> LLMResponse:
        try:
            start = time.perf_counter()

            # GPT-5/5.1 (reasoning model) の場合はResponses APIを使用
            if "gpt-5" in model_id and "mini" not in model_id:
                return self._generate_with_responses_api(
                    prompt, model_id, reasoning_effort, verbosity, max_completion_tokens, start)
            else:
                return self._generate_with_chat_api(
                    prompt, model_id, temperature, max_completion_tokens, start)
        except Exception as e:
            return LLMResponse("", 0, 0, 0, model_id, str(e), 0)

    def _generate_with_responses_api(self, prompt: str, model_id: str,
                                      reasoning_effort: str, verbosity: str,
                                      max_tokens: int, start: float) -> LLMResponse:
        """GPT-5/5.1用のResponses API"""
        params = {
            "model": model_id,
            "input": prompt,
            "max_output_tokens": max_tokens,
        }

        # reasoning_effort設定
        if reasoning_effort:
            params["reasoning"] = {"effort": reasoning_effort}

        # verbosity設定
        if verbosity:
            params["text"] = {"verbosity": verbosity}

        response = self.client.responses.create(**params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # レスポンステキストを抽出（output_textを優先使用）
        content = ""
        if hasattr(response, 'output_text') and response.output_text:
            content = response.output_text
        elif response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for c in item.content:
                        if hasattr(c, "text"):
                            content += c.text

        # トークン情報を取得
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0
        reasoning_tokens = 0

        # reasoning_tokensを取得
        if response.usage and hasattr(response.usage, 'output_tokens_details'):
            details = response.usage.output_tokens_details
            if details and hasattr(details, 'reasoning_tokens') and details.reasoning_tokens:
                reasoning_tokens = details.reasoning_tokens
                output_tokens = output_tokens - reasoning_tokens

        return LLMResponse(content, input_tokens, output_tokens, elapsed_ms, model_id, None, reasoning_tokens)

    def _generate_with_chat_api(self, prompt: str, model_id: str,
                                 temperature: float, max_tokens: int, start: float) -> LLMResponse:
        """GPT-4o等の通常モデル用のChat Completions API"""
        params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
        }

        if temperature is not None:
            params["temperature"] = temperature

        response = self.client.chat.completions.create(**params)
        elapsed_ms = (time.perf_counter() - start) * 1000

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        return LLMResponse(
            response.choices[0].message.content or "",
            input_tokens,
            output_tokens,
            elapsed_ms,
            model_id,
            None,
            0,
        )
