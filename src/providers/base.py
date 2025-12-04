"""LLMクライアントの基底クラス"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    error: str | None = None
    reasoning_tokens: int = 0

    def calculate_cost(self, input_price: float, output_price: float) -> float:
        input_cost = (self.input_tokens / 1_000_000) * input_price
        # 出力コストにはreasoning_tokensも含める（実際の課金額）
        total_output = self.output_tokens + self.reasoning_tokens
        output_cost = (total_output / 1_000_000) * output_price
        return input_cost + output_cost

class BaseLLMClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, model_id: str, **kwargs) -> LLMResponse:
        pass
