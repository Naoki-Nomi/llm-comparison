"""LLMクライアントの基底クラス"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    error: str | None = None
    reasoning_tokens: int = 0
    raw_response: dict = field(default_factory=dict)

    def calculate_cost(self, input_price: float, output_price: float) -> float:
        input_cost = (self.input_tokens / 1_000_000) * input_price
        output_cost = (self.output_tokens / 1_000_000) * output_price
        return input_cost + output_cost

class BaseLLMClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, model_id: str, **kwargs) -> LLMResponse:
        pass
