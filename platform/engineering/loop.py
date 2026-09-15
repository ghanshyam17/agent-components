from __future__ import annotations

import logging
import asyncio
import functools
from typing import Any, Dict, List, Optional, Tuple, Callable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class LoopConfig(BaseModel):
    max_iterations: int = Field(default=10)
    max_duration_seconds: int = Field(default=300)
    max_tokens: int = Field(default=100000)
    max_cost_usd: float = Field(default=1.0)
    backoff_strategy: str = Field(default="none", description="none, linear, exponential")
    circuit_breaker_threshold: int = Field(default=3, description="consecutive failures before breaking")
    early_exit_confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    escalation_on_uncertainty: bool = Field(default=True)

class LoopTelemetry(BaseModel):
    iteration_count: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    tool_calls_count: int = Field(default=0)
    tool_calls_by_name: Dict[str, int] = Field(default_factory=dict)
    duration_ms: int = Field(default=0)
    early_exit: bool = Field(default=False)
    circuit_broken: bool = Field(default=False)

class LoopGuardrail(BaseModel):
    type: str = Field(..., description="max_spend, max_time, content_gate, custom")
    threshold: float = Field(...)
    action: str = Field(..., description="warn, stop, escalate")

class LoopController:
    """
    Advanced agent loop controls and telemetry.
    """
    def __init__(self, config: LoopConfig, guardrails: Optional[List[LoopGuardrail]] = None):
        self.config = config
        self.guardrails = guardrails or []

    async def should_continue(self, telemetry: LoopTelemetry) -> Tuple[bool, str]:
        if telemetry.iteration_count >= self.config.max_iterations:
            return False, "Max iterations reached"
        if (telemetry.duration_ms / 1000) >= self.config.max_duration_seconds:
            return False, "Max duration reached"
        if telemetry.total_tokens >= self.config.max_tokens:
            return False, "Max tokens reached"
        if telemetry.total_cost_usd >= self.config.max_cost_usd:
            return False, "Max cost reached"
        if telemetry.circuit_broken:
            return False, "Circuit broken"
        
        triggered_guardrails = self.check_guardrails(telemetry)
        for gr in triggered_guardrails:
            if "stop" in gr or "escalate" in gr:
                return False, f"Guardrail triggered: {gr}"

        return True, "Conditions normal"

    def check_guardrails(self, telemetry: LoopTelemetry) -> List[str]:
        triggered = []
        for gr in self.guardrails:
            if gr.type == "max_spend" and telemetry.total_cost_usd >= gr.threshold:
                triggered.append(f"max_spend threshold {gr.threshold} exceeded ({gr.action})")
            elif gr.type == "max_time" and (telemetry.duration_ms / 1000) >= gr.threshold:
                triggered.append(f"max_time threshold {gr.threshold} exceeded ({gr.action})")
        return triggered

    def calculate_backoff(self, iteration: int) -> float:
        if self.config.backoff_strategy == "linear":
            return float(iteration)
        elif self.config.backoff_strategy == "exponential":
            return min(60.0, 2.0 ** iteration)
        return 0.0

    def check_circuit_breaker(self, consecutive_failures: int) -> bool:
        return consecutive_failures >= self.config.circuit_breaker_threshold

    def to_config_dict(self) -> Dict[str, Any]:
        return self.config.model_dump()

    @classmethod
    def from_agent_spec(cls, agent_spec: Any) -> LoopController:
        config_data = getattr(agent_spec, "loop_config", {})
        guardrails_data = getattr(agent_spec, "loop_guardrails", [])
        return cls(
            config=LoopConfig(**config_data),
            guardrails=[LoopGuardrail(**gr) for gr in guardrails_data]
        )

def loop_controlled(controller: LoopController):
    """
    Decorator/wrapper that can be applied to AgentPattern.stream() to add loop controls.
    """
    def decorator(func: Callable):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _coro_wrapper(*args, **kwargs):
                telemetry = LoopTelemetry()
                should_cont, reason = await controller.should_continue(telemetry)
                if not should_cont:
                    logger.warning(f"Loop blocked early: {reason}")
                    return None
                return await func(*args, **kwargs)
            return _coro_wrapper
        else:
            @functools.wraps(func)
            async def _agen_wrapper(*args, **kwargs):
                telemetry = LoopTelemetry()
                should_cont, reason = await controller.should_continue(telemetry)
                if not should_cont:
                    logger.warning(f"Loop blocked early: {reason}")
                    return
                async for item in func(*args, **kwargs):
                    yield item
            return _agen_wrapper
    return decorator

