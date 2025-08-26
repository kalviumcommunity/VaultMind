"""
VaultMind Token Counting and Cost Tracking System

This module implements comprehensive token counting, cost estimation, and usage
logging for all AI interactions in VaultMind. Provides real-time tracking of
input/output tokens, cost calculations, and detailed usage analytics.

Token tracking is essential for managing AI costs, optimizing prompt efficiency,
and understanding the computational requirements of different prompting strategies.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import re
import math
import statistics
from collections import defaultdict, Counter
import threading
import time


class ModelProvider(Enum):
    """Supported AI model providers with different tokenization schemes."""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    GOOGLE_GEMINI = "google_gemini"
    COHERE_COMMAND = "cohere_command"
    MISTRAL_7B = "mistral_7b"
    LLAMA2_70B = "llama2_70b"


class TokenType(Enum):
    """Different types of tokens for cost calculation."""
    INPUT_TOKEN = "input_token"
    OUTPUT_TOKEN = "output_token"
    CACHED_TOKEN = "cached_token"
    SYSTEM_TOKEN = "system_token"
    USER_TOKEN = "user_token"
    ASSISTANT_TOKEN = "assistant_token"


class PromptingStrategy(Enum):
    """Prompting strategies for usage analytics."""
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    MULTI_SHOT = "multi_shot"
    DYNAMIC = "dynamic"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class ModelPricing:
    """Pricing information for different AI models."""
    provider: ModelProvider
    model_name: str
    input_token_cost: float  # Cost per 1K input tokens
    output_token_cost: float  # Cost per 1K output tokens
    context_window: int  # Maximum context length
    cached_token_cost: float = 0.0  # Cost per 1K cached tokens (if applicable)

    def calculate_cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        """Calculate total cost for a request."""
        input_cost = (input_tokens / 1000) * self.input_token_cost
        output_cost = (output_tokens / 1000) * self.output_token_cost
        cached_cost = (cached_tokens / 1000) * self.cached_token_cost if cached_tokens > 0 else 0
        return input_cost + output_cost + cached_cost


@dataclass
class TokenUsage:
    """Represents token usage for a single AI interaction."""
    request_id: str
    timestamp: datetime
    model_provider: ModelProvider
    prompting_strategy: PromptingStrategy
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    prompt_length_chars: int
    response_length_chars: int
    processing_time: float = 0.0
    cached_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_efficiency_score(self) -> float:
        """Calculate efficiency score (output value per input token)."""
        if self.input_tokens == 0:
            return 0.0
        return self.output_tokens / self.input_tokens

    def get_cost_per_output_token(self) -> float:
        """Calculate cost per output token."""
        if self.output_tokens == 0:
            return 0.0
        return self.estimated_cost / self.output_tokens


@dataclass
class SessionSummary:
    """Summary of token usage for a session or time period."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    average_tokens_per_request: float = 0.0
    strategy_breakdown: Dict[PromptingStrategy, int] = field(default_factory=dict)
    model_breakdown: Dict[ModelProvider, int] = field(default_factory=dict)
    cost_by_strategy: Dict[PromptingStrategy, float] = field(default_factory=dict)

    def update_summary(self, usage: TokenUsage):
        """Update summary with new token usage."""
        self.total_requests += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost += usage.estimated_cost

        # Update averages
        total_tokens = self.total_input_tokens + self.total_output_tokens
        self.average_tokens_per_request = total_tokens / self.total_requests

        # Update breakdowns
        self.strategy_breakdown[usage.prompting_strategy] = \
            self.strategy_breakdown.get(usage.prompting_strategy, 0) + 1
        self.model_breakdown[usage.model_provider] = \
            self.model_breakdown.get(usage.model_provider, 0) + 1
        self.cost_by_strategy[usage.prompting_strategy] = \
            self.cost_by_strategy.get(usage.prompting_strategy, 0.0) + usage.estimated_cost


class TokenizerInterface(ABC):
    """Abstract interface for different tokenization implementations."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        pass


class EstimatedTokenizer(TokenizerInterface):
    """Estimated tokenizer using heuristics when exact tokenizer unavailable."""

    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        self.chars_per_token = self._get_chars_per_token_estimate()

    def _get_chars_per_token_estimate(self) -> float:
        """Get estimated characters per token for different models."""
        estimates = {
            ModelProvider.OPENAI_GPT4: 4.0,
            ModelProvider.OPENAI_GPT35: 4.0,
            ModelProvider.ANTHROPIC_CLAUDE: 3.8,
            ModelProvider.GOOGLE_GEMINI: 4.2,
            ModelProvider.COHERE_COMMAND: 4.1,
            ModelProvider.MISTRAL_7B: 4.3,
            ModelProvider.LLAMA2_70B: 4.2
        }
        return estimates.get(self.model_provider, 4.0)

    def count_tokens(self, text: str) -> int:
        """Estimate token count based on character count and model characteristics."""
        if not text:
            return 0

        # Basic character-based estimation
        base_estimate = len(text) / self.chars_per_token

        # Adjust for common patterns
        adjustments = 0

        # Special tokens (newlines, punctuation)
        adjustments += text.count('\n') * 0.1
        adjustments += len(re.findall(r'[.!?;:]', text)) * 0.1

        # Code-like patterns (more tokens per character)
        if '```' in text or text.count('{') > 5:
            adjustments += base_estimate * 0.15

        # JSON/structured data
        if text.strip().startswith('{') and text.strip().endswith('}'):
            adjustments += base_estimate * 0.1

        # Technical terms (tend to be tokenized into more pieces)
        technical_patterns = len(re.findall(r'[A-Z]{2,}|[a-z]+_[a-z]+|\w+\.\w+', text))
        adjustments += technical_patterns * 0.05

        return int(base_estimate + adjustments)

    def encode(self, text: str) -> List[int]:
        """Mock encoding for estimated tokenizer."""
        token_count = self.count_tokens(text)
        return list(range(token_count))  # Mock token IDs

    def decode(self, token_ids: List[int]) -> str:
        """Mock decoding for estimated tokenizer."""
        return f"[Decoded text from {len(token_ids)} tokens]"


class PreciseTokenizer(TokenizerInterface):
    """Precise tokenizer using actual model tokenizers (when available)."""

    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load the actual tokenizer for the model."""
        # In real implementation, this would load actual tokenizers:
        # - tiktoken for OpenAI models
        # - transformers tokenizers for open-source models
        # - API-specific tokenizers for proprietary models

        # For demo purposes, return None (falls back to estimation)
        return None

    def count_tokens(self, text: str) -> int:
        """Count tokens using precise tokenizer or fall back to estimation."""
        if self.tokenizer is None:
            # Fall back to estimation
            estimator = EstimatedTokenizer(self.model_provider)
            return estimator.count_tokens(text)

        # Use actual tokenizer (implementation would vary by provider)
        # return len(self.tokenizer.encode(text))
        return EstimatedTokenizer(self.model_provider).count_tokens(text)

    def encode(self, text: str) -> List[int]:
        """Encode using precise tokenizer."""
        if self.tokenizer is None:
            return EstimatedTokenizer(self.model_provider).encode(text)
        # return self.tokenizer.encode(text)
        return EstimatedTokenizer(self.model_provider).encode(text)

    def decode(self, token_ids: List[int]) -> str:
        """Decode using precise tokenizer."""
        if self.tokenizer is None:
            return EstimatedTokenizer(self.model_provider).decode(token_ids)
        # return self.tokenizer.decode(token_ids)
        return EstimatedTokenizer(self.model_provider).decode(token_ids)


class TokenCounter:
    """
    Main class for comprehensive token counting, cost tracking, and usage analytics.

    Provides real-time token counting, cost estimation, usage logging, and detailed
    analytics for AI interactions across different prompting strategies.
    """

    def __init__(self,
                 default_model: ModelProvider = ModelProvider.OPENAI_GPT4,
                 enable_logging: bool = True,
                 enable_real_time_display: bool = False):
        self.default_model = default_model
        self.enable_logging = enable_logging
        self.enable_real_time_display = enable_real_time_display

        # Initialize components
        self.model_pricing = self._initialize_model_pricing()
        self.tokenizers = self._initialize_tokenizers()
        self.usage_log: List[TokenUsage] = []
        self.current_session = self._start_new_session()

        # Thread-safe counters
        self._lock = threading.Lock()

        # Real-time display
        if self.enable_real_time_display:
            self._start_real_time_display()

    def _initialize_model_pricing(self) -> Dict[ModelProvider, ModelPricing]:
        """Initialize current pricing for different AI models (as of August 2025)."""
        return {
            ModelProvider.OPENAI_GPT4: ModelPricing(
                provider=ModelProvider.OPENAI_GPT4,
                model_name="GPT-4",
                input_token_cost=0.03,  # $0.03 per 1K input tokens
                output_token_cost=0.06,  # $0.06 per 1K output tokens
                context_window=128000,
                cached_token_cost=0.015
            ),
            ModelProvider.OPENAI_GPT35: ModelPricing(
                provider=ModelProvider.OPENAI_GPT35,
                model_name="GPT-3.5 Turbo",
                input_token_cost=0.0015,
                output_token_cost=0.002,
                context_window=16385
            ),
            ModelProvider.ANTHROPIC_CLAUDE: ModelPricing(
                provider=ModelProvider.ANTHROPIC_CLAUDE,
                model_name="Claude-3.5 Sonnet",
                input_token_cost=0.003,
                output_token_cost=0.015,
                context_window=200000
            ),
            ModelProvider.GOOGLE_GEMINI: ModelPricing(
                provider=ModelProvider.GOOGLE_GEMINI,
                model_name="Gemini 1.5 Flash",
                input_token_cost=0.00075,
                output_token_cost=0.003,
                context_window=1000000
            ),
            ModelProvider.COHERE_COMMAND: ModelPricing(
                provider=ModelProvider.COHERE_COMMAND,
                model_name="Command R+",
                input_token_cost=0.003,
                output_token_cost=0.015,
                context_window=128000
            ),
            ModelProvider.MISTRAL_7B: ModelPricing(
                provider=ModelProvider.MISTRAL_7B,
                model_name="Mistral 7B",
                input_token_cost=0.0002,
                output_token_cost=0.0002,
                context_window=8192
            ),
            ModelProvider.LLAMA2_70B: ModelPricing(
                provider=ModelProvider.LLAMA2_70B,
                model_name="Llama 2 70B",
                input_token_cost=0.0007,
                output_token_cost=0.0009,
                context_window=4096
            )
        }

    def _initialize_tokenizers(self) -> Dict[ModelProvider, TokenizerInterface]:
        """Initialize tokenizers for different models."""
        tokenizers = {}
        for provider in ModelProvider:
            # Try to load precise tokenizer, fall back to estimated
            try:
                tokenizers[provider] = PreciseTokenizer(provider)
            except Exception:
                tokenizers[provider] = EstimatedTokenizer(provider)

        return tokenizers

    def _start_new_session(self) -> SessionSummary:
        """Start a new usage tracking session."""
        return SessionSummary(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now()
        )

    def _start_real_time_display(self):
        """Start real-time token usage display in a separate thread."""
        def display_loop():
            while self.enable_real_time_display:
                time.sleep(2)  # Update every 2 seconds
                self._print_real_time_status()

        display_thread = threading.Thread(target=display_loop, daemon=True)
        display_thread.start()

    def count_prompt_tokens(self,
                          prompt: str,
                          model_provider: ModelProvider = None) -> int:
        """Count tokens in a prompt text."""
        provider = model_provider or self.default_model
        tokenizer = self.tokenizers[provider]
        return tokenizer.count_tokens(prompt)

    def count_response_tokens(self,
                            response: str,
                            model_provider: ModelProvider = None) -> int:
        """Count tokens in a response text."""
        provider = model_provider or self.default_model
        tokenizer = self.tokenizers[provider]
        return tokenizer.count_tokens(response)

    def log_ai_interaction(self,
                          prompt: str,
                          response: str,
                          model_provider: ModelProvider = None,
                          strategy: PromptingStrategy = PromptingStrategy.ZERO_SHOT,
                          processing_time: float = 0.0,
                          metadata: Dict[str, Any] = None) -> TokenUsage:
        """
        Log a complete AI interaction with comprehensive token tracking.

        Args:
            prompt: The input prompt text
            response: The AI response text
            model_provider: The model used
            strategy: The prompting strategy employed
            processing_time: Time taken for the request
            metadata: Additional metadata

        Returns:
            TokenUsage object with detailed token and cost information
        """
        with self._lock:
            provider = model_provider or self.default_model

            # Count tokens
            input_tokens = self.count_prompt_tokens(prompt, provider)
            output_tokens = self.count_response_tokens(response, provider)
            total_tokens = input_tokens + output_tokens

            # Calculate cost
            pricing = self.model_pricing[provider]
            estimated_cost = pricing.calculate_cost(input_tokens, output_tokens)

            # Create usage record
            usage = TokenUsage(
                request_id=f"req_{len(self.usage_log) + 1:06d}",
                timestamp=datetime.now(),
                model_provider=provider,
                prompting_strategy=strategy,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                estimated_cost=estimated_cost,
                prompt_length_chars=len(prompt),
                response_length_chars=len(response),
                processing_time=processing_time,
                metadata=metadata or {}
            )

            # Store usage
            self.usage_log.append(usage)
            self.current_session.update_summary(usage)

            # Log to console if enabled
            if self.enable_logging:
                self._log_interaction_to_console(usage)

            return usage

    def _log_interaction_to_console(self, usage: TokenUsage):
        """Log interaction details to console."""
        timestamp = usage.timestamp.strftime("%H:%M:%S")

        print(f"\n🤖 AI Interaction Logged [{timestamp}]")
        print(f"   Request ID: {usage.request_id}")
        print(f"   Strategy: {usage.prompting_strategy.value}")
        print(f"   Model: {usage.model_provider.value}")
        print(f"   Tokens: {usage.input_tokens:,} in → {usage.output_tokens:,} out ({usage.total_tokens:,} total)")
        print(f"   Cost: ${usage.estimated_cost:.4f}")
        print(f"   Efficiency: {usage.get_efficiency_score():.2f} output tokens per input token")

        if usage.processing_time > 0:
            tokens_per_second = usage.output_tokens / usage.processing_time
            print(f"   Speed: {tokens_per_second:.1f} tokens/second")

        print(f"   Session Total: ${self.current_session.total_cost:.4f}")

    def _print_real_time_status(self):
        """Print real-time status update."""
        if not self.usage_log:
            return

        session = self.current_session
        recent_usage = self.usage_log[-5:]  # Last 5 interactions

        print(f"\n📊 Real-time Token Usage Status")
        print(f"   Session: {session.total_requests} requests, ${session.total_cost:.4f} total")
        print(f"   Last 5 interactions:")

        for usage in recent_usage:
            efficiency = usage.get_efficiency_score()
            print(f"     {usage.timestamp.strftime('%H:%M:%S')} | "
                  f"{usage.prompting_strategy.value[:10]:10} | "
                  f"{usage.total_tokens:,} tokens | "
                  f"${usage.estimated_cost:.4f} | "
                  f"eff: {efficiency:.2f}")

    def get_session_summary(self) -> SessionSummary:
        """Get current session summary."""
        self.current_session.end_time = datetime.now()
        return self.current_session

    def get_cost_breakdown_by_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get detailed cost breakdown by prompting strategy."""
        breakdown = defaultdict(lambda: {
            'total_cost': 0.0,
            'total_tokens': 0,
            'request_count': 0,
            'avg_cost_per_request': 0.0,
            'avg_tokens_per_request': 0.0
        })

        for usage in self.usage_log:
            strategy = usage.prompting_strategy.value
            breakdown[strategy]['total_cost'] += usage.estimated_cost
            breakdown[strategy]['total_tokens'] += usage.total_tokens
            breakdown[strategy]['request_count'] += 1

        # Calculate averages
        for strategy, data in breakdown.items():
            if data['request_count'] > 0:
                data['avg_cost_per_request'] = data['total_cost'] / data['request_count']
                data['avg_tokens_per_request'] = data['total_tokens'] / data['request_count']

        return dict(breakdown)

    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance across different models."""
        model_stats = defaultdict(lambda: {
            'usage_count': 0,
            'total_cost': 0.0,
            'total_tokens': 0,
            'avg_efficiency': 0.0,
            'cost_per_1k_tokens': 0.0
        })

        for usage in self.usage_log:
            model = usage.model_provider.value
            model_stats[model]['usage_count'] += 1
            model_stats[model]['total_cost'] += usage.estimated_cost
            model_stats[model]['total_tokens'] += usage.total_tokens
            model_stats[model]['avg_efficiency'] += usage.get_efficiency_score()

        # Calculate derived metrics
        for model, stats in model_stats.items():
            if stats['usage_count'] > 0:
                stats['avg_efficiency'] /= stats['usage_count']
                if stats['total_tokens'] > 0:
                    stats['cost_per_1k_tokens'] = (stats['total_cost'] / stats['total_tokens']) * 1000

        return dict(model_stats)

    def get_efficiency_trends(self, time_window_hours: int = 24) -> Dict[str, List[float]]:
        """Get efficiency trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_usage = [u for u in self.usage_log if u.timestamp >= cutoff_time]

        # Group by strategy
        strategy_trends = defaultdict(list)

        for usage in recent_usage:
            strategy = usage.prompting_strategy.value
            efficiency = usage.get_efficiency_score()
            strategy_trends[strategy].append(efficiency)

        return dict(strategy_trends)

    def optimize_token_usage(self) -> Dict[str, Any]:
        """Provide recommendations for optimizing token usage and costs."""
        if not self.usage_log:
            return {"error": "No usage data available for optimization analysis"}

        recommendations = []
        analysis = {}

        # Analyze strategy efficiency
        strategy_breakdown = self.get_cost_breakdown_by_strategy()

        # Find most expensive strategy
        most_expensive = max(strategy_breakdown.items(),
                           key=lambda x: x[1]['avg_cost_per_request'])
        recommendations.append(
            f"Most expensive strategy: {most_expensive[0]} "
            f"(${most_expensive[1]['avg_cost_per_request']:.4f} per request)"
        )

        # Find most efficient strategy
        efficiencies = {}
        for usage in self.usage_log:
            strategy = usage.prompting_strategy.value
            if strategy not in efficiencies:
                efficiencies[strategy] = []
            efficiencies[strategy].append(usage.get_efficiency_score())

        avg_efficiencies = {s: statistics.mean(e) for s, e in efficiencies.items() if e}
        if avg_efficiencies:
            most_efficient = max(avg_efficiencies.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most efficient strategy: {most_efficient[0]} "
                f"({most_efficient[1]:.2f} output tokens per input token)"
            )

        # Analyze model costs
        model_comparison = self.get_model_comparison()
        if model_comparison:
            cheapest_model = min(model_comparison.items(),
                               key=lambda x: x[1]['cost_per_1k_tokens'])
            recommendations.append(
                f"Most cost-effective model: {cheapest_model[0]} "
                f"(${cheapest_model[1]['cost_per_1k_tokens']:.4f} per 1K tokens)"
            )

        # Token usage patterns
        recent_usage = self.usage_log[-10:]  # Last 10 interactions
        if recent_usage:
            avg_input_tokens = statistics.mean([u.input_tokens for u in recent_usage])
            avg_output_tokens = statistics.mean([u.output_tokens for u in recent_usage])

            if avg_input_tokens > 2000:
                recommendations.append("Consider breaking down long prompts to reduce input token costs")

            if avg_output_tokens < avg_input_tokens * 0.3:
                recommendations.append("Low output efficiency - consider optimizing prompts for more substantial responses")

        analysis['recommendations'] = recommendations
        analysis['strategy_breakdown'] = strategy_breakdown
        analysis['model_comparison'] = model_comparison

        return analysis

    def export_usage_data(self, format: str = "json", time_window_hours: int = None) -> str:
        """Export usage data in specified format."""
        usage_data = self.usage_log

        if time_window_hours:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            usage_data = [u for u in usage_data if u.timestamp >= cutoff_time]

        if format == "json":
            serializable_data = []
            for usage in usage_data:
                serializable_data.append({
                    "request_id": usage.request_id,
                    "timestamp": usage.timestamp.isoformat(),
                    "model_provider": usage.model_provider.value,
                    "prompting_strategy": usage.prompting_strategy.value,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost": usage.estimated_cost,
                    "efficiency_score": usage.get_efficiency_score(),
                    "processing_time": usage.processing_time,
                    "metadata": usage.metadata
                })

            return json.dumps({
                "export_timestamp": datetime.now().isoformat(),
                "total_interactions": len(usage_data),
                "usage_data": serializable_data,
                "session_summary": {
                    "total_cost": self.current_session.total_cost,
                    "total_tokens": self.current_session.total_input_tokens + self.current_session.total_output_tokens,
                    "strategy_breakdown": {k.value: v for k, v in self.current_session.strategy_breakdown.items()}
                }
            }, indent=2)

        elif format == "csv":
            lines = ["timestamp,request_id,model,strategy,input_tokens,output_tokens,total_tokens,cost,efficiency"]
            for usage in usage_data:
                lines.append(
                    f"{usage.timestamp.isoformat()},"
                    f"{usage.request_id},"
                    f"{usage.model_provider.value},"
                    f"{usage.prompting_strategy.value},"
                    f"{usage.input_tokens},"
                    f"{usage.output_tokens},"
                    f"{usage.total_tokens},"
                    f"{usage.estimated_cost:.6f},"
                    f"{usage.get_efficiency_score():.4f}"
                )
            return "\n".join(lines)

        else:
            # Text summary format
            lines = [
                "VaultMind Token Usage Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Interactions: {len(usage_data)}",
                ""
            ]

            if usage_data:
                total_cost = sum(u.estimated_cost for u in usage_data)
                total_tokens = sum(u.total_tokens for u in usage_data)

                lines.extend([
                    f"Total Cost: ${total_cost:.4f}",
                    f"Total Tokens: {total_tokens:,}",
                    f"Average Cost per Request: ${total_cost / len(usage_data):.4f}",
                    f"Average Tokens per Request: {total_tokens // len(usage_data):,}",
                    ""
                ])

                # Strategy breakdown
                strategy_costs = defaultdict(float)
                for usage in usage_data:
                    strategy_costs[usage.prompting_strategy.value] += usage.estimated_cost

                lines.append("Cost by Strategy:")
                for strategy, cost in sorted(strategy_costs.items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"  {strategy}: ${cost:.4f}")

            return "\n".join(lines)

    def reset_session(self):
        """Reset the current session and start a new one."""
        with self._lock:
            self.current_session = self._start_new_session()
            print(f"🔄 Started new session: {self.current_session.session_id}")

    def set_budget_alert(self, daily_budget: float):
        """Set up budget alerts for cost monitoring."""
        def check_budget():
            today_usage = [
                u for u in self.usage_log
                if u.timestamp.date() == datetime.now().date()
            ]
            today_cost = sum(u.estimated_cost for u in today_usage)

            if today_cost >= daily_budget * 0.8:  # 80% threshold
                print(f"⚠️  Budget Alert: ${today_cost:.4f} of ${daily_budget:.2f} daily budget used")

            if today_cost >= daily_budget:
                print(f"🚨 Budget Exceeded: ${today_cost:.4f} over ${daily_budget:.2f} daily limit!")

        # In real implementation, this would set up periodic checking
        check_budget()


# Convenience functions for common token counting scenarios
def count_tokens(text: str, model_provider: ModelProvider = ModelProvider.OPENAI_GPT4) -> int:
    """Quick function to count tokens in text."""
    counter = TokenCounter(default_model=model_provider, enable_logging=False)
    return counter.count_prompt_tokens(text)


def estimate_cost(prompt: str, expected_response_tokens: int = 500,
                 model_provider: ModelProvider = ModelProvider.OPENAI_GPT4) -> float:
    """Quick function to estimate cost for a prompt."""
    counter = TokenCounter(default_model=model_provider, enable_logging=False)
    input_tokens = counter.count_prompt_tokens(prompt)

    pricing = counter.model_pricing[model_provider]
    return pricing.calculate_cost(input_tokens, expected_response_tokens)


def compare_model_costs(prompt: str, expected_response_tokens: int = 500) -> Dict[str, float]:
    """Compare costs across different models for the same prompt."""
    costs = {}

    for provider in ModelProvider:
        try:
            cost = estimate_cost(prompt, expected_response_tokens, provider)
            costs[provider.value] = cost
        except Exception:
            continue

    return dict(sorted(costs.items(), key=lambda x: x[1]))


def analyze_prompt_efficiency(prompt: str, response: str,
                            model_provider: ModelProvider = ModelProvider.OPENAI_GPT4) -> Dict[str, Any]:
    """Analyze the efficiency of a prompt-response pair."""
    counter = TokenCounter(default_model=model_provider, enable_logging=False)

    input_tokens = counter.count_prompt_tokens(prompt)
    output_tokens = counter.count_response_tokens(response)

    pricing = counter.model_pricing[model_provider]
    cost = pricing.calculate_cost(input_tokens, output_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": cost,
        "efficiency_score": output_tokens / input_tokens if input_tokens > 0 else 0,
        "cost_per_output_token": cost / output_tokens if output_tokens > 0 else 0,
        "characters_per_token": {
            "input": len(prompt) / input_tokens if input_tokens > 0 else 0,
            "output": len(response) / output_tokens if output_tokens > 0 else 0
        }
    }


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Token Counter Demo ===\n")

    # Initialize token counter with logging enabled
    counter = TokenCounter(
        default_model=ModelProvider.OPENAI_GPT4,
        enable_logging=True,
        enable_real_time_display=False
    )

    print("1. Model Pricing Overview:")
    for provider, pricing in counter.model_pricing.items():
        print(f"   {provider.value}:")
        print(f"     Input: ${pricing.input_token_cost:.4f}/1K tokens")
        print(f"     Output: ${pricing.output_token_cost:.4f}/1K tokens")
        print(f"     Context: {pricing.context_window:,} tokens")
    print()

    print("2. Token Counting Examples:")

    # Simple text
    simple_text = "Hello, how are you today?"
    simple_tokens = counter.count_prompt_tokens(simple_text)
    print(f"   Simple text: '{simple_text}'")
    print(f"   Tokens: {simple_tokens} ({len(simple_text)} chars, {len(simple_text)/simple_tokens:.1f} chars/token)")

    # Complex prompt
    complex_prompt = """
    Analyze the following project retrospective and identify:
    1. Key patterns in team performance
    2. Root causes of identified issues  
    3. Strategic recommendations for improvement
    
    Project Context: VaultMind development, Q3 2025, team of 6 engineers
    
    Retrospective Data:
    - Delivered 85% of planned features
    - Technical debt increased 40% 
    - Team satisfaction: 7.2/10
    - User feedback: 92% positive
    - Integration issues caused 2-week delay
    """

    complex_tokens = counter.count_prompt_tokens(complex_prompt)
    print(f"\n   Complex prompt: {complex_tokens} tokens ({len(complex_prompt)} chars)")
    print(f"   Estimated cost (with 300 token response): ${estimate_cost(complex_prompt, 300):.4f}")
    print()

    print("3. Simulated AI Interactions:")

    # Simulate different prompting strategies
    strategies_data = [
        (PromptingStrategy.ZERO_SHOT, "Analyze this meeting note.", "Basic analysis of meeting content with key points identified.", 0.8),
        (PromptingStrategy.ONE_SHOT, complex_prompt, "Comprehensive analysis following example pattern with detailed insights and recommendations.", 2.3),
        (PromptingStrategy.MULTI_SHOT, complex_prompt + "\n\nExample patterns from 4 previous analyses...", "Multi-perspective analysis showing pattern variations and comprehensive insights across multiple dimensions.", 3.1),
        (PromptingStrategy.CHAIN_OF_THOUGHT, complex_prompt, "Step-by-step analysis: 1) Observe key patterns... 2) Hypothesize causes... 3) Gather evidence... [continues with 8 more reasoning steps]", 4.2),
        (PromptingStrategy.DYNAMIC, "Personalized analysis based on your vault patterns: " + complex_prompt, "Contextually adapted analysis incorporating user's historical patterns and preferences.", 2.7)
    ]

    for strategy, prompt, response, processing_time in strategies_data:
        usage = counter.log_ai_interaction(
            prompt=prompt,
            response=response,
            strategy=strategy,
            processing_time=processing_time,
            metadata={"test_scenario": True}
        )

    print("\n4. Session Summary:")
    session = counter.get_session_summary()
    print(f"   Session ID: {session.session_id}")
    print(f"   Total Requests: {session.total_requests}")
    print(f"   Total Cost: ${session.total_cost:.4f}")
    print(f"   Total Tokens: {session.total_input_tokens + session.total_output_tokens:,}")
    print(f"   Average Tokens/Request: {session.average_tokens_per_request:.0f}")
    print()

    print("5. Cost Breakdown by Strategy:")
    breakdown = counter.get_cost_breakdown_by_strategy()
    for strategy, data in sorted(breakdown.items(), key=lambda x: x[1]['total_cost'], reverse=True):
        print(f"   {strategy}:")
        print(f"     Total Cost: ${data['total_cost']:.4f}")
        print(f"     Avg Cost/Request: ${data['avg_cost_per_request']:.4f}")
        print(f"     Avg Tokens/Request: {data['avg_tokens_per_request']:.0f}")
    print()

    print("6. Model Cost Comparison:")
    sample_prompt = "Analyze the sentiment and extract key insights from this journal entry about career transitions."
    model_costs = compare_model_costs(sample_prompt, 400)

    print(f"   Sample prompt cost comparison (400 token response):")
    for model, cost in model_costs.items():
        savings = ((max(model_costs.values()) - cost) / max(model_costs.values())) * 100
        print(f"     {model}: ${cost:.4f} ({savings:.0f}% savings vs most expensive)")
    print()

    print("7. Optimization Recommendations:")
    optimization = counter.optimize_token_usage()
    print("   Recommendations:")
    for rec in optimization['recommendations']:
        print(f"     • {rec}")
    print()

    print("8. Quick Token Analysis:")
    efficiency_analysis = analyze_prompt_efficiency(
        prompt=complex_prompt,
        response="Detailed strategic analysis with actionable recommendations and comprehensive insights.",
        model_provider=ModelProvider.OPENAI_GPT4
    )

    print(f"   Efficiency Score: {efficiency_analysis['efficiency_score']:.2f}")
    print(f"   Cost per Output Token: ${efficiency_analysis['cost_per_output_token']:.6f}")
    print(f"   Input Chars/Token: {efficiency_analysis['characters_per_token']['input']:.1f}")
    print(f"   Output Chars/Token: {efficiency_analysis['characters_per_token']['output']:.1f}")

    print("\n=== Token Counting System Ready ===")
    print("Key capabilities:")
    print("• Real-time token counting with multiple tokenizer support")
    print("• Comprehensive cost tracking across 7+ AI models")
    print("• Strategy-based usage analytics and optimization")
    print("• Session management with detailed breakdowns")
    print("• Export capabilities (JSON, CSV, text reports)")
    print("• Budget monitoring and cost optimization recommendations")
