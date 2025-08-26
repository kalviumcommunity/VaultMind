"""
VaultMind Top P Control System

This module implements Top P (nucleus sampling) parameter control for managing
AI response quality, creativity, and consistency. Provides intelligent defaults
for different task types and integrates with temperature for balanced control.

Top P sampling selects from the smallest set of tokens whose cumulative
probability exceeds the threshold, providing more consistent quality control
than pure temperature-based sampling.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import json
import statistics
import math
from collections import defaultdict


class TaskType(Enum):
    """Different types of tasks requiring different Top P settings."""
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    CODE_GENERATION = "code_generation"
    FACTUAL_EXTRACTION = "factual_extraction"
    BRAINSTORMING = "brainstorming"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    REASONING = "reasoning"
    STORYTELLING = "storytelling"


class QualityProfile(Enum):
    """Quality profiles balancing creativity vs consistency."""
    DETERMINISTIC = "deterministic"        # Maximum consistency, minimal creativity
    CONSERVATIVE = "conservative"          # High consistency, low creativity
    BALANCED = "balanced"                 # Moderate balance
    CREATIVE = "creative"                 # High creativity, moderate consistency
    EXPERIMENTAL = "experimental"         # Maximum creativity, variable consistency


class SamplingStrategy(Enum):
    """Different sampling strategies for response generation."""
    NUCLEUS_ONLY = "nucleus_only"         # Pure Top P sampling
    TEMPERATURE_ONLY = "temperature_only" # Pure temperature sampling
    HYBRID = "hybrid"                     # Combined Top P + temperature
    ADAPTIVE = "adaptive"                 # Dynamic adjustment based on context
    CONSTRAINED = "constrained"           # Additional constraints (Top K, etc.)


@dataclass
class SamplingParameters:
    """Complete set of sampling parameters for response generation."""
    top_p: float = 0.9
    temperature: float = 0.7
    top_k: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None

    def validate(self) -> bool:
        """Validate parameter ranges."""
        if not (0.0 <= self.top_p <= 1.0):
            return False
        if not (0.0 <= self.temperature <= 2.0):
            return False
        if self.top_k is not None and self.top_k < 1:
            return False
        if not (-2.0 <= self.frequency_penalty <= 2.0):
            return False
        if not (-2.0 <= self.presence_penalty <= 2.0):
            return False
        if not (0.1 <= self.repetition_penalty <= 2.0):
            return False
        return True

    def get_creativity_score(self) -> float:
        """Calculate overall creativity score from parameters."""
        # Higher top_p and temperature = more creativity
        creativity = (self.top_p * 0.4) + (self.temperature * 0.6)

        # Penalties reduce creativity
        creativity -= abs(self.frequency_penalty) * 0.1
        creativity -= abs(self.presence_penalty) * 0.1
        creativity -= abs(self.repetition_penalty - 1.0) * 0.1

        return max(0.0, min(1.0, creativity))

    def get_consistency_score(self) -> float:
        """Calculate expected consistency score from parameters."""
        # Lower values = higher consistency
        consistency = 1.0 - self.get_creativity_score()
        return max(0.0, min(1.0, consistency))


@dataclass
class QualityMetrics:
    """Metrics for evaluating response quality under different parameters."""
    parameter_set: SamplingParameters
    task_type: TaskType
    sample_count: int = 0
    average_quality: float = 0.0
    quality_variance: float = 0.0
    creativity_rating: float = 0.0
    consistency_rating: float = 0.0
    user_satisfaction: float = 0.0
    response_times: List[float] = field(default_factory=list)
    token_efficiency: float = 0.0

    def update_metrics(self, quality_score: float, creativity: float,
                      consistency: float, response_time: float):
        """Update metrics with new measurement."""
        self.sample_count += 1

        # Update running averages
        prev_avg = self.average_quality
        self.average_quality = prev_avg + (quality_score - prev_avg) / self.sample_count

        # Update variance (simplified)
        if self.sample_count > 1:
            self.quality_variance = ((self.sample_count - 2) * self.quality_variance +
                                   (quality_score - prev_avg) * (quality_score - self.average_quality)) / (self.sample_count - 1)

        # Update other metrics
        self.creativity_rating += (creativity - self.creativity_rating) / self.sample_count
        self.consistency_rating += (consistency - self.consistency_rating) / self.sample_count
        self.response_times.append(response_time)


@dataclass
class TaskConfiguration:
    """Configuration for a specific task type."""
    task_type: TaskType
    optimal_top_p: float
    optimal_temperature: float
    top_p_range: Tuple[float, float]
    temperature_range: Tuple[float, float]
    quality_profile: QualityProfile
    description: str
    use_cases: List[str] = field(default_factory=list)

    def get_sampling_parameters(self, creativity_adjustment: float = 0.0) -> SamplingParameters:
        """Get sampling parameters with optional creativity adjustment."""
        adjusted_top_p = max(self.top_p_range[0],
                           min(self.top_p_range[1],
                               self.optimal_top_p + creativity_adjustment))

        adjusted_temperature = max(self.temperature_range[0],
                                 min(self.temperature_range[1],
                                     self.optimal_temperature + creativity_adjustment))

        return SamplingParameters(
            top_p=adjusted_top_p,
            temperature=adjusted_temperature
        )


class TopPManager:
    """
    Main class for managing Top P and related sampling parameters.

    Provides intelligent defaults for different task types, quality profiles,
    and adaptive parameter adjustment based on performance feedback.
    """

    def __init__(self):
        self.task_configurations = self._initialize_task_configurations()
        self.quality_profiles = self._initialize_quality_profiles()
        self.performance_history = defaultdict(list)
        self.user_preferences = {}

    def _initialize_task_configurations(self) -> Dict[TaskType, TaskConfiguration]:
        """Initialize optimal configurations for different task types."""
        return {
            TaskType.ANALYSIS: TaskConfiguration(
                task_type=TaskType.ANALYSIS,
                optimal_top_p=0.9,
                optimal_temperature=0.3,
                top_p_range=(0.85, 0.95),
                temperature_range=(0.1, 0.5),
                quality_profile=QualityProfile.CONSERVATIVE,
                description="Analytical tasks requiring consistency and factual accuracy",
                use_cases=["Data analysis", "Research synthesis", "Report generation", "Pattern identification"]
            ),

            TaskType.CREATIVE_WRITING: TaskConfiguration(
                task_type=TaskType.CREATIVE_WRITING,
                optimal_top_p=0.95,
                optimal_temperature=0.8,
                top_p_range=(0.9, 0.98),
                temperature_range=(0.6, 1.0),
                quality_profile=QualityProfile.CREATIVE,
                description="Creative tasks benefiting from high variability and novelty",
                use_cases=["Story writing", "Poetry", "Creative brainstorming", "Artistic content"]
            ),

            TaskType.CODE_GENERATION: TaskConfiguration(
                task_type=TaskType.CODE_GENERATION,
                optimal_top_p=0.85,
                optimal_temperature=0.2,
                top_p_range=(0.8, 0.9),
                temperature_range=(0.1, 0.4),
                quality_profile=QualityProfile.DETERMINISTIC,
                description="Code generation requiring high precision and syntactic correctness",
                use_cases=["Function generation", "Bug fixing", "Code completion", "Algorithm implementation"]
            ),

            TaskType.FACTUAL_EXTRACTION: TaskConfiguration(
                task_type=TaskType.FACTUAL_EXTRACTION,
                optimal_top_p=0.8,
                optimal_temperature=0.1,
                top_p_range=(0.75, 0.85),
                temperature_range=(0.05, 0.2),
                quality_profile=QualityProfile.DETERMINISTIC,
                description="Factual extraction requiring maximum consistency and accuracy",
                use_cases=["Information extraction", "Data parsing", "Entity recognition", "Fact verification"]
            ),

            TaskType.BRAINSTORMING: TaskConfiguration(
                task_type=TaskType.BRAINSTORMING,
                optimal_top_p=0.98,
                optimal_temperature=0.9,
                top_p_range=(0.95, 1.0),
                temperature_range=(0.7, 1.2),
                quality_profile=QualityProfile.EXPERIMENTAL,
                description="Brainstorming tasks benefiting from maximum creativity and idea diversity",
                use_cases=["Idea generation", "Problem solving", "Innovation workshops", "Creative exploration"]
            ),

            TaskType.TECHNICAL_DOCUMENTATION: TaskConfiguration(
                task_type=TaskType.TECHNICAL_DOCUMENTATION,
                optimal_top_p=0.88,
                optimal_temperature=0.3,
                top_p_range=(0.85, 0.92),
                temperature_range=(0.2, 0.4),
                quality_profile=QualityProfile.CONSERVATIVE,
                description="Technical writing requiring clarity, precision, and consistency",
                use_cases=["API documentation", "User guides", "Technical specifications", "Process documentation"]
            ),

            TaskType.CONVERSATION: TaskConfiguration(
                task_type=TaskType.CONVERSATION,
                optimal_top_p=0.92,
                optimal_temperature=0.7,
                top_p_range=(0.88, 0.96),
                temperature_range=(0.5, 0.9),
                quality_profile=QualityProfile.BALANCED,
                description="Conversational tasks balancing naturalness with consistency",
                use_cases=["Chat responses", "Customer service", "Personal assistance", "Interactive dialogue"]
            ),

            TaskType.SUMMARIZATION: TaskConfiguration(
                task_type=TaskType.SUMMARIZATION,
                optimal_top_p=0.87,
                optimal_temperature=0.25,
                top_p_range=(0.82, 0.92),
                temperature_range=(0.15, 0.4),
                quality_profile=QualityProfile.CONSERVATIVE,
                description="Summarization requiring factual accuracy and comprehensive coverage",
                use_cases=["Document summarization", "Meeting notes", "Article abstracts", "Key point extraction"]
            ),

            TaskType.REASONING: TaskConfiguration(
                task_type=TaskType.REASONING,
                optimal_top_p=0.85,
                optimal_temperature=0.3,
                top_p_range=(0.8, 0.9),
                temperature_range=(0.2, 0.5),
                quality_profile=QualityProfile.CONSERVATIVE,
                description="Logical reasoning requiring step-by-step consistency",
                use_cases=["Problem solving", "Logical deduction", "Mathematical reasoning", "Causal analysis"]
            ),

            TaskType.STORYTELLING: TaskConfiguration(
                task_type=TaskType.STORYTELLING,
                optimal_top_p=0.94,
                optimal_temperature=0.75,
                top_p_range=(0.9, 0.97),
                temperature_range=(0.6, 0.9),
                quality_profile=QualityProfile.CREATIVE,
                description="Storytelling balancing creativity with narrative coherence",
                use_cases=["Narrative generation", "Character development", "Plot creation", "Descriptive writing"]
            )
        }

    def _initialize_quality_profiles(self) -> Dict[QualityProfile, SamplingParameters]:
        """Initialize quality profiles with different parameter combinations."""
        return {
            QualityProfile.DETERMINISTIC: SamplingParameters(
                top_p=0.8, temperature=0.1, frequency_penalty=0.1, presence_penalty=0.0
            ),
            QualityProfile.CONSERVATIVE: SamplingParameters(
                top_p=0.87, temperature=0.3, frequency_penalty=0.0, presence_penalty=0.0
            ),
            QualityProfile.BALANCED: SamplingParameters(
                top_p=0.92, temperature=0.7, frequency_penalty=0.0, presence_penalty=0.0
            ),
            QualityProfile.CREATIVE: SamplingParameters(
                top_p=0.95, temperature=0.8, frequency_penalty=-0.1, presence_penalty=0.1
            ),
            QualityProfile.EXPERIMENTAL: SamplingParameters(
                top_p=0.98, temperature=0.9, frequency_penalty=-0.2, presence_penalty=0.2
            )
        }

    def get_optimal_parameters(self,
                             task_type: TaskType,
                             quality_profile: QualityProfile = None,
                             creativity_boost: float = 0.0,
                             consistency_boost: float = 0.0) -> SamplingParameters:
        """
        Get optimal sampling parameters for a specific task and quality requirements.

        Args:
            task_type: The type of task being performed
            quality_profile: Desired quality profile (overrides task default if provided)
            creativity_boost: Adjustment to increase creativity (-0.2 to +0.2)
            consistency_boost: Adjustment to increase consistency (-0.2 to +0.2)

        Returns:
            Optimized SamplingParameters for the task
        """
        if task_type not in self.task_configurations:
            raise ValueError(f"Unknown task type: {task_type}")

        config = self.task_configurations[task_type]

        # Use provided quality profile or task default
        profile = quality_profile or config.quality_profile
        base_params = self.quality_profiles[profile].copy() if hasattr(self.quality_profiles[profile], 'copy') else SamplingParameters(
            top_p=self.quality_profiles[profile].top_p,
            temperature=self.quality_profiles[profile].temperature,
            frequency_penalty=self.quality_profiles[profile].frequency_penalty,
            presence_penalty=self.quality_profiles[profile].presence_penalty
        )

        # Apply task-specific optimizations
        base_params.top_p = config.optimal_top_p
        base_params.temperature = config.optimal_temperature

        # Apply creativity/consistency adjustments
        net_adjustment = creativity_boost - consistency_boost

        # Adjust Top P (primary creativity control)
        adjusted_top_p = base_params.top_p + (net_adjustment * 0.05)
        base_params.top_p = max(config.top_p_range[0],
                               min(config.top_p_range[1], adjusted_top_p))

        # Adjust temperature (secondary creativity control)
        adjusted_temp = base_params.temperature + (net_adjustment * 0.1)
        base_params.temperature = max(config.temperature_range[0],
                                    min(config.temperature_range[1], adjusted_temp))

        # Validate final parameters
        if not base_params.validate():
            raise ValueError("Generated invalid sampling parameters")

        return base_params

    def explain_parameters(self, params: SamplingParameters, task_type: TaskType = None) -> Dict[str, Any]:
        """
        Explain what the sampling parameters mean and their expected effects.

        Args:
            params: The sampling parameters to explain
            task_type: Context task type for more specific explanations

        Returns:
            Detailed explanation of parameter effects
        """
        explanation = {
            "parameter_analysis": {},
            "expected_behavior": {},
            "quality_predictions": {},
            "recommendations": []
        }

        # Analyze Top P
        if params.top_p >= 0.95:
            explanation["parameter_analysis"]["top_p"] = {
                "value": params.top_p,
                "interpretation": "High diversity - considers most of probability mass",
                "effect": "More creative and varied responses, higher novelty"
            }
        elif params.top_p >= 0.85:
            explanation["parameter_analysis"]["top_p"] = {
                "value": params.top_p,
                "interpretation": "Moderate diversity - balanced token selection",
                "effect": "Good balance between creativity and consistency"
            }
        else:
            explanation["parameter_analysis"]["top_p"] = {
                "value": params.top_p,
                "interpretation": "Low diversity - focuses on high probability tokens",
                "effect": "More consistent and predictable responses"
            }

        # Analyze Temperature
        if params.temperature >= 0.8:
            explanation["parameter_analysis"]["temperature"] = {
                "value": params.temperature,
                "interpretation": "High randomness - flattened probability distribution",
                "effect": "More unexpected word choices, creative combinations"
            }
        elif params.temperature >= 0.3:
            explanation["parameter_analysis"]["temperature"] = {
                "value": params.temperature,
                "interpretation": "Moderate randomness - balanced selection",
                "effect": "Natural variety without excessive unpredictability"
            }
        else:
            explanation["parameter_analysis"]["temperature"] = {
                "value": params.temperature,
                "interpretation": "Low randomness - peaked probability distribution",
                "effect": "Highly consistent, predictable token selection"
            }

        # Expected behavior
        creativity_score = params.get_creativity_score()
        consistency_score = params.get_consistency_score()

        explanation["expected_behavior"] = {
            "creativity_level": "High" if creativity_score > 0.7 else "Medium" if creativity_score > 0.4 else "Low",
            "consistency_level": "High" if consistency_score > 0.7 else "Medium" if consistency_score > 0.4 else "Low",
            "response_variability": "High" if creativity_score > 0.6 else "Medium" if creativity_score > 0.3 else "Low",
            "factual_reliability": "High" if consistency_score > 0.6 else "Medium" if consistency_score > 0.3 else "Low"
        }

        # Quality predictions
        explanation["quality_predictions"] = {
            "novelty": creativity_score,
            "coherence": consistency_score,
            "appropriateness": min(creativity_score, consistency_score) + 0.2,
            "user_satisfaction": (creativity_score + consistency_score) / 2
        }

        # Task-specific recommendations
        if task_type:
            config = self.task_configurations.get(task_type)
            if config:
                if abs(params.top_p - config.optimal_top_p) > 0.05:
                    explanation["recommendations"].append(
                        f"Consider adjusting Top P closer to {config.optimal_top_p:.2f} for optimal {task_type.value} performance"
                    )

                if abs(params.temperature - config.optimal_temperature) > 0.1:
                    explanation["recommendations"].append(
                        f"Consider adjusting temperature closer to {config.optimal_temperature:.2f} for {task_type.value} tasks"
                    )

        return explanation

    def compare_parameter_sets(self, param_sets: List[Tuple[str, SamplingParameters]]) -> Dict[str, Any]:
        """
        Compare multiple parameter sets and their expected performance.

        Args:
            param_sets: List of (name, parameters) tuples to compare

        Returns:
            Detailed comparison analysis
        """
        comparison = {
            "parameter_comparison": {},
            "creativity_ranking": [],
            "consistency_ranking": [],
            "recommendations": {}
        }

        creativity_scores = []
        consistency_scores = []

        for name, params in param_sets:
            creativity = params.get_creativity_score()
            consistency = params.get_consistency_score()

            comparison["parameter_comparison"][name] = {
                "top_p": params.top_p,
                "temperature": params.temperature,
                "creativity_score": creativity,
                "consistency_score": consistency,
                "balance_score": abs(creativity - consistency)  # Lower is more balanced
            }

            creativity_scores.append((name, creativity))
            consistency_scores.append((name, consistency))

        # Create rankings
        comparison["creativity_ranking"] = sorted(creativity_scores, key=lambda x: x[1], reverse=True)
        comparison["consistency_ranking"] = sorted(consistency_scores, key=lambda x: x[1], reverse=True)

        # Generate recommendations
        most_creative = comparison["creativity_ranking"][0]
        most_consistent = comparison["consistency_ranking"][0]

        comparison["recommendations"] = {
            "for_creative_tasks": f"Use '{most_creative[0]}' - highest creativity score ({most_creative[1]:.2f})",
            "for_analytical_tasks": f"Use '{most_consistent[0]}' - highest consistency score ({most_consistent[1]:.2f})",
            "most_balanced": min(param_sets, key=lambda x: abs(x[1].get_creativity_score() - x[1].get_consistency_score()))[0]
        }

        return comparison

    def adaptive_adjustment(self,
                          current_params: SamplingParameters,
                          performance_feedback: Dict[str, float],
                          task_type: TaskType) -> SamplingParameters:
        """
        Adaptively adjust parameters based on performance feedback.

        Args:
            current_params: Current sampling parameters
            performance_feedback: Dictionary with quality metrics
            task_type: Type of task being performed

        Returns:
            Adjusted sampling parameters
        """
        adjusted_params = SamplingParameters(
            top_p=current_params.top_p,
            temperature=current_params.temperature,
            frequency_penalty=current_params.frequency_penalty,
            presence_penalty=current_params.presence_penalty
        )

        # Extract feedback metrics
        quality_score = performance_feedback.get("quality", 0.5)
        creativity_score = performance_feedback.get("creativity", 0.5)
        consistency_score = performance_feedback.get("consistency", 0.5)
        user_satisfaction = performance_feedback.get("satisfaction", 0.5)

        # Calculate adjustment magnitude based on performance
        adjustment_strength = 1.0 - user_satisfaction  # Higher dissatisfaction = larger adjustments

        config = self.task_configurations[task_type]

        # Adjust based on specific feedback
        if quality_score < 0.6:  # Quality too low
            # Move towards more conservative settings
            target_top_p = min(current_params.top_p, config.optimal_top_p - 0.02)
            target_temp = min(current_params.temperature, config.optimal_temperature - 0.05)

            adjusted_params.top_p = current_params.top_p + (target_top_p - current_params.top_p) * adjustment_strength
            adjusted_params.temperature = current_params.temperature + (target_temp - current_params.temperature) * adjustment_strength

        elif creativity_score < 0.4 and task_type in [TaskType.CREATIVE_WRITING, TaskType.BRAINSTORMING]:
            # Increase creativity for creative tasks
            target_top_p = min(0.98, current_params.top_p + 0.03)
            target_temp = min(1.0, current_params.temperature + 0.1)

            adjusted_params.top_p = current_params.top_p + (target_top_p - current_params.top_p) * adjustment_strength
            adjusted_params.temperature = current_params.temperature + (target_temp - current_params.temperature) * adjustment_strength

        elif consistency_score < 0.4 and task_type in [TaskType.ANALYSIS, TaskType.CODE_GENERATION]:
            # Increase consistency for analytical tasks
            target_top_p = max(0.75, current_params.top_p - 0.03)
            target_temp = max(0.1, current_params.temperature - 0.1)

            adjusted_params.top_p = current_params.top_p + (target_top_p - current_params.top_p) * adjustment_strength
            adjusted_params.temperature = current_params.temperature + (target_temp - current_params.temperature) * adjustment_strength

        # Ensure parameters stay within valid ranges
        config = self.task_configurations[task_type]
        adjusted_params.top_p = max(config.top_p_range[0], min(config.top_p_range[1], adjusted_params.top_p))
        adjusted_params.temperature = max(config.temperature_range[0], min(config.temperature_range[1], adjusted_params.temperature))

        # Validate final parameters
        if not adjusted_params.validate():
            return current_params  # Return original if adjustment created invalid params

        return adjusted_params

    def get_task_recommendations(self, task_description: str) -> List[Tuple[TaskType, float]]:
        """
        Recommend task types based on task description.

        Args:
            task_description: Natural language description of the task

        Returns:
            List of (TaskType, confidence) tuples, sorted by relevance
        """
        keywords = task_description.lower().split()
        task_scores = defaultdict(float)

        # Keyword matching for task type identification
        task_keywords = {
            TaskType.ANALYSIS: ["analyze", "analysis", "examine", "study", "investigate", "research", "evaluate"],
            TaskType.CREATIVE_WRITING: ["write", "story", "creative", "fiction", "narrative", "poem", "novel"],
            TaskType.CODE_GENERATION: ["code", "program", "function", "algorithm", "script", "debug", "implement"],
            TaskType.FACTUAL_EXTRACTION: ["extract", "facts", "data", "information", "details", "parse", "find"],
            TaskType.BRAINSTORMING: ["brainstorm", "ideas", "creative", "innovative", "generate", "think", "ideate"],
            TaskType.TECHNICAL_DOCUMENTATION: ["document", "guide", "manual", "documentation", "instructions", "how-to"],
            TaskType.CONVERSATION: ["chat", "talk", "discuss", "conversation", "dialogue", "respond"],
            TaskType.SUMMARIZATION: ["summary", "summarize", "abstract", "overview", "brief", "condensed"],
            TaskType.REASONING: ["reason", "logic", "solve", "problem", "deduce", "infer", "conclude"],
            TaskType.STORYTELLING: ["story", "tell", "narrative", "plot", "character", "adventure"]
        }

        for task_type, task_kw in task_keywords.items():
            for keyword in keywords:
                if keyword in task_kw:
                    task_scores[task_type] += 1.0
                # Partial matches
                for task_word in task_kw:
                    if keyword in task_word or task_word in keyword:
                        task_scores[task_type] += 0.5

        # Normalize scores and return top matches
        if not task_scores:
            return [(TaskType.CONVERSATION, 0.5)]  # Default fallback

        max_score = max(task_scores.values())
        normalized_scores = [(task, score / max_score) for task, score in task_scores.items()]

        return sorted(normalized_scores, key=lambda x: x[1], reverse=True)[:3]

    def log_performance(self,
                       params: SamplingParameters,
                       task_type: TaskType,
                       quality_metrics: Dict[str, float]):
        """Log performance data for future optimization."""
        metrics = QualityMetrics(
            parameter_set=params,
            task_type=task_type
        )

        metrics.update_metrics(
            quality_score=quality_metrics.get("quality", 0.5),
            creativity=quality_metrics.get("creativity", 0.5),
            consistency=quality_metrics.get("consistency", 0.5),
            response_time=quality_metrics.get("response_time", 1.0)
        )

        self.performance_history[task_type].append(metrics)

    def export_configuration(self, format: str = "json") -> str:
        """Export current configuration and performance data."""
        if format == "json":
            export_data = {
                "task_configurations": {
                    task_type.value: {
                        "optimal_top_p": config.optimal_top_p,
                        "optimal_temperature": config.optimal_temperature,
                        "top_p_range": config.top_p_range,
                        "temperature_range": config.temperature_range,
                        "quality_profile": config.quality_profile.value,
                        "description": config.description,
                        "use_cases": config.use_cases
                    }
                    for task_type, config in self.task_configurations.items()
                },
                "quality_profiles": {
                    profile.value: {
                        "top_p": params.top_p,
                        "temperature": params.temperature,
                        "frequency_penalty": params.frequency_penalty,
                        "presence_penalty": params.presence_penalty
                    }
                    for profile, params in self.quality_profiles.items()
                },
                "performance_summary": {
                    task_type.value: len(metrics_list)
                    for task_type, metrics_list in self.performance_history.items()
                }
            }

            return json.dumps(export_data, indent=2)

        else:
            # Text format
            lines = [
                "VaultMind Top P Configuration",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Task Type Configurations:",
                ""
            ]

            for task_type, config in self.task_configurations.items():
                lines.extend([
                    f"**{task_type.value.title()}**:",
                    f"  Optimal Top P: {config.optimal_top_p}",
                    f"  Optimal Temperature: {config.optimal_temperature}",
                    f"  Quality Profile: {config.quality_profile.value}",
                    f"  Description: {config.description}",
                    ""
                ])

            return "\n".join(lines)


# Convenience functions for common Top P scenarios
def get_analysis_parameters(creativity_level: str = "low") -> SamplingParameters:
    """Quick function to get parameters optimized for analysis tasks."""
    manager = TopPManager()

    creativity_adjustments = {
        "low": -0.1,
        "medium": 0.0,
        "high": 0.1
    }

    return manager.get_optimal_parameters(
        TaskType.ANALYSIS,
        creativity_boost=creativity_adjustments.get(creativity_level, 0.0)
    )


def get_creative_parameters(consistency_level: str = "medium") -> SamplingParameters:
    """Quick function to get parameters optimized for creative tasks."""
    manager = TopPManager()

    consistency_adjustments = {
        "low": -0.1,
        "medium": 0.0,
        "high": 0.1
    }

    return manager.get_optimal_parameters(
        TaskType.CREATIVE_WRITING,
        consistency_boost=consistency_adjustments.get(consistency_level, 0.0)
    )


def compare_top_p_values(base_top_p: float, variations: List[float]) -> Dict[str, Any]:
    """Compare different Top P values for the same base configuration."""
    manager = TopPManager()
    base_params = SamplingParameters(top_p=base_top_p, temperature=0.7)

    param_sets = [(f"top_p_{base_top_p}", base_params)]

    for variation in variations:
        variant_params = SamplingParameters(top_p=variation, temperature=0.7)
        param_sets.append((f"top_p_{variation}", variant_params))

    return manager.compare_parameter_sets(param_sets)


def optimize_for_task(task_description: str) -> Dict[str, Any]:
    """Optimize parameters for a specific task description."""
    manager = TopPManager()

    # Get task recommendations
    task_recommendations = manager.get_task_recommendations(task_description)

    if not task_recommendations:
        return {"error": "Could not identify suitable task type"}

    # Use most likely task type
    best_task, confidence = task_recommendations[0]

    # Get optimal parameters
    optimal_params = manager.get_optimal_parameters(best_task)

    return {
        "task_description": task_description,
        "recommended_task_type": best_task.value,
        "confidence": confidence,
        "optimal_parameters": {
            "top_p": optimal_params.top_p,
            "temperature": optimal_params.temperature,
            "frequency_penalty": optimal_params.frequency_penalty,
            "presence_penalty": optimal_params.presence_penalty
        },
        "explanation": manager.explain_parameters(optimal_params, best_task),
        "alternative_tasks": [(task.value, conf) for task, conf in task_recommendations[1:]]
    }


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Top P Control System Demo ===\n")

    # Initialize Top P manager
    manager = TopPManager()

    print("1. Task Configuration Overview:")
    print(f"   Configured task types: {len(manager.task_configurations)}")

    # Show a few key configurations
    key_tasks = [TaskType.ANALYSIS, TaskType.CREATIVE_WRITING, TaskType.CODE_GENERATION]
    for task in key_tasks:
        config = manager.task_configurations[task]
        print(f"   {task.value}:")
        print(f"     Top P: {config.optimal_top_p}, Temperature: {config.optimal_temperature}")
        print(f"     Profile: {config.quality_profile.value}")
    print()

    print("2. Quality Profile Comparison:")
    profiles_to_compare = [
        ("Conservative", manager.quality_profiles[QualityProfile.CONSERVATIVE]),
        ("Balanced", manager.quality_profiles[QualityProfile.BALANCED]),
        ("Creative", manager.quality_profiles[QualityProfile.CREATIVE])
    ]

    comparison = manager.compare_parameter_sets(profiles_to_compare)
    print("   Creativity ranking:")
    for i, (name, score) in enumerate(comparison["creativity_ranking"], 1):
        print(f"     {i}. {name}: {score:.3f}")
    print()

    print("3. Task-Specific Optimization Examples:")

    # Analysis task
    analysis_params = get_analysis_parameters("low")
    print(f"   Analysis (low creativity): Top P = {analysis_params.top_p}, Temp = {analysis_params.temperature}")

    # Creative task
    creative_params = get_creative_parameters("medium")
    print(f"   Creative (medium consistency): Top P = {creative_params.top_p}, Temp = {creative_params.temperature}")

    # Code generation
    code_params = manager.get_optimal_parameters(TaskType.CODE_GENERATION)
    print(f"   Code generation: Top P = {code_params.top_p}, Temp = {code_params.temperature}")
    print()

    print("4. Parameter Explanation Example:")
    sample_params = manager.get_optimal_parameters(TaskType.BRAINSTORMING)
    explanation = manager.explain_parameters(sample_params, TaskType.BRAINSTORMING)

    print(f"   Brainstorming parameters: Top P = {sample_params.top_p}, Temp = {sample_params.temperature}")
    print(f"   Expected creativity level: {explanation['expected_behavior']['creativity_level']}")
    print(f"   Expected consistency level: {explanation['expected_behavior']['consistency_level']}")
    print(f"   Top P interpretation: {explanation['parameter_analysis']['top_p']['interpretation']}")
    print()

    print("5. Automatic Task Detection:")
    test_descriptions = [
        "Write a creative story about space exploration",
        "Analyze the quarterly sales data for trends",
        "Generate Python code for sorting algorithms",
        "Brainstorm innovative product ideas for sustainability"
    ]

    for desc in test_descriptions:
        optimization = optimize_for_task(desc)
        print(f"   '{desc[:40]}...'")
        print(f"     → {optimization['recommended_task_type']} (confidence: {optimization['confidence']:.2f})")
        print(f"     → Top P: {optimization['optimal_parameters']['top_p']}, Temp: {optimization['optimal_parameters']['temperature']}")
    print()

    print("6. Top P Value Comparison:")
    top_p_comparison = compare_top_p_values(0.9, [0.8, 0.95, 0.99])
    print("   Creativity ranking by Top P value:")
    for i, (name, score) in enumerate(top_p_comparison["creativity_ranking"], 1):
        creativity_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
        print(f"     {i}. {name}: {score:.3f} ({creativity_level})")
    print()

    print("7. Adaptive Adjustment Demo:")
    current_params = manager.get_optimal_parameters(TaskType.ANALYSIS)
    print(f"   Current analysis parameters: Top P = {current_params.top_p}, Temp = {current_params.temperature}")

    # Simulate poor performance feedback
    poor_feedback = {
        "quality": 0.3,
        "creativity": 0.4,
        "consistency": 0.2,
        "satisfaction": 0.3
    }

    adjusted_params = manager.adaptive_adjustment(current_params, poor_feedback, TaskType.ANALYSIS)
    print(f"   After poor feedback: Top P = {adjusted_params.top_p}, Temp = {adjusted_params.temperature}")
    print(f"   Adjustment: More conservative settings for better consistency")
    print()

    print("8. Configuration Export Preview:")
    config_export = manager.export_configuration("json")
    config_data = json.loads(config_export)
    print(f"   Exportable configurations: {len(config_data['task_configurations'])} tasks")
    print(f"   Quality profiles: {len(config_data['quality_profiles'])}")
    print(f"   Configuration size: {len(config_export):,} characters")

    print("\n=== Top P Control System Ready ===")
    print("Key capabilities:")
    print("• Task-specific Top P and temperature optimization")
    print("• Quality profile management (deterministic to experimental)")
    print("• Automatic task detection from descriptions")
    print("• Parameter explanation and effect prediction")
    print("• Adaptive adjustment based on performance feedback")
    print("• Comprehensive comparison and analysis tools")
