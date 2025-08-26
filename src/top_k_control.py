"""
VaultMind Top K Control System

This module implements Top K parameter control for vocabulary limitation in LLM
responses. Provides intelligent vocabulary control for different task types,
domain-specific optimization, and integration with Top P and temperature
for comprehensive response quality management.

Top K sampling limits the vocabulary to the K most probable tokens at each step,
providing focused control over response vocabulary and preventing low-probability
token selection that can degrade quality.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable, Set
from datetime import datetime
import json
import statistics
import math
from collections import defaultdict, Counter


class TaskType(Enum):
    """Different types of tasks requiring different Top K settings."""
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
    ACADEMIC_WRITING = "academic_writing"
    BUSINESS_COMMUNICATION = "business_communication"


class Domain(Enum):
    """Domain-specific contexts for vocabulary optimization."""
    GENERAL = "general"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    LEGAL = "legal"
    ACADEMIC = "academic"
    BUSINESS = "business"
    CREATIVE = "creative"
    SCIENTIFIC = "scientific"
    FINANCIAL = "financial"
    EDUCATIONAL = "educational"


class VocabularyProfile(Enum):
    """Vocabulary control profiles for different precision needs."""
    ULTRA_FOCUSED = "ultra_focused"      # top_k=20, maximum precision
    FOCUSED = "focused"                  # top_k=40, high precision
    BALANCED = "balanced"                # top_k=100, moderate precision
    DIVERSE = "diverse"                  # top_k=250, broad vocabulary
    UNLIMITED = "unlimited"              # No top_k, maximum creativity


class SamplingMode(Enum):
    """Different sampling modes combining Top K with other parameters."""
    TOP_K_ONLY = "top_k_only"           # Pure Top K sampling
    TOP_K_PLUS_P = "top_k_plus_p"       # Top K + Top P combination
    HYBRID = "hybrid"                    # Top K + temperature + penalties
    ADAPTIVE = "adaptive"                # Dynamic Top K based on context
    DOMAIN_TUNED = "domain_tuned"        # Domain-specific vocabulary control


@dataclass
class VocabularyMetrics:
    """Metrics for tracking vocabulary usage and diversity."""
    unique_tokens_used: Set[str] = field(default_factory=set)
    vocabulary_diversity: float = 0.0
    domain_relevance_score: float = 0.0
    repetition_rate: float = 0.0
    complexity_score: float = 0.0
    readability_score: float = 0.0

    def calculate_diversity(self, total_tokens: int) -> float:
        """Calculate vocabulary diversity ratio."""
        if total_tokens == 0:
            return 0.0
        return len(self.unique_tokens_used) / total_tokens

    def update_metrics(self, response_tokens: List[str], domain: Domain = Domain.GENERAL):
        """Update vocabulary metrics with new response."""
        self.unique_tokens_used.update(response_tokens)

        # Calculate basic diversity
        token_counts = Counter(response_tokens)
        total_tokens = len(response_tokens)

        if total_tokens > 0:
            self.vocabulary_diversity = len(set(response_tokens)) / total_tokens

            # Calculate repetition rate
            repeated_tokens = sum(1 for count in token_counts.values() if count > 1)
            self.repetition_rate = repeated_tokens / len(token_counts)

            # Estimate complexity (average token length as proxy)
            avg_token_length = sum(len(token) for token in response_tokens) / total_tokens
            self.complexity_score = min(1.0, avg_token_length / 8.0)  # Normalize to 0-1


@dataclass
class TopKConfiguration:
    """Configuration for Top K sampling for specific task and domain."""
    task_type: TaskType
    domain: Domain
    optimal_top_k: Optional[int]
    top_k_range: Tuple[Optional[int], Optional[int]]
    vocabulary_profile: VocabularyProfile
    description: str
    benefits: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)

    def get_effective_top_k(self, creativity_adjustment: float = 0.0) -> Optional[int]:
        """Get effective Top K value with creativity adjustment."""
        if self.optimal_top_k is None:
            return None

        # Creativity adjustment affects vocabulary size
        adjustment = int(self.optimal_top_k * creativity_adjustment * 0.5)
        adjusted_k = self.optimal_top_k + adjustment

        # Ensure within valid range
        min_k, max_k = self.top_k_range
        if min_k is not None:
            adjusted_k = max(min_k, adjusted_k)
        if max_k is not None:
            adjusted_k = min(max_k, adjusted_k)

        return adjusted_k if adjusted_k > 0 else None


@dataclass
class SamplingParameters:
    """Complete sampling parameters including Top K integration."""
    top_k: Optional[int] = None
    top_p: float = 0.9
    temperature: float = 0.7
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def validate(self) -> bool:
        """Validate parameter ranges."""
        if self.top_k is not None and self.top_k < 1:
            return False
        if not (0.0 <= self.top_p <= 1.0):
            return False
        if not (0.0 <= self.temperature <= 2.0):
            return False
        return True

    def get_vocabulary_constraint_level(self) -> str:
        """Get vocabulary constraint level description."""
        if self.top_k is None:
            return "Unlimited"
        elif self.top_k <= 20:
            return "Ultra-Focused"
        elif self.top_k <= 50:
            return "Focused"
        elif self.top_k <= 150:
            return "Balanced"
        elif self.top_k <= 300:
            return "Diverse"
        else:
            return "Very Diverse"


class VocabularyAnalyzer:
    """Analyzes vocabulary patterns and provides optimization recommendations."""

    def __init__(self):
        self.domain_vocabularies = self._initialize_domain_vocabularies()
        self.task_vocabulary_requirements = self._initialize_task_requirements()

    def _initialize_domain_vocabularies(self) -> Dict[Domain, Dict[str, Any]]:
        """Initialize domain-specific vocabulary characteristics."""
        return {
            Domain.TECHNICAL: {
                "preferred_complexity": "high",
                "jargon_tolerance": "high",
                "precision_requirement": "critical",
                "vocabulary_size": "focused",
                "key_indicators": ["implementation", "architecture", "algorithm", "optimization"]
            },
            Domain.MEDICAL: {
                "preferred_complexity": "very_high",
                "jargon_tolerance": "critical",
                "precision_requirement": "critical",
                "vocabulary_size": "ultra_focused",
                "key_indicators": ["diagnosis", "treatment", "symptoms", "pathology"]
            },
            Domain.LEGAL: {
                "preferred_complexity": "high",
                "jargon_tolerance": "high",
                "precision_requirement": "critical",
                "vocabulary_size": "focused",
                "key_indicators": ["contract", "liability", "statute", "precedent"]
            },
            Domain.BUSINESS: {
                "preferred_complexity": "medium",
                "jargon_tolerance": "medium",
                "precision_requirement": "high",
                "vocabulary_size": "balanced",
                "key_indicators": ["revenue", "strategy", "stakeholder", "metrics"]
            },
            Domain.CREATIVE: {
                "preferred_complexity": "variable",
                "jargon_tolerance": "low",
                "precision_requirement": "low",
                "vocabulary_size": "unlimited",
                "key_indicators": ["narrative", "character", "metaphor", "imagery"]
            },
            Domain.ACADEMIC: {
                "preferred_complexity": "high",
                "jargon_tolerance": "high",
                "precision_requirement": "high",
                "vocabulary_size": "diverse",
                "key_indicators": ["research", "methodology", "analysis", "theoretical"]
            }
        }

    def _initialize_task_requirements(self) -> Dict[TaskType, Dict[str, Any]]:
        """Initialize task-specific vocabulary requirements."""
        return {
            TaskType.CODE_GENERATION: {
                "vocabulary_precision": "critical",
                "allowed_deviation": "minimal",
                "domain_binding": "strict",
                "creativity_tolerance": "low"
            },
            TaskType.CREATIVE_WRITING: {
                "vocabulary_precision": "low",
                "allowed_deviation": "high",
                "domain_binding": "loose",
                "creativity_tolerance": "high"
            },
            TaskType.ANALYSIS: {
                "vocabulary_precision": "high",
                "allowed_deviation": "low",
                "domain_binding": "moderate",
                "creativity_tolerance": "low"
            },
            TaskType.CONVERSATION: {
                "vocabulary_precision": "medium",
                "allowed_deviation": "medium",
                "domain_binding": "loose",
                "creativity_tolerance": "medium"
            }
        }

    def analyze_domain_fit(self, text: str, expected_domain: Domain) -> float:
        """Analyze how well text fits expected domain vocabulary."""
        domain_info = self.domain_vocabularies.get(expected_domain, {})
        key_indicators = domain_info.get("key_indicators", [])

        if not key_indicators:
            return 0.5  # Neutral score for unknown domain

        text_lower = text.lower()
        indicator_matches = sum(1 for indicator in key_indicators if indicator in text_lower)

        return min(1.0, indicator_matches / len(key_indicators))

    def recommend_top_k(self, text_sample: str, task_type: TaskType, domain: Domain) -> Optional[int]:
        """Recommend Top K value based on text analysis."""
        task_reqs = self.task_vocabulary_requirements.get(task_type, {})
        domain_info = self.domain_vocabularies.get(domain, {})

        # Base recommendation on precision requirements
        precision_level = task_reqs.get("vocabulary_precision", "medium")
        creativity_tolerance = task_reqs.get("creativity_tolerance", "medium")

        # Analyze text complexity
        words = text_sample.split()
        unique_words = len(set(words))
        total_words = len(words)
        complexity_ratio = unique_words / max(total_words, 1)

        # Calculate base Top K
        if precision_level == "critical":
            base_k = 30
        elif precision_level == "high":
            base_k = 60
        elif precision_level == "medium":
            base_k = 120
        else:
            base_k = 200

        # Adjust for creativity tolerance
        if creativity_tolerance == "high":
            base_k = int(base_k * 1.5)
        elif creativity_tolerance == "low":
            base_k = int(base_k * 0.7)

        # Adjust for domain vocabulary size
        vocab_size = domain_info.get("vocabulary_size", "balanced")
        if vocab_size == "ultra_focused":
            base_k = min(base_k, 25)
        elif vocab_size == "focused":
            base_k = min(base_k, 50)
        elif vocab_size == "unlimited":
            return None  # No Top K constraint

        # Adjust for observed text complexity
        if complexity_ratio > 0.8:  # High diversity text
            base_k = int(base_k * 1.2)
        elif complexity_ratio < 0.4:  # Low diversity text
            base_k = int(base_k * 0.8)

        return max(10, base_k)  # Minimum of 10


class TopKManager:
    """
    Main class for managing Top K parameter control and vocabulary optimization.

    Provides intelligent Top K defaults for different tasks and domains,
    vocabulary analysis, and adaptive parameter adjustment.
    """

    def __init__(self):
        self.configurations = self._initialize_configurations()
        self.vocabulary_profiles = self._initialize_vocabulary_profiles()
        self.vocabulary_analyzer = VocabularyAnalyzer()
        self.performance_history = defaultdict(list)

    def _initialize_configurations(self) -> Dict[Tuple[TaskType, Domain], TopKConfiguration]:
        """Initialize task and domain-specific Top K configurations."""
        configs = {}

        # Analysis tasks - focused vocabulary for precision
        configs[(TaskType.ANALYSIS, Domain.GENERAL)] = TopKConfiguration(
            task_type=TaskType.ANALYSIS,
            domain=Domain.GENERAL,
            optimal_top_k=40,
            top_k_range=(25, 60),
            vocabulary_profile=VocabularyProfile.FOCUSED,
            description="General analysis requiring focused vocabulary for clear insights",
            benefits=["Reduces tangential language", "Improves clarity", "Maintains analytical tone"],
            use_cases=["Data analysis", "Research synthesis", "Performance evaluation"]
        )

        configs[(TaskType.ANALYSIS, Domain.TECHNICAL)] = TopKConfiguration(
            task_type=TaskType.ANALYSIS,
            domain=Domain.TECHNICAL,
            optimal_top_k=35,
            top_k_range=(20, 50),
            vocabulary_profile=VocabularyProfile.FOCUSED,
            description="Technical analysis requiring precise terminology",
            benefits=["Prevents non-technical terminology", "Ensures precision", "Maintains technical accuracy"],
            use_cases=["System analysis", "Code review", "Architecture evaluation"]
        )

        # Creative tasks - unlimited vocabulary for maximum expression
        configs[(TaskType.CREATIVE_WRITING, Domain.CREATIVE)] = TopKConfiguration(
            task_type=TaskType.CREATIVE_WRITING,
            domain=Domain.CREATIVE,
            optimal_top_k=None,  # Unlimited
            top_k_range=(None, None),
            vocabulary_profile=VocabularyProfile.UNLIMITED,
            description="Creative writing benefiting from unlimited vocabulary access",
            benefits=["Maximum word choice variety", "Enables unique expressions", "Supports creative metaphors"],
            use_cases=["Story writing", "Poetry", "Creative content", "Artistic expression"]
        )

        # Conversation - balanced vocabulary for natural interaction
        configs[(TaskType.CONVERSATION, Domain.GENERAL)] = TopKConfiguration(
            task_type=TaskType.CONVERSATION,
            domain=Domain.GENERAL,
            optimal_top_k=100,
            top_k_range=(70, 150),
            vocabulary_profile=VocabularyProfile.BALANCED,
            description="General conversation with natural vocabulary range",
            benefits=["Natural language flow", "Appropriate formality", "Conversational variety"],
            use_cases=["Chat responses", "Customer service", "Personal assistance"]
        )

        # Code generation - ultra-focused for syntax correctness
        configs[(TaskType.CODE_GENERATION, Domain.TECHNICAL)] = TopKConfiguration(
            task_type=TaskType.CODE_GENERATION,
            domain=Domain.TECHNICAL,
            optimal_top_k=25,
            top_k_range=(15, 40),
            vocabulary_profile=VocabularyProfile.ULTRA_FOCUSED,
            description="Code generation requiring precise technical vocabulary",
            benefits=["Reduces syntax errors", "Ensures valid keywords", "Maintains code conventions"],
            use_cases=["Function generation", "Code completion", "API implementation"]
        )

        # Factual extraction - focused for accuracy
        configs[(TaskType.FACTUAL_EXTRACTION, Domain.GENERAL)] = TopKConfiguration(
            task_type=TaskType.FACTUAL_EXTRACTION,
            domain=Domain.GENERAL,
            optimal_top_k=30,
            top_k_range=(20, 45),
            vocabulary_profile=VocabularyProfile.FOCUSED,
            description="Factual extraction requiring precise, unambiguous language",
            benefits=["Prevents interpretive language", "Ensures factual precision", "Reduces ambiguity"],
            use_cases=["Information extraction", "Data parsing", "Fact verification"]
        )

        # Business communication - balanced with professional tone
        configs[(TaskType.BUSINESS_COMMUNICATION, Domain.BUSINESS)] = TopKConfiguration(
            task_type=TaskType.BUSINESS_COMMUNICATION,
            domain=Domain.BUSINESS,
            optimal_top_k=80,
            top_k_range=(60, 120),
            vocabulary_profile=VocabularyProfile.BALANCED,
            description="Business communication with professional vocabulary control",
            benefits=["Maintains professional tone", "Ensures business appropriateness", "Balances formality"],
            use_cases=["Email composition", "Report writing", "Presentation content"]
        )

        # Academic writing - diverse vocabulary for scholarly work
        configs[(TaskType.ACADEMIC_WRITING, Domain.ACADEMIC)] = TopKConfiguration(
            task_type=TaskType.ACADEMIC_WRITING,
            domain=Domain.ACADEMIC,
            optimal_top_k=200,
            top_k_range=(150, 300),
            vocabulary_profile=VocabularyProfile.DIVERSE,
            description="Academic writing requiring diverse scholarly vocabulary",
            benefits=["Enables complex terminology", "Supports nuanced expression", "Maintains academic tone"],
            use_cases=["Research papers", "Academic analysis", "Scholarly communication"]
        )

        # Brainstorming - diverse vocabulary for idea generation
        configs[(TaskType.BRAINSTORMING, Domain.GENERAL)] = TopKConfiguration(
            task_type=TaskType.BRAINSTORMING,
            domain=Domain.GENERAL,
            optimal_top_k=250,
            top_k_range=(200, None),
            vocabulary_profile=VocabularyProfile.DIVERSE,
            description="Brainstorming benefiting from diverse vocabulary for idea variety",
            benefits=["Encourages word variety", "Supports creative connections", "Enables unexpected associations"],
            use_cases=["Idea generation", "Problem solving", "Creative exploration"]
        )

        return configs

    def _initialize_vocabulary_profiles(self) -> Dict[VocabularyProfile, Dict[str, Any]]:
        """Initialize vocabulary profiles with their characteristics."""
        return {
            VocabularyProfile.ULTRA_FOCUSED: {
                "top_k_range": (10, 30),
                "description": "Extremely limited vocabulary for maximum precision",
                "use_cases": ["Code generation", "Technical specifications", "Medical diagnoses"],
                "trade_offs": {"precision": "maximum", "creativity": "minimal", "variety": "limited"}
            },
            VocabularyProfile.FOCUSED: {
                "top_k_range": (25, 60),
                "description": "Limited vocabulary for high precision with moderate variety",
                "use_cases": ["Analysis", "Factual extraction", "Technical documentation"],
                "trade_offs": {"precision": "high", "creativity": "low", "variety": "moderate"}
            },
            VocabularyProfile.BALANCED: {
                "top_k_range": (70, 150),
                "description": "Balanced vocabulary for natural communication",
                "use_cases": ["Conversation", "General writing", "Business communication"],
                "trade_offs": {"precision": "moderate", "creativity": "moderate", "variety": "good"}
            },
            VocabularyProfile.DIVERSE: {
                "top_k_range": (150, 300),
                "description": "Diverse vocabulary for expressive communication",
                "use_cases": ["Academic writing", "Brainstorming", "Complex analysis"],
                "trade_offs": {"precision": "moderate", "creativity": "high", "variety": "extensive"}
            },
            VocabularyProfile.UNLIMITED: {
                "top_k_range": (None, None),
                "description": "No vocabulary constraints for maximum creativity",
                "use_cases": ["Creative writing", "Artistic expression", "Experimental content"],
                "trade_offs": {"precision": "variable", "creativity": "maximum", "variety": "unlimited"}
            }
        }

    def get_optimal_top_k(self,
                         task_type: TaskType,
                         domain: Domain = Domain.GENERAL,
                         creativity_boost: float = 0.0,
                         precision_boost: float = 0.0) -> Optional[int]:
        """
        Get optimal Top K value for specific task and domain.

        Args:
            task_type: The type of task being performed
            domain: The domain context
            creativity_boost: Adjustment to increase vocabulary diversity (-0.5 to +0.5)
            precision_boost: Adjustment to increase vocabulary focus (-0.5 to +0.5)

        Returns:
            Optimal Top K value (None for unlimited)
        """
        # Try exact match first
        config_key = (task_type, domain)
        if config_key in self.configurations:
            config = self.configurations[config_key]
        else:
            # Fallback to general domain for the task
            general_key = (task_type, Domain.GENERAL)
            config = self.configurations.get(general_key)

            if not config:
                # Final fallback based on task type
                config = self._get_fallback_config(task_type)

        if config.optimal_top_k is None:
            return None

        # Apply adjustments
        net_adjustment = creativity_boost - precision_boost
        adjusted_k = config.get_effective_top_k(net_adjustment)

        return adjusted_k

    def _get_fallback_config(self, task_type: TaskType) -> TopKConfiguration:
        """Get fallback configuration for unknown task-domain combinations."""
        fallback_top_k = {
            TaskType.CODE_GENERATION: 25,
            TaskType.FACTUAL_EXTRACTION: 30,
            TaskType.ANALYSIS: 40,
            TaskType.TECHNICAL_DOCUMENTATION: 50,
            TaskType.BUSINESS_COMMUNICATION: 80,
            TaskType.CONVERSATION: 100,
            TaskType.SUMMARIZATION: 60,
            TaskType.QUESTION_ANSWERING: 70,
            TaskType.REASONING: 50,
            TaskType.TRANSLATION: 150,
            TaskType.ACADEMIC_WRITING: 200,
            TaskType.BRAINSTORMING: 250,
            TaskType.CREATIVE_WRITING: None,
            TaskType.STORYTELLING: None
        }

        top_k = fallback_top_k.get(task_type, 100)

        return TopKConfiguration(
            task_type=task_type,
            domain=Domain.GENERAL,
            optimal_top_k=top_k,
            top_k_range=(max(10, int(top_k * 0.7)) if top_k else None,
                        int(top_k * 1.5) if top_k else None),
            vocabulary_profile=VocabularyProfile.BALANCED,
            description=f"Fallback configuration for {task_type.value}",
            benefits=["General vocabulary control"],
            use_cases=[task_type.value]
        )

    def get_complete_sampling_parameters(self,
                                       task_type: TaskType,
                                       domain: Domain = Domain.GENERAL,
                                       top_p: float = None,
                                       temperature: float = None,
                                       creativity_adjustment: float = 0.0) -> SamplingParameters:
        """
        Get complete sampling parameters with optimized Top K integration.

        Args:
            task_type: Task type for optimization
            domain: Domain context
            top_p: Override Top P (uses intelligent default if None)
            temperature: Override temperature (uses intelligent default if None)
            creativity_adjustment: Overall creativity adjustment

        Returns:
            Complete optimized sampling parameters
        """
        # Get optimal Top K
        top_k = self.get_optimal_top_k(task_type, domain, creativity_adjustment)

        # Set intelligent defaults for Top P and temperature based on Top K
        if top_p is None:
            if top_k is None:  # Unlimited vocabulary
                top_p = 0.95  # Higher Top P for creativity
            elif top_k <= 30:  # Very focused vocabulary
                top_p = 0.85  # Lower Top P to avoid conflict
            elif top_k <= 100:  # Moderate vocabulary
                top_p = 0.9   # Balanced Top P
            else:  # Diverse vocabulary
                top_p = 0.92  # Slightly higher Top P

        if temperature is None:
            if top_k is None:  # Unlimited vocabulary
                temperature = 0.8  # Higher temperature for creativity
            elif top_k <= 30:  # Very focused vocabulary
                temperature = 0.3  # Lower temperature for consistency
            elif top_k <= 100:  # Moderate vocabulary
                temperature = 0.6  # Balanced temperature
            else:  # Diverse vocabulary
                temperature = 0.7  # Moderate temperature

        # Apply creativity adjustment to temperature
        temperature += creativity_adjustment * 0.2
        temperature = max(0.1, min(1.5, temperature))

        return SamplingParameters(
            top_k=top_k,
            top_p=top_p,
            temperature=temperature
        )

    def analyze_vocabulary_effectiveness(self,
                                       response_text: str,
                                       expected_domain: Domain,
                                       task_type: TaskType,
                                       used_top_k: Optional[int]) -> Dict[str, Any]:
        """
        Analyze vocabulary effectiveness in a response.

        Args:
            response_text: The generated response text
            expected_domain: Expected domain context
            task_type: Task type that was performed
            used_top_k: Top K value that was used

        Returns:
            Comprehensive vocabulary analysis
        """
        analysis = {
            "vocabulary_metrics": {},
            "domain_alignment": {},
            "optimization_suggestions": [],
            "quality_assessment": {}
        }

        # Basic vocabulary metrics
        tokens = response_text.split()
        unique_tokens = set(tokens)

        metrics = VocabularyMetrics()
        metrics.update_metrics(tokens, expected_domain)

        analysis["vocabulary_metrics"] = {
            "total_tokens": len(tokens),
            "unique_tokens": len(unique_tokens),
            "vocabulary_diversity": metrics.vocabulary_diversity,
            "repetition_rate": metrics.repetition_rate,
            "complexity_score": metrics.complexity_score
        }

        # Domain alignment analysis
        domain_fit = self.vocabulary_analyzer.analyze_domain_fit(response_text, expected_domain)
        analysis["domain_alignment"] = {
            "domain_fit_score": domain_fit,
            "expected_domain": expected_domain.value,
            "alignment_quality": "High" if domain_fit > 0.7 else "Medium" if domain_fit > 0.4 else "Low"
        }

        # Generate optimization suggestions
        recommended_top_k = self.vocabulary_analyzer.recommend_top_k(response_text, task_type, expected_domain)

        if used_top_k and recommended_top_k:
            if abs(used_top_k - recommended_top_k) > 20:
                analysis["optimization_suggestions"].append(
                    f"Consider adjusting Top K from {used_top_k} to {recommended_top_k} for better vocabulary control"
                )

        if metrics.repetition_rate > 0.3:
            analysis["optimization_suggestions"].append(
                "High repetition detected - consider increasing Top K or adding repetition penalties"
            )

        if metrics.vocabulary_diversity < 0.3:
            analysis["optimization_suggestions"].append(
                "Low vocabulary diversity - consider increasing Top K for more varied expression"
            )
        elif metrics.vocabulary_diversity > 0.8 and task_type in [TaskType.CODE_GENERATION, TaskType.FACTUAL_EXTRACTION]:
            analysis["optimization_suggestions"].append(
                "High vocabulary diversity for precision task - consider reducing Top K for more focused output"
            )

        # Quality assessment
        task_requirements = self.vocabulary_analyzer.task_vocabulary_requirements.get(task_type, {})
        precision_requirement = task_requirements.get("vocabulary_precision", "medium")

        quality_score = 0.5
        if precision_requirement == "critical":
            quality_score = 1.0 - metrics.vocabulary_diversity * 0.5  # Lower diversity is better
        elif precision_requirement == "low":
            quality_score = metrics.vocabulary_diversity  # Higher diversity is better
        else:
            quality_score = 0.7 if 0.4 <= metrics.vocabulary_diversity <= 0.7 else 0.4

        analysis["quality_assessment"] = {
            "overall_score": quality_score,
            "precision_alignment": "Good" if quality_score > 0.6 else "Needs Improvement",
            "vocabulary_appropriateness": "Appropriate" if domain_fit > 0.5 else "Misaligned"
        }

        return analysis

    def compare_top_k_values(self, values: List[Optional[int]], task_type: TaskType, domain: Domain) -> Dict[str, Any]:
        """Compare different Top K values for the same task and domain."""
        comparison = {
            "value_analysis": {},
            "recommendations": {},
            "trade_offs": {}
        }

        config_key = (task_type, domain)
        optimal_config = self.configurations.get(config_key)

        for top_k in values:
            constraint_level = "Unlimited" if top_k is None else SamplingParameters(top_k=top_k).get_vocabulary_constraint_level()

            # Analyze expected effects
            expected_effects = []
            if top_k is None:
                expected_effects = ["Maximum vocabulary access", "High creativity potential", "Variable precision"]
            elif top_k <= 30:
                expected_effects = ["High precision", "Limited creativity", "Consistent terminology"]
            elif top_k <= 100:
                expected_effects = ["Balanced precision", "Moderate creativity", "Natural variety"]
            else:
                expected_effects = ["High variety", "Creative flexibility", "Potential precision loss"]

            # Calculate appropriateness score
            if optimal_config:
                if top_k == optimal_config.optimal_top_k:
                    appropriateness = 1.0
                elif optimal_config.optimal_top_k is None:
                    appropriateness = 0.3 if top_k and top_k < 100 else 0.8
                else:
                    diff = abs(top_k - optimal_config.optimal_top_k) if top_k else 100
                    appropriateness = max(0.1, 1.0 - (diff / 100))
            else:
                appropriateness = 0.5

            comparison["value_analysis"][f"top_k_{top_k}"] = {
                "constraint_level": constraint_level,
                "expected_effects": expected_effects,
                "task_appropriateness": appropriateness,
                "domain_fit": appropriateness  # Simplified for demo
            }

        # Generate recommendations
        best_value = max(values, key=lambda x: comparison["value_analysis"][f"top_k_{x}"]["task_appropriateness"])
        comparison["recommendations"]["best_for_task"] = f"top_k_{best_value}"

        if optimal_config:
            comparison["recommendations"]["optimal_value"] = optimal_config.optimal_top_k
            comparison["recommendations"]["reasoning"] = optimal_config.description

        return comparison

    def get_domain_specific_recommendations(self, domain: Domain) -> Dict[str, Any]:
        """Get domain-specific Top K recommendations and guidelines."""
        domain_info = self.vocabulary_analyzer.domain_vocabularies.get(domain, {})

        recommendations = {
            "domain_characteristics": domain_info,
            "suggested_ranges": {},
            "task_recommendations": {},
            "best_practices": []
        }

        # Generate task-specific recommendations for this domain
        domain_configs = {k: v for k, v in self.configurations.items() if k[1] == domain}

        for (task_type, _), config in domain_configs.items():
            recommendations["task_recommendations"][task_type.value] = {
                "optimal_top_k": config.optimal_top_k,
                "range": config.top_k_range,
                "benefits": config.benefits,
                "use_cases": config.use_cases
            }

        # Domain-specific best practices
        if domain == Domain.TECHNICAL:
            recommendations["best_practices"] = [
                "Use lower Top K values (20-50) for code generation",
                "Prioritize precision over creativity for technical documentation",
                "Consider domain-specific vocabulary when setting constraints"
            ]
        elif domain == Domain.CREATIVE:
            recommendations["best_practices"] = [
                "Use unlimited Top K for maximum creative expression",
                "Allow high vocabulary diversity for unique voice",
                "Balance creativity with coherence using Top P and temperature"
            ]
        elif domain == Domain.BUSINESS:
            recommendations["best_practices"] = [
                "Use moderate Top K (60-120) for professional tone",
                "Maintain consistency in business communication",
                "Consider audience and formality requirements"
            ]

        return recommendations

    def export_configuration(self, format: str = "json") -> str:
        """Export Top K configurations and analysis."""
        if format == "json":
            export_data = {
                "configurations": {
                    f"{task.value}_{domain.value}": {
                        "optimal_top_k": config.optimal_top_k,
                        "top_k_range": config.top_k_range,
                        "vocabulary_profile": config.vocabulary_profile.value,
                        "description": config.description,
                        "benefits": config.benefits,
                        "use_cases": config.use_cases
                    }
                    for (task, domain), config in self.configurations.items()
                },
                "vocabulary_profiles": {
                    profile.value: {
                        "top_k_range": info["top_k_range"],
                        "description": info["description"],
                        "trade_offs": info["trade_offs"]
                    }
                    for profile, info in self.vocabulary_profiles.items()
                }
            }
            return json.dumps(export_data, indent=2)

        else:
            lines = [
                "VaultMind Top K Configuration Guide",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Task-Domain Configurations:",
                ""
            ]

            for (task_type, domain), config in sorted(self.configurations.items()):
                lines.extend([
                    f"**{task_type.value.title()} - {domain.value.title()}**:",
                    f"  Optimal Top K: {config.optimal_top_k or 'Unlimited'}",
                    f"  Range: {config.top_k_range}",
                    f"  Profile: {config.vocabulary_profile.value}",
                    f"  Description: {config.description}",
                    ""
                ])

            return "\n".join(lines)


# Convenience functions for common Top K scenarios
def get_focused_parameters(task_type: TaskType, domain: Domain = Domain.GENERAL) -> SamplingParameters:
    """Get parameters optimized for focused vocabulary control."""
    manager = TopKManager()
    return manager.get_complete_sampling_parameters(
        task_type=task_type,
        domain=domain,
        creativity_adjustment=-0.2  # Reduce creativity for focus
    )


def get_creative_parameters(task_type: TaskType, domain: Domain = Domain.GENERAL) -> SamplingParameters:
    """Get parameters optimized for creative vocabulary diversity."""
    manager = TopKManager()
    return manager.get_complete_sampling_parameters(
        task_type=task_type,
        domain=domain,
        creativity_adjustment=0.3  # Increase creativity
    )


def analyze_top_k_effectiveness(response_text: str,
                              task_type: TaskType,
                              domain: Domain,
                              used_top_k: Optional[int]) -> Dict[str, Any]:
    """Quick function to analyze Top K effectiveness in a response."""
    manager = TopKManager()
    return manager.analyze_vocabulary_effectiveness(response_text, domain, task_type, used_top_k)


def recommend_top_k_for_content(content_sample: str,
                               task_type: TaskType,
                               domain: Domain = Domain.GENERAL) -> Dict[str, Any]:
    """Recommend optimal Top K value based on content analysis."""
    manager = TopKManager()
    analyzer = VocabularyAnalyzer()

    recommended_k = analyzer.recommend_top_k(content_sample, task_type, domain)
    optimal_k = manager.get_optimal_top_k(task_type, domain)

    return {
        "content_based_recommendation": recommended_k,
        "optimal_default": optimal_k,
        "reasoning": f"Based on {task_type.value} task in {domain.value} domain",
        "comparison": {
            "content_driven": recommended_k,
            "task_optimized": optimal_k,
            "difference": abs((recommended_k or 0) - (optimal_k or 0)) if recommended_k and optimal_k else "N/A"
        }
    }


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Top K Control System Demo ===\n")

    # Initialize Top K manager
    manager = TopKManager()

    print("1. Configuration Overview:")
    print(f"   Total configurations: {len(manager.configurations)}")

    # Show key configurations
    key_configs = [
        (TaskType.ANALYSIS, Domain.GENERAL),
        (TaskType.CREATIVE_WRITING, Domain.CREATIVE),
        (TaskType.CODE_GENERATION, Domain.TECHNICAL),
        (TaskType.CONVERSATION, Domain.GENERAL)
    ]

    for task, domain in key_configs:
        config = manager.configurations.get((task, domain))
        if config:
            top_k_display = config.optimal_top_k or "Unlimited"
            print(f"   {task.value} ({domain.value}): Top K = {top_k_display}")
    print()

    print("2. Vocabulary Profile Comparison:")
    profiles_to_show = [VocabularyProfile.ULTRA_FOCUSED, VocabularyProfile.BALANCED, VocabularyProfile.UNLIMITED]

    for profile in profiles_to_show:
        info = manager.vocabulary_profiles[profile]
        range_display = f"{info['top_k_range'][0]}-{info['top_k_range'][1]}" if info['top_k_range'][0] else "No limit"
        print(f"   {profile.value}: {range_display} tokens")
        print(f"     Description: {info['description']}")
    print()

    print("3. Task-Specific Optimization Examples:")
    test_tasks = [
        (TaskType.CODE_GENERATION, Domain.TECHNICAL, "Maximum precision for code"),
        (TaskType.BRAINSTORMING, Domain.GENERAL, "Diverse vocabulary for ideas"),
        (TaskType.BUSINESS_COMMUNICATION, Domain.BUSINESS, "Professional balanced vocabulary")
    ]

    for task, domain, description in test_tasks:
        top_k = manager.get_optimal_top_k(task, domain)
        params = manager.get_complete_sampling_parameters(task, domain)

        print(f"   {task.value}:")
        print(f"     Top K: {top_k or 'Unlimited'}")
        print(f"     Integrated params: Top P = {params.top_p}, Temp = {params.temperature}")
        print(f"     Purpose: {description}")
    print()

    print("4. Vocabulary Analysis Demo:")
    sample_responses = [
        {
            "text": "The algorithm implementation requires careful optimization of the data structure to achieve optimal performance metrics.",
            "task": TaskType.CODE_GENERATION,
            "domain": Domain.TECHNICAL,
            "used_top_k": 25
        },
        {
            "text": "In the moonlit garden, whispers of ancient secrets danced among the silver leaves, painting dreams across the starlit canvas of night.",
            "task": TaskType.CREATIVE_WRITING,
            "domain": Domain.CREATIVE,
            "used_top_k": None
        }
    ]

    for sample in sample_responses:
        analysis = manager.analyze_vocabulary_effectiveness(
            sample["text"], sample["domain"], sample["task"], sample["used_top_k"]
        )

        print(f"   Sample: '{sample['text'][:50]}...'")
        print(f"     Vocabulary diversity: {analysis['vocabulary_metrics']['vocabulary_diversity']:.3f}")
        print(f"     Domain alignment: {analysis['domain_alignment']['alignment_quality']}")
        print(f"     Quality assessment: {analysis['quality_assessment']['precision_alignment']}")
        if analysis["optimization_suggestions"]:
            print(f"     Suggestion: {analysis['optimization_suggestions'][0]}")
    print()

    print("5. Top K Value Comparison:")
    comparison_values = [25, 50, 100, None]
    comparison = manager.compare_top_k_values(comparison_values, TaskType.ANALYSIS, Domain.GENERAL)

    print(f"   For {TaskType.ANALYSIS.value} in {Domain.GENERAL.value} domain:")
    for value_key, analysis in comparison["value_analysis"].items():
        value = value_key.split("_")[-1]
        if value == "None":
            value = "Unlimited"

        print(f"     Top K {value}: {analysis['constraint_level']} constraint")
        print(f"       Appropriateness: {analysis['task_appropriateness']:.2f}")
        print(f"       Effects: {', '.join(analysis['expected_effects'][:2])}")
    print()

    print("6. Domain-Specific Recommendations:")
    tech_recommendations = manager.get_domain_specific_recommendations(Domain.TECHNICAL)

    print(f"   {Domain.TECHNICAL.value.title()} Domain Guidelines:")
    print(f"     Precision requirement: {tech_recommendations['domain_characteristics']['precision_requirement']}")
    print(f"     Vocabulary size preference: {tech_recommendations['domain_characteristics']['vocabulary_size']}")

    if tech_recommendations["best_practices"]:
        print("     Best practices:")
        for practice in tech_recommendations["best_practices"][:2]:
            print(f"       • {practice}")
    print()

    print("7. Content-Based Recommendations:")
    test_content = "We need to analyze the quarterly sales performance metrics and identify key growth opportunities for the upcoming fiscal year."

    recommendation = recommend_top_k_for_content(test_content, TaskType.ANALYSIS, Domain.BUSINESS)

    print(f"   Content sample: '{test_content[:60]}...'")
    print(f"   Content-based recommendation: Top K = {recommendation['content_based_recommendation']}")
    print(f"   Task-optimized default: Top K = {recommendation['optimal_default']}")
    print(f"   Reasoning: {recommendation['reasoning']}")
    print()

    print("8. Focused vs Creative Parameter Comparison:")
    focused_params = get_focused_parameters(TaskType.CONVERSATION, Domain.GENERAL)
    creative_params = get_creative_parameters(TaskType.CONVERSATION, Domain.GENERAL)

    print(f"   Focused conversation: Top K = {focused_params.top_k}, Top P = {focused_params.top_p}, Temp = {focused_params.temperature}")
    print(f"   Creative conversation: Top K = {creative_params.top_k}, Top P = {creative_params.top_p}, Temp = {creative_params.temperature}")
    print(f"   Vocabulary constraint difference: {abs(focused_params.top_k - creative_params.top_k)} tokens")

    print("\n=== Top K Control System Ready ===")
    print("Key capabilities:")
    print("• Task and domain-specific Top K optimization")
    print("• Vocabulary profile management (ultra-focused to unlimited)")
    print("• Integrated parameter optimization with Top P and temperature")
    print("• Real-time vocabulary effectiveness analysis")
    print("• Content-based Top K recommendations")
    print("• Domain-specific vocabulary control strategies")
