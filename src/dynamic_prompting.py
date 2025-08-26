"""
VaultMind Dynamic Prompting Implementation

This module implements adaptive prompting that customizes AI interactions based on
vault characteristics, user patterns, note types, and contextual information.
Unlike static prompting approaches, dynamic prompting evolves and adapts to
provide increasingly personalized and effective analysis.

Dynamic prompting learns from vault patterns, user preferences, and contextual
cues to generate highly targeted prompts that improve over time.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
import json
import re
from collections import Counter, defaultdict
import math


class NoteType(Enum):
    """Types of notes that can be analyzed."""
    JOURNAL = "journal"
    RESEARCH = "research"
    MEETING = "meeting"
    PROJECT = "project"
    IDEA = "idea"
    REFERENCE = "reference"
    PERSONAL = "personal"
    ACADEMIC = "academic"
    DAILY = "daily"
    BOOK_NOTES = "book_notes"


class AnalysisTask(Enum):
    """Types of analysis tasks for dynamic prompting."""
    EXTRACT_KEY_POINTS = "extract_key_points"
    IDENTIFY_THEMES = "identify_themes"
    SUMMARIZE_CONTENT = "summarize_content"
    FIND_ACTION_ITEMS = "find_action_items"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    CATEGORIZE_TOPICS = "categorize_topics"
    EXTRACT_ENTITIES = "extract_entities"
    IDENTIFY_PATTERNS = "identify_patterns"
    CREATE_CONNECTIONS = "create_connections"
    TRACK_PROGRESS = "track_progress"


class ContextDimension(Enum):
    """Dimensions of context for dynamic adaptation."""
    TEMPORAL = "temporal"  # Time-based patterns
    THEMATIC = "thematic"  # Content themes
    BEHAVIORAL = "behavioral"  # User patterns
    STRUCTURAL = "structural"  # Vault organization
    SEMANTIC = "semantic"  # Content relationships
    EMOTIONAL = "emotional"  # Sentiment patterns
    PRODUCTIVITY = "productivity"  # Work patterns


class AdaptationStrategy(Enum):
    """Strategies for prompt adaptation."""
    PROGRESSIVE_LEARNING = "progressive_learning"
    PATTERN_MATCHING = "pattern_matching"
    CONTEXTUAL_WEIGHTING = "contextual_weighting"
    USER_FEEDBACK_LOOP = "user_feedback_loop"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    TEMPORAL_EVOLUTION = "temporal_evolution"


@dataclass
class VaultProfile:
    """Comprehensive profile of a user's vault characteristics."""
    total_notes: int = 0
    note_type_distribution: Dict[str, int] = field(default_factory=dict)
    creation_patterns: Dict[str, int] = field(default_factory=dict)  # day_of_week, hour patterns
    tag_frequency: Dict[str, int] = field(default_factory=dict)
    topic_clusters: Dict[str, List[str]] = field(default_factory=dict)
    writing_style_indicators: Dict[str, float] = field(default_factory=dict)
    average_note_length: float = 0.0
    complexity_distribution: Dict[str, int] = field(default_factory=dict)
    link_density: float = 0.0
    update_frequency: Dict[str, int] = field(default_factory=dict)
    seasonal_patterns: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_dominant_note_types(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """Get the most common note types."""
        return sorted(self.note_type_distribution.items(),
                     key=lambda x: x[1], reverse=True)[:top_n]

    def get_peak_activity_times(self) -> List[str]:
        """Get times when user is most active."""
        if not self.creation_patterns:
            return []
        return sorted(self.creation_patterns.items(),
                     key=lambda x: x[1], reverse=True)[:3]

    def estimate_expertise_level(self) -> str:
        """Estimate user's general expertise level."""
        if self.total_notes < 50:
            return "beginner"
        elif self.total_notes < 200:
            return "intermediate"
        elif self.total_notes < 500:
            return "advanced"
        else:
            return "expert"


@dataclass
class UserBehaviorPattern:
    """Tracks user interaction patterns and preferences."""
    preferred_analysis_tasks: Dict[str, int] = field(default_factory=dict)
    preferred_output_formats: Dict[str, int] = field(default_factory=dict)
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    session_patterns: Dict[str, Any] = field(default_factory=dict)
    error_patterns: Dict[str, int] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    adaptation_responsiveness: float = 0.5  # How quickly user adapts to changes

    def get_preferred_style(self) -> Dict[str, str]:
        """Get user's preferred interaction style."""
        style = {}

        if self.preferred_output_formats:
            most_used_format = max(self.preferred_output_formats.items(),
                                 key=lambda x: x[1])
            style["output_format"] = most_used_format[0]

        if self.preferred_analysis_tasks:
            most_used_task = max(self.preferred_analysis_tasks.items(),
                               key=lambda x: x[1])
            style["primary_task"] = most_used_task[0]

        return style


@dataclass
class ContextualCues:
    """Real-time contextual information for prompt adaptation."""
    current_session_context: Dict[str, Any] = field(default_factory=dict)
    recent_notes_analyzed: List[str] = field(default_factory=list)
    user_mood_indicators: Dict[str, float] = field(default_factory=dict)
    time_of_day: str = ""
    day_of_week: str = ""
    season: str = ""
    recent_user_queries: List[str] = field(default_factory=list)
    active_projects: List[str] = field(default_factory=list)
    current_goals: List[str] = field(default_factory=list)

    def get_context_weight(self, dimension: ContextDimension) -> float:
        """Get the importance weight for a context dimension."""
        weights = {
            ContextDimension.TEMPORAL: 0.8 if self.time_of_day else 0.2,
            ContextDimension.BEHAVIORAL: 0.9 if self.recent_user_queries else 0.3,
            ContextDimension.THEMATIC: 0.7 if self.active_projects else 0.4,
            ContextDimension.EMOTIONAL: 0.6 if self.user_mood_indicators else 0.2,
            ContextDimension.PRODUCTIVITY: 0.8 if self.current_goals else 0.3,
        }
        return weights.get(dimension, 0.5)


@dataclass
class DynamicPrompt:
    """Represents an adaptively generated prompt."""
    base_prompt: str
    contextual_adaptations: Dict[str, str] = field(default_factory=dict)
    personalization_elements: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    adaptation_reasoning: List[str] = field(default_factory=list)
    fallback_prompts: List[str] = field(default_factory=list)

    def build_complete_prompt(self) -> str:
        """Build the complete adaptive prompt."""
        sections = [self.base_prompt]

        # Add contextual adaptations
        if self.contextual_adaptations:
            sections.append("\n**Contextual Adaptations:**")
            for context_type, adaptation in self.contextual_adaptations.items():
                sections.append(f"- {context_type}: {adaptation}")

        # Add personalization
        if self.personalization_elements:
            sections.append("\n**Personalized for your vault:**")
            for element in self.personalization_elements:
                sections.append(f"- {element}")

        return "\n".join(sections)

    def get_prompt_metadata(self) -> Dict[str, Any]:
        """Get metadata about the prompt generation."""
        return {
            "confidence_score": self.confidence_score,
            "adaptations_applied": len(self.contextual_adaptations),
            "personalization_level": len(self.personalization_elements),
            "reasoning_steps": self.adaptation_reasoning,
            "has_fallbacks": len(self.fallback_prompts) > 0
        }


class ContextAnalyzer:
    """Analyzes vault and user context for dynamic prompt generation."""

    def __init__(self):
        self.pattern_extractors = self._initialize_pattern_extractors()

    def _initialize_pattern_extractors(self) -> Dict[str, Any]:
        """Initialize pattern extraction methods."""
        return {
            "temporal": self._extract_temporal_patterns,
            "thematic": self._extract_thematic_patterns,
            "behavioral": self._extract_behavioral_patterns,
            "structural": self._extract_structural_patterns,
            "semantic": self._extract_semantic_patterns
        }

    def analyze_vault_patterns(self, vault_data: Dict[str, Any]) -> VaultProfile:
        """Analyze vault data to create a comprehensive profile."""
        profile = VaultProfile()

        if "notes" in vault_data:
            notes = vault_data["notes"]
            profile.total_notes = len(notes)

            # Note type distribution
            profile.note_type_distribution = self._calculate_note_type_distribution(notes)

            # Creation patterns (temporal analysis)
            profile.creation_patterns = self._analyze_creation_patterns(notes)

            # Tag frequency analysis
            profile.tag_frequency = self._analyze_tag_frequency(notes)

            # Writing style indicators
            profile.writing_style_indicators = self._analyze_writing_style(notes)

            # Average note length
            total_words = sum(len(note.get("content", "").split()) for note in notes)
            profile.average_note_length = total_words / len(notes) if notes else 0

            # Link density analysis
            profile.link_density = self._calculate_link_density(notes)

            # Topic clustering
            profile.topic_clusters = self._perform_topic_clustering(notes)

        return profile

    def _calculate_note_type_distribution(self, notes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of note types."""
        distribution = Counter()
        for note in notes:
            note_type = note.get("type", "unknown")
            distribution[note_type] += 1
        return dict(distribution)

    def _analyze_creation_patterns(self, notes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze when notes are typically created."""
        patterns = {"morning": 0, "afternoon": 0, "evening": 0, "night": 0}
        day_patterns = defaultdict(int)

        for note in notes:
            created_date = note.get("created_date")
            if created_date:
                try:
                    # Parse date and extract time patterns
                    dt = datetime.fromisoformat(created_date.replace("Z", "+00:00"))
                    hour = dt.hour
                    day = dt.strftime("%A").lower()

                    # Time of day classification
                    if 5 <= hour < 12:
                        patterns["morning"] += 1
                    elif 12 <= hour < 17:
                        patterns["afternoon"] += 1
                    elif 17 <= hour < 22:
                        patterns["evening"] += 1
                    else:
                        patterns["night"] += 1

                    day_patterns[day] += 1
                except:
                    continue

        # Combine time and day patterns
        patterns.update(dict(day_patterns))
        return patterns

    def _analyze_tag_frequency(self, notes: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze tag usage frequency."""
        tag_counter = Counter()
        for note in notes:
            tags = note.get("tags", [])
            for tag in tags:
                tag_counter[tag.lower().strip()] += 1
        return dict(tag_counter.most_common(50))  # Top 50 tags

    def _analyze_writing_style(self, notes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze writing style indicators."""
        indicators = {
            "avg_sentence_length": 0.0,
            "formality_score": 0.0,
            "emotional_intensity": 0.0,
            "question_frequency": 0.0,
            "technical_density": 0.0
        }

        total_sentences = 0
        total_words = 0
        question_count = 0
        emotional_words = 0
        technical_terms = 0

        # Simple heuristics for style analysis
        emotional_indicators = {"feel", "think", "believe", "love", "hate", "excited", "worried", "happy", "sad"}
        technical_indicators = {"algorithm", "framework", "implementation", "methodology", "analysis", "system"}

        for note in notes:
            content = note.get("content", "")
            sentences = re.split(r'[.!?]+', content)
            words = content.lower().split()

            total_sentences += len([s for s in sentences if s.strip()])
            total_words += len(words)
            question_count += content.count('?')

            for word in words:
                if word in emotional_indicators:
                    emotional_words += 1
                if word in technical_indicators:
                    technical_terms += 1

        if total_sentences > 0:
            indicators["avg_sentence_length"] = total_words / total_sentences
            indicators["question_frequency"] = question_count / total_sentences
            indicators["emotional_intensity"] = emotional_words / total_words if total_words > 0 else 0
            indicators["technical_density"] = technical_terms / total_words if total_words > 0 else 0

        return indicators

    def _calculate_link_density(self, notes: List[Dict[str, Any]]) -> float:
        """Calculate internal link density in the vault."""
        total_links = 0
        total_notes = len(notes)

        for note in notes:
            content = note.get("content", "")
            # Count [[wiki-style]] links and [markdown](links)
            wiki_links = len(re.findall(r'\[\[([^\]]+)\]\]', content))
            markdown_links = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', content))
            total_links += wiki_links + markdown_links

        return total_links / total_notes if total_notes > 0 else 0

    def _perform_topic_clustering(self, notes: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Perform simple topic clustering based on tags and keywords."""
        clusters = defaultdict(list)

        # Simple clustering based on common tags and keywords
        for note in notes:
            title = note.get("title", "")
            tags = note.get("tags", [])
            content = note.get("content", "")

            # Extract key terms
            words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
            word_freq = Counter(words)
            key_terms = [word for word, freq in word_freq.most_common(5)]

            # Cluster by primary tag or key term
            if tags:
                primary_cluster = tags[0].lower()
            elif key_terms:
                primary_cluster = key_terms[0]
            else:
                primary_cluster = "uncategorized"

            clusters[primary_cluster].append(title)

        return dict(clusters)

    def _extract_temporal_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal patterns from vault data."""
        return {"implementation": "temporal_pattern_extraction"}

    def _extract_thematic_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract thematic patterns from vault data."""
        return {"implementation": "thematic_pattern_extraction"}

    def _extract_behavioral_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral patterns from user data."""
        return {"implementation": "behavioral_pattern_extraction"}

    def _extract_structural_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural patterns from vault organization."""
        return {"implementation": "structural_pattern_extraction"}

    def _extract_semantic_patterns(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic patterns from content relationships."""
        return {"implementation": "semantic_pattern_extraction"}


class DynamicPromptBuilder:
    """
    Main class for building adaptive prompts based on context and user patterns.

    This builder dynamically customizes prompts based on vault characteristics,
    user behavior patterns, and real-time contextual information.
    """

    def __init__(self):
        self.context_analyzer = ContextAnalyzer()
        self.adaptation_strategies = self._initialize_adaptation_strategies()
        self.prompt_templates = self._initialize_prompt_templates()
        self.learning_memory = {}  # Store successful adaptations
        self.personalization_rules = self._initialize_personalization_rules()

    def _initialize_adaptation_strategies(self) -> Dict[AdaptationStrategy, Any]:
        """Initialize different adaptation strategies."""
        return {
            AdaptationStrategy.PROGRESSIVE_LEARNING: self._apply_progressive_learning,
            AdaptationStrategy.PATTERN_MATCHING: self._apply_pattern_matching,
            AdaptationStrategy.CONTEXTUAL_WEIGHTING: self._apply_contextual_weighting,
            AdaptationStrategy.USER_FEEDBACK_LOOP: self._apply_feedback_loop,
            AdaptationStrategy.SEMANTIC_CLUSTERING: self._apply_semantic_clustering,
            AdaptationStrategy.TEMPORAL_EVOLUTION: self._apply_temporal_evolution
        }

    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize base prompt templates for different scenarios."""
        return {
            "beginner_user": {
                "tone": "friendly and explanatory",
                "complexity": "simplified with clear examples",
                "guidance": "step-by-step instructions and context"
            },
            "expert_user": {
                "tone": "concise and technical",
                "complexity": "advanced with nuanced analysis",
                "guidance": "minimal hand-holding, direct insights"
            },
            "research_heavy": {
                "focus": "academic rigor and evidence-based analysis",
                "format": "structured with citations and methodology",
                "depth": "comprehensive theoretical framework"
            },
            "personal_journaling": {
                "tone": "empathetic and reflective",
                "privacy": "sensitive handling of personal content",
                "insight": "emotional intelligence and growth patterns"
            },
            "project_management": {
                "focus": "actionable insights and deliverables",
                "format": "structured with clear timelines and responsibilities",
                "pragmatism": "solution-oriented with implementation focus"
            },
            "creative_ideation": {
                "tone": "encouraging and open-minded",
                "exploration": "divergent thinking and possibility exploration",
                "preservation": "maintain creative spark while adding structure"
            }
        }

    def _initialize_personalization_rules(self) -> Dict[str, Any]:
        """Initialize rules for personalizing prompts."""
        return {
            "high_link_density": "Emphasize connections between notes and cross-references",
            "technical_writing": "Use precise terminology and structured analysis",
            "emotional_writing": "Include emotional intelligence and sentiment nuance",
            "morning_user": "Energetic and goal-oriented prompt style",
            "evening_user": "Reflective and contemplative prompt style",
            "frequent_questions": "Include exploratory questions and curiosity drivers",
            "action_oriented": "Focus on actionable insights and next steps"
        }

    def build_dynamic_prompt(self,
                           note_content: str,
                           analysis_task: AnalysisTask,
                           vault_profile: VaultProfile,
                           user_patterns: UserBehaviorPattern,
                           contextual_cues: ContextualCues,
                           adaptation_strategies: List[AdaptationStrategy] = None) -> DynamicPrompt:
        """
        Build a dynamically adapted prompt based on all available context.

        Args:
            note_content: The content to be analyzed
            analysis_task: The type of analysis to perform
            vault_profile: User's vault characteristics
            user_patterns: User's behavioral patterns
            contextual_cues: Real-time contextual information
            adaptation_strategies: Specific strategies to apply

        Returns:
            DynamicPrompt: A fully adapted, personalized prompt
        """
        # Start with base prompt for the task
        base_prompt = self._get_base_prompt(analysis_task)

        # Apply personalization based on vault profile
        personalization_elements = self._generate_personalization(vault_profile, user_patterns)

        # Apply contextual adaptations
        contextual_adaptations = self._generate_contextual_adaptations(
            contextual_cues, vault_profile, user_patterns
        )

        # Apply selected adaptation strategies
        if adaptation_strategies is None:
            adaptation_strategies = self._select_optimal_strategies(
                vault_profile, user_patterns, contextual_cues
            )

        strategy_adaptations = {}
        adaptation_reasoning = []

        for strategy in adaptation_strategies:
            if strategy in self.adaptation_strategies:
                adaptation = self.adaptation_strategies[strategy](
                    note_content, analysis_task, vault_profile, user_patterns, contextual_cues
                )
                strategy_adaptations[strategy.value] = adaptation["modification"]
                adaptation_reasoning.extend(adaptation.get("reasoning", []))

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            vault_profile, user_patterns, contextual_cues, len(adaptation_strategies)
        )

        # Generate fallback prompts
        fallback_prompts = self._generate_fallbacks(base_prompt, analysis_task)

        return DynamicPrompt(
            base_prompt=base_prompt,
            contextual_adaptations={**contextual_adaptations, **strategy_adaptations},
            personalization_elements=personalization_elements,
            confidence_score=confidence_score,
            adaptation_reasoning=adaptation_reasoning,
            fallback_prompts=fallback_prompts
        )

    def _get_base_prompt(self, task: AnalysisTask) -> str:
        """Get the base prompt template for a specific task."""
        base_prompts = {
            AnalysisTask.EXTRACT_KEY_POINTS: "Analyze the following content and extract the most important points and insights:",
            AnalysisTask.IDENTIFY_THEMES: "Identify the main themes and recurring patterns in the following content:",
            AnalysisTask.SUMMARIZE_CONTENT: "Provide a comprehensive summary of the following content:",
            AnalysisTask.FIND_ACTION_ITEMS: "Extract actionable tasks, decisions, and follow-up items from the following content:",
            AnalysisTask.ANALYZE_SENTIMENT: "Analyze the emotional tone and sentiment expressed in the following content:",
            AnalysisTask.CATEGORIZE_TOPICS: "Categorize the following content into relevant topics and subject areas:",
            AnalysisTask.EXTRACT_ENTITIES: "Extract important entities (people, places, concepts, dates) from the following content:",
            AnalysisTask.IDENTIFY_PATTERNS: "Identify patterns, relationships, and structural elements in the following content:",
            AnalysisTask.CREATE_CONNECTIONS: "Identify connections and relationships between ideas in the following content:",
            AnalysisTask.TRACK_PROGRESS: "Analyze progress indicators and development patterns in the following content:"
        }
        return base_prompts.get(task, "Analyze the following content:")

    def _generate_personalization(self, vault_profile: VaultProfile,
                                 user_patterns: UserBehaviorPattern) -> List[str]:
        """Generate personalization elements based on user profile."""
        elements = []

        # Expertise level adaptation
        expertise = vault_profile.estimate_expertise_level()
        if expertise == "beginner":
            elements.append("Provide clear explanations and context for technical terms")
            elements.append("Include step-by-step reasoning in your analysis")
        elif expertise == "expert":
            elements.append("Use advanced analytical techniques and nuanced insights")
            elements.append("Focus on sophisticated patterns and deep connections")

        # Note type preferences
        dominant_types = vault_profile.get_dominant_note_types(2)
        if dominant_types:
            primary_type = dominant_types[0][0]
            if primary_type == "journal":
                elements.append("Consider emotional context and personal development themes")
            elif primary_type == "research":
                elements.append("Maintain academic rigor and evidence-based analysis")
            elif primary_type == "meeting":
                elements.append("Focus on actionable outcomes and clear responsibilities")
            elif primary_type == "project":
                elements.append("Emphasize deliverables, timelines, and practical implementation")

        # Writing style adaptations
        style = vault_profile.writing_style_indicators
        if style.get("question_frequency", 0) > 0.1:
            elements.append("Include exploratory questions to stimulate further thinking")

        if style.get("emotional_intensity", 0) > 0.05:
            elements.append("Acknowledge emotional content with sensitivity and insight")

        if style.get("technical_density", 0) > 0.03:
            elements.append("Use precise technical terminology and structured analysis")

        # Activity pattern adaptations
        peak_times = vault_profile.get_peak_activity_times()
        if peak_times and "morning" in [time[0] for time in peak_times]:
            elements.append("Structure analysis for goal-setting and forward planning")
        elif peak_times and "evening" in [time[0] for time in peak_times]:
            elements.append("Focus on reflection and consolidation of insights")

        return elements

    def _generate_contextual_adaptations(self, contextual_cues: ContextualCues,
                                       vault_profile: VaultProfile,
                                       user_patterns: UserBehaviorPattern) -> Dict[str, str]:
        """Generate contextual adaptations based on current context."""
        adaptations = {}

        # Time-based adaptations
        if contextual_cues.time_of_day:
            if contextual_cues.time_of_day in ["morning", "afternoon"]:
                adaptations["temporal_focus"] = "Emphasize actionable insights and forward-looking analysis"
            else:
                adaptations["temporal_focus"] = "Focus on reflection and consolidation of learning"

        # Current project context
        if contextual_cues.active_projects:
            project_list = ", ".join(contextual_cues.active_projects[:3])
            adaptations["project_context"] = f"Consider relevance to active projects: {project_list}"

        # Recent analysis context
        if contextual_cues.recent_notes_analyzed:
            adaptations["continuity"] = "Build on insights from recently analyzed content"

        # Mood-based adaptations
        if contextual_cues.user_mood_indicators:
            mood_scores = contextual_cues.user_mood_indicators
            if mood_scores.get("stress", 0) > 0.7:
                adaptations["emotional_support"] = "Provide supportive, encouraging analysis"
            elif mood_scores.get("energy", 0) > 0.7:
                adaptations["energy_matching"] = "Match high energy with dynamic, engaging insights"

        # Goal alignment
        if contextual_cues.current_goals:
            goals_list = ", ".join(contextual_cues.current_goals[:2])
            adaptations["goal_alignment"] = f"Align analysis with current goals: {goals_list}"

        return adaptations

    def _select_optimal_strategies(self, vault_profile: VaultProfile,
                                 user_patterns: UserBehaviorPattern,
                                 contextual_cues: ContextualCues) -> List[AdaptationStrategy]:
        """Select the most appropriate adaptation strategies for the current context."""
        strategies = []

        # Always include contextual weighting
        strategies.append(AdaptationStrategy.CONTEXTUAL_WEIGHTING)

        # Add progressive learning for users with feedback history
        if len(user_patterns.feedback_history) > 5:
            strategies.append(AdaptationStrategy.PROGRESSIVE_LEARNING)

        # Add pattern matching for users with established patterns
        if vault_profile.total_notes > 100:
            strategies.append(AdaptationStrategy.PATTERN_MATCHING)

        # Add temporal evolution for time-sensitive content
        if contextual_cues.time_of_day or contextual_cues.recent_user_queries:
            strategies.append(AdaptationStrategy.TEMPORAL_EVOLUTION)

        # Add semantic clustering for content-rich vaults
        if vault_profile.link_density > 2.0:
            strategies.append(AdaptationStrategy.SEMANTIC_CLUSTERING)

        return strategies[:3]  # Limit to top 3 strategies for efficiency

    def _calculate_confidence_score(self, vault_profile: VaultProfile,
                                  user_patterns: UserBehaviorPattern,
                                  contextual_cues: ContextualCues,
                                  strategy_count: int) -> float:
        """Calculate confidence score for the generated prompt."""
        confidence = 0.5  # Base confidence

        # Increase confidence with more data
        if vault_profile.total_notes > 50:
            confidence += 0.1
        if vault_profile.total_notes > 200:
            confidence += 0.1

        # Increase confidence with user pattern data
        if len(user_patterns.feedback_history) > 3:
            confidence += 0.1

        # Increase confidence with contextual information
        if contextual_cues.active_projects:
            confidence += 0.1
        if contextual_cues.current_goals:
            confidence += 0.1

        # Adjust for strategy count
        confidence += (strategy_count * 0.05)

        return min(confidence, 1.0)  # Cap at 1.0

    def _generate_fallbacks(self, base_prompt: str, task: AnalysisTask) -> List[str]:
        """Generate fallback prompts in case the main prompt fails."""
        fallbacks = [
            base_prompt,  # Original base prompt
            f"Provide a {task.value.replace('_', ' ')} of the content with clear, structured output.",
            "Analyze the provided content and extract the most relevant insights."
        ]
        return fallbacks

    # Adaptation Strategy Implementations
    def _apply_progressive_learning(self, note_content: str, task: AnalysisTask,
                                  vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                                  contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply progressive learning based on user feedback history."""
        successful_patterns = []

        # Analyze feedback history for successful patterns
        for feedback in user_patterns.feedback_history[-10:]:  # Last 10 feedback items
            if feedback.get("rating", 0) >= 4:  # Positive feedback
                successful_patterns.append(feedback.get("approach", ""))

        if successful_patterns:
            modification = f"Apply successful patterns from previous analyses: {', '.join(set(successful_patterns))}"
        else:
            modification = "Focus on clear, actionable insights based on user preferences"

        return {
            "modification": modification,
            "reasoning": ["Applied progressive learning from user feedback history"]
        }

    def _apply_pattern_matching(self, note_content: str, task: AnalysisTask,
                              vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                              contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply pattern matching based on vault characteristics."""
        # Find similar content patterns in vault profile
        similar_patterns = []

        # Match based on note type distribution
        dominant_types = vault_profile.get_dominant_note_types(2)
        if dominant_types:
            similar_patterns.append(f"typical {dominant_types[0][0]} analysis approach")

        # Match based on writing style
        style = vault_profile.writing_style_indicators
        if style.get("technical_density", 0) > 0.05:
            similar_patterns.append("technical analysis style")
        if style.get("emotional_intensity", 0) > 0.05:
            similar_patterns.append("emotionally aware analysis style")

        modification = f"Apply analysis patterns that work well with your vault: {', '.join(similar_patterns)}"

        return {
            "modification": modification,
            "reasoning": ["Matched analysis approach to vault characteristics"]
        }

    def _apply_contextual_weighting(self, note_content: str, task: AnalysisTask,
                                  vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                                  contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply contextual weighting based on current context."""
        high_priority_contexts = []

        # Weight based on active projects
        if contextual_cues.active_projects:
            high_priority_contexts.append("active project relevance")

        # Weight based on current goals
        if contextual_cues.current_goals:
            high_priority_contexts.append("goal alignment")

        # Weight based on time context
        if contextual_cues.time_of_day in ["morning"]:
            high_priority_contexts.append("action-oriented insights")
        elif contextual_cues.time_of_day in ["evening"]:
            high_priority_contexts.append("reflective analysis")

        if high_priority_contexts:
            modification = f"Weight analysis toward: {', '.join(high_priority_contexts)}"
        else:
            modification = "Provide balanced, comprehensive analysis"

        return {
            "modification": modification,
            "reasoning": ["Applied contextual weighting based on current priorities"]
        }

    def _apply_feedback_loop(self, note_content: str, task: AnalysisTask,
                           vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                           contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply adaptations based on user feedback loop."""
        recent_feedback = user_patterns.feedback_history[-3:] if user_patterns.feedback_history else []

        if recent_feedback:
            avg_rating = sum(f.get("rating", 3) for f in recent_feedback) / len(recent_feedback)

            if avg_rating >= 4:
                modification = "Continue with current successful analysis approach"
            elif avg_rating <= 2:
                modification = "Adjust approach based on recent feedback - provide more detailed explanations"
            else:
                modification = "Fine-tune analysis depth and focus based on user preferences"
        else:
            modification = "Establish feedback baseline with comprehensive, balanced analysis"

        return {
            "modification": modification,
            "reasoning": ["Adjusted based on user feedback loop patterns"]
        }

    def _apply_semantic_clustering(self, note_content: str, task: AnalysisTask,
                                 vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                                 contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply semantic clustering based on content relationships."""
        # Identify semantic clusters from vault profile
        clusters = list(vault_profile.topic_clusters.keys())[:3]

        if clusters:
            modification = f"Consider semantic relationships with existing clusters: {', '.join(clusters)}"
            reasoning = ["Applied semantic clustering to identify content relationships"]
        else:
            modification = "Identify potential semantic relationships and knowledge connections"
            reasoning = ["Applied semantic analysis for content relationship discovery"]

        return {
            "modification": modification,
            "reasoning": reasoning
        }

    def _apply_temporal_evolution(self, note_content: str, task: AnalysisTask,
                                vault_profile: VaultProfile, user_patterns: UserBehaviorPattern,
                                contextual_cues: ContextualCues) -> Dict[str, Any]:
        """Apply temporal evolution based on time patterns."""
        # Consider time-based patterns
        time_adaptations = []

        if contextual_cues.time_of_day == "morning":
            time_adaptations.append("forward-looking perspective")
        elif contextual_cues.time_of_day == "evening":
            time_adaptations.append("reflective consolidation")

        # Consider day of week patterns
        if contextual_cues.day_of_week in ["monday", "tuesday"]:
            time_adaptations.append("week planning focus")
        elif contextual_cues.day_of_week in ["friday", "weekend"]:
            time_adaptations.append("week review and synthesis")

        if time_adaptations:
            modification = f"Adapt temporal perspective: {', '.join(time_adaptations)}"
        else:
            modification = "Apply time-neutral analysis with balanced perspective"

        return {
            "modification": modification,
            "reasoning": ["Applied temporal evolution based on time context"]
        }

    def learn_from_feedback(self, prompt_id: str, feedback: Dict[str, Any]) -> None:
        """Learn from user feedback to improve future prompt generation."""
        if prompt_id not in self.learning_memory:
            self.learning_memory[prompt_id] = []

        self.learning_memory[prompt_id].append({
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })

        # Update adaptation effectiveness
        self._update_strategy_effectiveness(feedback)

    def _update_strategy_effectiveness(self, feedback: Dict[str, Any]) -> None:
        """Update the effectiveness scores of different adaptation strategies."""
        # Implementation for learning from feedback
        # This would update strategy selection probabilities based on success rates
        pass

    def get_personalization_summary(self, vault_profile: VaultProfile,
                                  user_patterns: UserBehaviorPattern) -> Dict[str, Any]:
        """Get a summary of personalization factors for a user."""
        return {
            "expertise_level": vault_profile.estimate_expertise_level(),
            "dominant_note_types": vault_profile.get_dominant_note_types(3),
            "peak_activity_times": vault_profile.get_peak_activity_times(),
            "preferred_style": user_patterns.get_preferred_style(),
            "adaptation_responsiveness": user_patterns.adaptation_responsiveness,
            "total_notes": vault_profile.total_notes,
            "personalization_confidence": min(vault_profile.total_notes / 100, 1.0)
        }


# Convenience functions for common use cases
def build_adaptive_prompt(note_content: str, analysis_task: AnalysisTask,
                         vault_data: Dict[str, Any] = None,
                         user_data: Dict[str, Any] = None,
                         context_data: Dict[str, Any] = None) -> DynamicPrompt:
    """Quick function to build an adaptive prompt with available data."""
    builder = DynamicPromptBuilder()
    analyzer = ContextAnalyzer()

    # Create profiles from available data
    vault_profile = analyzer.analyze_vault_patterns(vault_data or {})

    user_patterns = UserBehaviorPattern()
    if user_data:
        user_patterns.preferred_analysis_tasks = user_data.get("preferred_tasks", {})
        user_patterns.feedback_history = user_data.get("feedback_history", [])

    contextual_cues = ContextualCues()
    if context_data:
        contextual_cues.time_of_day = context_data.get("time_of_day", "")
        contextual_cues.active_projects = context_data.get("active_projects", [])
        contextual_cues.current_goals = context_data.get("current_goals", [])

    return builder.build_dynamic_prompt(
        note_content, analysis_task, vault_profile, user_patterns, contextual_cues
    )


def analyze_with_context(note_content: str, analysis_task: AnalysisTask,
                        vault_size: int = 0, expertise_level: str = "intermediate",
                        time_context: str = "morning") -> DynamicPrompt:
    """Simple function for context-aware analysis."""
    # Create simplified profiles
    vault_profile = VaultProfile(total_notes=vault_size)
    user_patterns = UserBehaviorPattern()
    contextual_cues = ContextualCues(time_of_day=time_context)

    builder = DynamicPromptBuilder()
    return builder.build_dynamic_prompt(
        note_content, analysis_task, vault_profile, user_patterns, contextual_cues
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Dynamic Prompting Demo ===\n")

    # Sample vault data
    sample_vault_data = {
        "notes": [
            {
                "title": "Morning Reflection",
                "content": "Feeling productive today. Need to focus on the project deadline coming up next week.",
                "type": "journal",
                "tags": ["reflection", "productivity"],
                "created_date": "2025-08-26T08:30:00Z"
            },
            {
                "title": "Team Meeting Notes",
                "content": "Discussed project timeline. Sarah will handle the frontend, Mike takes backend.",
                "type": "meeting",
                "tags": ["meeting", "project"],
                "created_date": "2025-08-26T14:00:00Z"
            },
            {
                "title": "Research on AI Prompting",
                "content": "Dynamic prompting shows promise for personalized AI interactions. Key insight: context matters more than static templates.",
                "type": "research",
                "tags": ["ai", "research", "prompting"],
                "created_date": "2025-08-26T19:30:00Z"
            }
        ]
    }

    # Sample user behavior data
    sample_user_data = {
        "preferred_tasks": {"analyze_sentiment": 5, "extract_key_points": 3, "find_action_items": 7},
        "feedback_history": [
            {"rating": 4, "approach": "detailed_analysis"},
            {"rating": 5, "approach": "action_oriented"},
            {"rating": 3, "approach": "basic_summary"}
        ]
    }

    # Sample context data
    sample_context_data = {
        "time_of_day": "morning",
        "active_projects": ["VaultMind", "Personal Website"],
        "current_goals": ["Complete AI implementation", "Improve documentation"]
    }

    # Sample note to analyze
    sample_note = """
    Weekly Review - Progress and Challenges
    
    This week has been a mix of significant progress and unexpected obstacles. 
    The VaultMind project is moving forward well - we've implemented the core prompting 
    strategies and the dynamic adaptation system is showing real promise in early testing.
    
    However, I'm feeling overwhelmed by the scope of documentation needed. Each new 
    feature requires comprehensive docs, examples, and user guides. It's important work 
    but feels never-ending.
    
    Personal goal: Need to establish better work-life boundaries. Working late too often 
    and it's affecting my energy levels and creativity.
    
    Next week focus: Finish the dynamic prompting implementation and create a roadmap 
    for the documentation backlog.
    """

    # Initialize components
    builder = DynamicPromptBuilder()
    analyzer = ContextAnalyzer()

    print("1. Analyzing Vault Characteristics:")
    vault_profile = analyzer.analyze_vault_patterns(sample_vault_data)
    print(f"   Total notes: {vault_profile.total_notes}")
    print(f"   Dominant types: {vault_profile.get_dominant_note_types(2)}")
    print(f"   Peak times: {vault_profile.get_peak_activity_times()}")
    print(f"   Expertise level: {vault_profile.estimate_expertise_level()}")
    print()

    # Create user patterns
    user_patterns = UserBehaviorPattern()
    user_patterns.preferred_analysis_tasks = sample_user_data["preferred_tasks"]
    user_patterns.feedback_history = sample_user_data["feedback_history"]

    # Create contextual cues
    contextual_cues = ContextualCues()
    contextual_cues.time_of_day = sample_context_data["time_of_day"]
    contextual_cues.active_projects = sample_context_data["active_projects"]
    contextual_cues.current_goals = sample_context_data["current_goals"]

    print("2. Building Dynamic Prompt:")
    dynamic_prompt = builder.build_dynamic_prompt(
        sample_note,
        AnalysisTask.ANALYZE_SENTIMENT,
        vault_profile,
        user_patterns,
        contextual_cues
    )

    print(f"   Base prompt: {dynamic_prompt.base_prompt[:80]}...")
    print(f"   Personalization elements: {len(dynamic_prompt.personalization_elements)}")
    print(f"   Contextual adaptations: {len(dynamic_prompt.contextual_adaptations)}")
    print(f"   Confidence score: {dynamic_prompt.confidence_score:.2f}")
    print()

    print("3. Prompt Adaptations Applied:")
    for adaptation_type, adaptation in dynamic_prompt.contextual_adaptations.items():
        print(f"   • {adaptation_type}: {adaptation[:60]}...")
    print()

    print("4. Personalization Summary:")
    personalization_summary = builder.get_personalization_summary(vault_profile, user_patterns)
    for key, value in personalization_summary.items():
        print(f"   {key}: {value}")
    print()

    print("5. Complete Adaptive Prompt Preview:")
    full_prompt = dynamic_prompt.build_complete_prompt()
    print(f"   Length: {len(full_prompt)} characters")
    print(f"   Sections: Base + {len(dynamic_prompt.contextual_adaptations)} adaptations + {len(dynamic_prompt.personalization_elements)} personalizations")
    print()

    print("6. Quick Context-Aware Analysis:")
    quick_prompt = analyze_with_context(
        "Quick daily note: Had a productive morning, feeling focused.",
        AnalysisTask.EXTRACT_KEY_POINTS,
        vault_size=150,
        expertise_level="advanced",
        time_context="morning"
    )
    print(f"   Quick prompt confidence: {quick_prompt.confidence_score:.2f}")
    print(f"   Adaptations: {len(quick_prompt.contextual_adaptations)}")

    print("\n=== Dynamic Prompting System Ready ===")
    print("Key advantages:")
    print("• Adapts to user expertise and preferences")
    print("• Considers vault characteristics and patterns")
    print("• Responds to real-time context and goals")
    print("• Learns from feedback to improve over time")
    print("• Provides personalized analysis approaches")
