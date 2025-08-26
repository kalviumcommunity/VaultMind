"""
VaultMind RTFC Framework Implementation

This module implements the RTFC (Role, Task, Format, Context) framework for
structured prompt engineering in VaultMind CLI tool.

RTFC Components:
- Role: Defines who the AI is and its expertise
- Task: Specifies what the AI should accomplish
- Format: Determines how the response should be structured
- Context: Provides relevant information for informed responses

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any


class RTFCRole(Enum):
    """Available AI roles for VaultMind operations."""
    KNOWLEDGE_ANALYST = "knowledge_analyst"
    VAULT_EXPLORER = "vault_explorer"
    INSIGHT_EXTRACTOR = "insight_extractor"
    PERSONAL_ASSISTANT = "personal_assistant"
    NOTE_SYNTHESIZER = "note_synthesizer"


class RTFCTaskType(Enum):
    """Types of tasks the AI can perform."""
    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    EXTRACT_INSIGHTS = "extract_insights"
    FIND_CONNECTIONS = "find_connections"
    ANSWER_QUESTION = "answer_question"
    ORGANIZE_INFORMATION = "organize_information"


class RTFCFormat(Enum):
    """Output format specifications."""
    JSON = "json"
    MARKDOWN = "markdown"
    BULLET_POINTS = "bullet_points"
    CONVERSATIONAL = "conversational"
    STRUCTURED_LIST = "structured_list"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class RTFCContext:
    """Context information for RTFC framework."""
    vault_info: Dict[str, Any] = None
    user_query: str = ""
    relevant_notes: List[str] = None
    timestamp: str = ""
    user_preferences: Dict[str, Any] = None

    def __post_init__(self):
        if self.vault_info is None:
            self.vault_info = {}
        if self.relevant_notes is None:
            self.relevant_notes = []
        if self.user_preferences is None:
            self.user_preferences = {}


class RTFCPrompt(ABC):
    """Abstract base class for RTFC-structured prompts."""

    def __init__(self, role: RTFCRole, task: RTFCTaskType, format_type: RTFCFormat, context: RTFCContext):
        self.role = role
        self.task = task
        self.format_type = format_type
        self.context = context

    @abstractmethod
    def build_prompt(self) -> str:
        """Build the complete RTFC prompt."""
        pass

    def _build_role_section(self) -> str:
        """Build the Role section of RTFC."""
        role_definitions = {
            RTFCRole.KNOWLEDGE_ANALYST: """You are VaultMind's Knowledge Analyst, an expert in personal knowledge management and information synthesis. You specialize in:
- Analyzing patterns in personal note-taking systems
- Identifying knowledge gaps and connections
- Extracting actionable insights from unstructured information
- Understanding individual learning and thinking patterns""",

            RTFCRole.VAULT_EXPLORER: """You are VaultMind's Vault Explorer, a specialist in navigating and understanding Obsidian vault structures. Your expertise includes:
- Mapping relationships between notes and ideas
- Understanding vault organization patterns
- Discovering hidden connections across topics
- Analyzing information architecture""",

            RTFCRole.INSIGHT_EXTRACTOR: """You are VaultMind's Insight Extractor, focused on deriving meaningful conclusions from personal knowledge bases. You excel at:
- Identifying recurring themes and patterns
- Synthesizing information from multiple sources
- Generating actionable recommendations
- Highlighting important trends and developments""",

            RTFCRole.PERSONAL_ASSISTANT: """You are VaultMind's Personal Assistant, designed to help users interact with their knowledge in natural, conversational ways. You specialize in:
- Understanding user intent and context
- Providing personalized responses based on individual knowledge
- Facilitating knowledge discovery and exploration
- Supporting decision-making with relevant information""",

            RTFCRole.NOTE_SYNTHESIZER: """You are VaultMind's Note Synthesizer, expert in combining and organizing information from multiple sources. Your strengths include:
- Merging complementary information from different notes
- Creating coherent summaries from fragmented content
- Identifying and resolving information conflicts
- Structuring knowledge for optimal understanding"""
        }

        return f"# ROLE\n{role_definitions[self.role]}\n"

    def _build_task_section(self) -> str:
        """Build the Task section of RTFC."""
        task_instructions = {
            RTFCTaskType.ANALYZE: "Analyze the provided information to identify patterns, structures, and key characteristics. Focus on understanding the underlying organization and relationships.",

            RTFCTaskType.SUMMARIZE: "Create a comprehensive summary that captures the essential information, main points, and key takeaways while maintaining the original meaning and context.",

            RTFCTaskType.EXTRACT_INSIGHTS: "Extract meaningful insights, trends, and actionable information from the provided content. Focus on discoveries that add value to the user's understanding.",

            RTFCTaskType.FIND_CONNECTIONS: "Identify and map connections, relationships, and patterns between different pieces of information. Highlight both obvious and subtle relationships.",

            RTFCTaskType.ANSWER_QUESTION: "Provide a thorough, accurate answer to the user's question using the available context. Ensure the response directly addresses the query.",

            RTFCTaskType.ORGANIZE_INFORMATION: "Structure and organize the provided information in a logical, coherent manner that enhances understanding and usability."
        }

        return f"# TASK\n{task_instructions[self.task]}\n"

    def _build_format_section(self) -> str:
        """Build the Format section of RTFC."""
        format_specifications = {
            RTFCFormat.JSON: """Provide your response in valid JSON format with clear key-value pairs. Structure should be logical and easy to parse programmatically.""",

            RTFCFormat.MARKDOWN: """Format your response using clean Markdown syntax with appropriate headers, lists, and emphasis. Use proper structure for readability.""",

            RTFCFormat.BULLET_POINTS: """Present information as clear, concise bullet points. Use nested bullets for sub-items and maintain consistent formatting.""",

            RTFCFormat.CONVERSATIONAL: """Respond in a natural, conversational tone as if speaking directly to the user. Be friendly, informative, and engaging.""",

            RTFCFormat.STRUCTURED_LIST: """Organize information in numbered or categorized lists with clear sections and subsections for easy navigation.""",

            RTFCFormat.EXECUTIVE_SUMMARY: """Provide a professional executive summary format with key findings, recommendations, and action items clearly highlighted."""
        }

        return f"# FORMAT\n{format_specifications[self.format_type]}\n"

    def _build_context_section(self) -> str:
        """Build the Context section of RTFC."""
        context_parts = ["# CONTEXT"]

        if self.context.vault_info:
            vault_summary = []
            for key, value in self.context.vault_info.items():
                if isinstance(value, list):
                    vault_summary.append(f"{key.title()}: {', '.join(map(str, value[:5]))}")
                else:
                    vault_summary.append(f"{key.title()}: {value}")
            context_parts.append(f"Vault Information: {'; '.join(vault_summary)}")

        if self.context.user_query:
            context_parts.append(f"User Query: {self.context.user_query}")

        if self.context.relevant_notes:
            context_parts.append(f"Relevant Notes Available: {len(self.context.relevant_notes)} notes")

        if self.context.user_preferences:
            prefs = [f"{k}: {v}" for k, v in self.context.user_preferences.items()]
            context_parts.append(f"User Preferences: {'; '.join(prefs)}")

        return "\n".join(context_parts) + "\n"


class SystemPrompt(RTFCPrompt):
    """System prompt implementing RTFC framework for AI behavior definition."""

    def __init__(self, role: RTFCRole, task: RTFCTaskType, format_type: RTFCFormat, context: RTFCContext, custom_instructions: str = ""):
        super().__init__(role, task, format_type, context)
        self.custom_instructions = custom_instructions

    def build_prompt(self) -> str:
        """Build complete system prompt using RTFC framework."""
        prompt_parts = [
            self._build_role_section(),
            self._build_task_section(),
            self._build_format_section(),
            self._build_context_section()
        ]

        if self.custom_instructions:
            prompt_parts.append(f"# ADDITIONAL INSTRUCTIONS\n{self.custom_instructions}\n")

        prompt_parts.append("# GUIDELINES\n- Be accurate and honest about limitations\n- Cite sources when referencing specific notes\n- Maintain user privacy and data confidentiality\n- Provide actionable and relevant insights\n")

        return "\n".join(prompt_parts)


class UserPrompt(RTFCPrompt):
    """User prompt implementing RTFC framework for specific user requests."""

    def __init__(self, task: RTFCTaskType, format_type: RTFCFormat, context: RTFCContext, specific_request: str = ""):
        # User prompts typically use PERSONAL_ASSISTANT role
        super().__init__(RTFCRole.PERSONAL_ASSISTANT, task, format_type, context)
        self.specific_request = specific_request

    def build_prompt(self) -> str:
        """Build complete user prompt using RTFC framework."""
        if self.specific_request:
            return f"USER REQUEST: {self.specific_request}\n\nPlease respond according to the system instructions above."

        # Default user prompt based on task type
        task_requests = {
            RTFCTaskType.ANALYZE: "Please analyze my vault and provide insights about the content, structure, and patterns you discover.",
            RTFCTaskType.SUMMARIZE: "Please summarize the key information from my notes in a clear and concise manner.",
            RTFCTaskType.EXTRACT_INSIGHTS: "What insights and patterns can you extract from my knowledge base?",
            RTFCTaskType.FIND_CONNECTIONS: "Help me find connections and relationships between different topics in my vault.",
            RTFCTaskType.ANSWER_QUESTION: f"Please answer my question: {self.context.user_query}",
            RTFCTaskType.ORGANIZE_INFORMATION: "Help me organize and structure the information in my vault."
        }

        return f"USER REQUEST: {task_requests.get(self.task, 'Please assist me with my vault.')}\n\nPlease respond according to the system instructions above."


class RTFCPromptBuilder:
    """Builder class for creating RTFC-structured prompts easily."""

    @staticmethod
    def create_analysis_prompt(vault_info: Dict[str, Any], format_type: RTFCFormat = RTFCFormat.MARKDOWN) -> tuple[SystemPrompt, UserPrompt]:
        """Create prompts for vault analysis."""
        context = RTFCContext(vault_info=vault_info, user_query="Analyze my vault")

        system_prompt = SystemPrompt(
            role=RTFCRole.KNOWLEDGE_ANALYST,
            task=RTFCTaskType.ANALYZE,
            format_type=format_type,
            context=context
        )

        user_prompt = UserPrompt(
            task=RTFCTaskType.ANALYZE,
            format_type=format_type,
            context=context
        )

        return system_prompt, user_prompt

    @staticmethod
    def create_chat_prompt(user_question: str, vault_info: Dict[str, Any], relevant_notes: List[str] = None) -> tuple[SystemPrompt, UserPrompt]:
        """Create prompts for conversational interaction."""
        context = RTFCContext(
            vault_info=vault_info,
            user_query=user_question,
            relevant_notes=relevant_notes or []
        )

        system_prompt = SystemPrompt(
            role=RTFCRole.PERSONAL_ASSISTANT,
            task=RTFCTaskType.ANSWER_QUESTION,
            format_type=RTFCFormat.CONVERSATIONAL,
            context=context
        )

        user_prompt = UserPrompt(
            task=RTFCTaskType.ANSWER_QUESTION,
            format_type=RTFCFormat.CONVERSATIONAL,
            context=context,
            specific_request=user_question
        )

        return system_prompt, user_prompt

    @staticmethod
    def create_summary_prompt(vault_info: Dict[str, Any], notes_to_summarize: List[str], format_type: RTFCFormat = RTFCFormat.EXECUTIVE_SUMMARY) -> tuple[SystemPrompt, UserPrompt]:
        """Create prompts for summarization tasks."""
        context = RTFCContext(
            vault_info=vault_info,
            relevant_notes=notes_to_summarize,
            user_query="Summarize the provided notes"
        )

        system_prompt = SystemPrompt(
            role=RTFCRole.NOTE_SYNTHESIZER,
            task=RTFCTaskType.SUMMARIZE,
            format_type=format_type,
            context=context
        )

        user_prompt = UserPrompt(
            task=RTFCTaskType.SUMMARIZE,
            format_type=format_type,
            context=context
        )

        return system_prompt, user_prompt


# Convenience functions for quick prompt creation
def create_quick_analysis(vault_info: Dict[str, Any]) -> str:
    """Quick function to create analysis prompt string."""
    system, user = RTFCPromptBuilder.create_analysis_prompt(vault_info)
    return f"{system.build_prompt()}\n\n{user.build_prompt()}"


def create_quick_chat(question: str, vault_info: Dict[str, Any]) -> str:
    """Quick function to create chat prompt string."""
    system, user = RTFCPromptBuilder.create_chat_prompt(question, vault_info)
    return f"{system.build_prompt()}\n\n{user.build_prompt()}"


def create_quick_summary(vault_info: Dict[str, Any], notes: List[str]) -> str:
    """Quick function to create summary prompt string."""
    system, user = RTFCPromptBuilder.create_summary_prompt(vault_info, notes)
    return f"{system.build_prompt()}\n\n{user.build_prompt()}"
