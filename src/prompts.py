"""
VaultMind Prompt Management System

This module implements the RTFC (Role, Task, Format, Context) framework for
sophisticated prompt engineering in VaultMind's AI interactions.

Example Usage:
    # Basic usage
    from src.prompts import PromptManager, VaultContext, PromptType, OutputFormat

    # Create vault context
    vault = VaultContext(
        total_notes=120,
        note_types=["journal", "research"],
        main_themes=["AI", "productivity"],
        tags=["#ai", "#learning"]
    )
    
    # Create complete prompt
    manager = PromptManager()
    prompt = manager.create_complete_prompt(
        PromptType.CHAT_ASSISTANT,
        "CHAT_RESPONSE",
        vault_context=vault,
        output_format=OutputFormat.CONVERSATIONAL,
        user_question="What are my recurring themes this month?"
    )
    
    # Quick convenience functions
    from src.prompts import create_chat_prompt, create_vault_analyzer

    chat_prompt = create_chat_prompt("How can I improve productivity?", vault)
    analysis_prompt = create_vault_analyzer(vault, "focus on learning patterns")

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable
import re
from datetime import datetime


class PromptType(Enum):
    """Enumeration of available prompt types."""
    VAULT_ANALYZER = "vault_analyzer"
    CHAT_ASSISTANT = "chat_assistant"
    INSIGHT_EXTRACTOR = "insight_extractor"
    SUMMARIZER = "summarizer"
    PATTERN_FINDER = "pattern_finder"
    THEME_EXPLORER = "theme_explorer"


class OutputFormat(Enum):
    """Enumeration of supported output formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    YAML = "yaml"


@dataclass
class VaultContext:
    """Context information about the user's Obsidian vault."""
    total_notes: int = 0
    note_types: List[str] = field(default_factory=list)
    main_themes: List[str] = field(default_factory=list)
    date_range: Optional[str] = None
    vault_size_mb: Optional[float] = None
    recent_notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_context_string(self) -> str:
        """Convert vault context to a readable string for prompts."""
        context_parts = [
            f"Vault contains {self.total_notes} notes"
        ]
        
        if self.note_types:
            context_parts.append(f"Note types: {', '.join(self.note_types)}")
        
        if self.main_themes:
            context_parts.append(f"Main themes: {', '.join(self.main_themes[:5])}")
        
        if self.tags:
            context_parts.append(f"Popular tags: {', '.join(self.tags[:10])}")
        
        if self.date_range:
            context_parts.append(f"Date range: {self.date_range}")
            
        return ". ".join(context_parts) + "."


class BasePrompt(ABC):
    """Abstract base class for all prompt types."""
    
    def __init__(self, template: str, variables: Optional[Dict[str, Any]] = None):
        self.template = template
        self.variables = variables or {}
        self.created_at = datetime.now()
    
    @abstractmethod
    def render(self, **kwargs) -> str:
        """Render the prompt with provided variables."""
        pass
    
    def substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in text using {variable_name} format."""
        try:
            return text.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable: {e}")


class SystemPrompt(BasePrompt):
    """
    Defines the AI's role, personality, and core capabilities using RTFC framework.
    
    The system prompt establishes:
    - Role: Who the AI is (VaultMind knowledge analyst)
    - Task: What it should do (analyze, extract, assist)
    - Format: How it should respond (structured, conversational)
    - Context: What it knows about (Obsidian vaults, personal knowledge)
    """
    
    def __init__(self, prompt_type: PromptType, custom_instructions: Optional[str] = None):
        self.prompt_type = prompt_type
        self.custom_instructions = custom_instructions
        template = self._get_base_template()
        super().__init__(template)
    
    def _get_base_template(self) -> str:
        """Get the base RTFC template for the system prompt."""
        base_rtfc = """# ROLE
You are VaultMind, an expert knowledge analyst and personal AI assistant specializing in Obsidian vault analysis. You have deep expertise in:
- Personal knowledge management systems
- Markdown document analysis
- Pattern recognition in unstructured text
- Information synthesis and insight extraction
- Academic and personal note-taking methodologies

# TASK
Your primary responsibilities include:
- Analyzing Obsidian vault structures and content patterns
- Extracting meaningful insights from personal notes and documents
- Identifying connections between ideas, themes, and concepts
- Providing contextual responses based on vault content
- Helping users understand their knowledge landscape
- Maintaining privacy and confidentiality of personal information

# FORMAT
Always provide responses that are:
- Clear, concise, and actionable
- Properly formatted in the requested output style
- Supported by evidence from the vault content
- Respectful of personal and sensitive information
- Structured for easy comprehension and follow-up

# CONTEXT
You are working with personal Obsidian vaults that may contain:
- Daily journals and reflections
- Research notes and academic content
- Project documentation and planning
- Meeting notes and professional content
- Personal thoughts and creative writing
- Knowledge base articles and references

Remember: This is personal, private content. Always maintain confidentiality and provide helpful, non-judgmental assistance."""

        return base_rtfc + self._get_specialized_instructions()
    
    def _get_specialized_instructions(self) -> str:
        """Get specialized instructions based on prompt type."""
        specializations = {
            PromptType.VAULT_ANALYZER: """

# SPECIALIZED ROLE: Vault Structure Analyst
Focus on providing comprehensive vault analysis including:
- Note distribution and organization patterns
- Content type classification
- Temporal patterns in note creation
- Link density and knowledge graph structure
- Tag usage and categorization effectiveness""",

            PromptType.CHAT_ASSISTANT: """

# SPECIALIZED ROLE: Conversational Knowledge Assistant
Engage in natural, helpful conversations while:
- Drawing insights from vault content to answer questions
- Providing context-aware responses
- Suggesting related notes and connections
- Maintaining conversational flow while being informative
- Respecting the personal nature of the content""",

            PromptType.INSIGHT_EXTRACTOR: """

# SPECIALIZED ROLE: Insight and Pattern Analyst
Excel at identifying and extracting:
- Recurring themes and concepts across notes
- Evolution of ideas over time
- Hidden connections between disparate topics
- Emerging patterns in thinking and interests
- Knowledge gaps and areas for exploration""",

            PromptType.SUMMARIZER: """

# SPECIALIZED ROLE: Intelligent Content Summarizer
Provide high-quality summaries that:
- Preserve essential information and key insights
- Maintain the original context and intent
- Highlight important connections and relationships
- Structure information for easy scanning and review
- Adapt summary depth based on content complexity""",

            PromptType.PATTERN_FINDER: """

# SPECIALIZED ROLE: Pattern Recognition Expert
Identify and analyze patterns including:
- Writing habits and productivity cycles
- Topic evolution and interest shifts
- Knowledge acquisition patterns
- Note-taking methodology effectiveness
- Cross-references and idea development""",

            PromptType.THEME_EXPLORER: """

# SPECIALIZED ROLE: Thematic Content Explorer
Deep dive into thematic analysis by:
- Identifying major and minor themes across the vault
- Tracking theme development over time
- Exploring inter-theme relationships and hierarchies
- Suggesting theme-based content organization
- Highlighting unique or emerging thematic elements"""
        }
        
        return specializations.get(self.prompt_type, "")
    
    def render(self, output_format: OutputFormat = OutputFormat.CONVERSATIONAL, **kwargs) -> str:
        """Render the system prompt with specified format requirements."""
        format_instructions = self._get_format_instructions(output_format)
        
        full_prompt = self.template
        if format_instructions:
            full_prompt += f"\n\n# OUTPUT FORMAT REQUIREMENTS\n{format_instructions}"
        
        if self.custom_instructions:
            full_prompt += f"\n\n# ADDITIONAL INSTRUCTIONS\n{self.custom_instructions}"
        
        return self.substitute_variables(full_prompt, kwargs)
    
    def _get_format_instructions(self, output_format: OutputFormat) -> str:
        """Get specific format instructions."""
        format_map = {
            OutputFormat.JSON: "Always respond with valid JSON format. Use appropriate data types and structure.",
            OutputFormat.MARKDOWN: "Format responses using proper Markdown syntax with headers, lists, and emphasis.",
            OutputFormat.CONVERSATIONAL: "Respond in a natural, conversational tone while being informative and helpful.",
            OutputFormat.STRUCTURED: "Use clear structure with numbered points, bullet lists, and logical organization.",
            OutputFormat.YAML: "Provide responses in valid YAML format when structured data is requested."
        }
        return format_map.get(output_format, "")


class UserPrompt(BasePrompt):
    """
    Formats user queries with appropriate vault context and task-specific guidance.
    
    Handles dynamic content injection and ensures queries are properly contextualized
    for optimal AI performance.
    """
    
    def __init__(self, query: str, vault_context: Optional[VaultContext] = None, 
                 template_name: Optional[str] = None):
        self.query = query
        self.vault_context = vault_context
        self.template_name = template_name
        template = self._build_template()
        super().__init__(template)
    
    def _build_template(self) -> str:
        """Build the user prompt template."""
        template_parts = []
        
        if self.vault_context:
            template_parts.append("# VAULT CONTEXT")
            template_parts.append(self.vault_context.to_context_string())
            template_parts.append("")
        
        if self.template_name:
            template_parts.append("# TASK")
            template_parts.append(self._get_task_template())
            template_parts.append("")
        
        template_parts.append("# USER QUERY")
        template_parts.append(self.query)
        
        return "\n".join(template_parts)
    
    def _get_task_template(self) -> str:
        """Get task-specific template based on template name."""
        task_templates = {
            "ANALYZE_PATTERNS": "Analyze the patterns in my vault and provide insights about recurring themes, topics, and trends.",
            "EXTRACT_INSIGHTS": "Extract key insights and connections from my notes, focusing on important relationships between ideas.",
            "SUMMARIZE_CONTENT": "Provide a comprehensive summary of the specified content, highlighting key points and takeaways.",
            "FIND_CONNECTIONS": "Identify interesting connections and relationships between different notes and topics in my vault.",
            "EXPLORE_THEMES": "Explore the major themes present in my vault and how they relate to each other.",
            "PRODUCTIVITY_ANALYSIS": "Analyze my note-taking patterns to provide insights about productivity and knowledge management.",
        }
        return task_templates.get(self.template_name, "")
    
    def render(self, **kwargs) -> str:
        """Render the user prompt with any additional variables."""
        variables = {**self.variables, **kwargs}
        return self.substitute_variables(self.template, variables)


class PromptTemplate:
    """
    Handles template management with variable substitution and validation.
    
    Supports dynamic content, conditional sections, and robust error handling.
    """
    
    def __init__(self, name: str, template: str, required_vars: Optional[List[str]] = None,
                 validators: Optional[Dict[str, Callable]] = None):
        self.name = name
        self.template = template
        self.required_vars = required_vars or []
        self.validators = validators or {}
    
    def validate_variables(self, variables: Dict[str, Any]) -> None:
        """Validate that all required variables are present and valid."""
        # Check required variables
        missing_vars = [var for var in self.required_vars if var not in variables]
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Run custom validators
        for var_name, validator in self.validators.items():
            if var_name in variables:
                if not validator(variables[var_name]):
                    raise ValueError(f"Invalid value for variable '{var_name}': {variables[var_name]}")
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with provided variables."""
        self.validate_variables(variables)
        
        # Handle conditional sections [if:variable_name]content[/if]
        rendered = self._process_conditionals(self.template, variables)
        
        # Substitute variables
        try:
            rendered = rendered.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template variable not found: {e}")
        
        return rendered.strip()
    
    def _process_conditionals(self, text: str, variables: Dict[str, Any]) -> str:
        """Process conditional sections in templates."""
        pattern = r'\[if:(\w+)\](.*?)\[/if\]'
        
        def replace_conditional(match):
            var_name = match.group(1)
            content = match.group(2)
            return content if variables.get(var_name) else ""
        
        return re.sub(pattern, replace_conditional, text, flags=re.DOTALL)


class PromptManager:
    """
    Central manager for all prompt operations.
    
    Combines system and user prompts intelligently, manages templates,
    and provides a unified interface for prompt generation.
    """
    
    def __init__(self):
        self.templates = {}
        self.system_prompts = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self) -> None:
        """Initialize default prompt templates."""
        # Vault Analysis Templates
        self.add_template(
            "ANALYZE_PATTERNS",
            """Based on the vault context provided, analyze the following patterns:

1. **Content Patterns**: What types of content are most common?
2. **Temporal Patterns**: How has note-taking evolved over time?
3. **Thematic Patterns**: What are the main themes and how do they connect?
4. **Structural Patterns**: How is information organized and linked?

[if:specific_focus]Focus particularly on: {specific_focus}[/if]

Provide actionable insights that could help improve knowledge management.""",
            required_vars=[],
            validators={}
        )
        
        self.add_template(
            "EXTRACT_INSIGHTS",
            """Extract key insights from the vault content with focus on:

1. **Major Themes**: Identify 3-5 dominant themes with examples
2. **Connections**: Notable relationships between different topics
3. **Trends**: Patterns in thinking or interest evolution
4. **Gaps**: Areas that might benefit from more exploration
5. **Strengths**: Well-developed knowledge areas

[if:time_period]Focus on insights from: {time_period}[/if]
[if:specific_topics]Pay special attention to: {specific_topics}[/if]

Format as a structured analysis with specific examples from notes.""",
            required_vars=[],
            validators={}
        )
        
        self.add_template(
            "CHAT_RESPONSE",
            """You are having a conversation about the user's personal knowledge vault. 

Previous context: [if:conversation_history]{conversation_history}[/if]

Current question: {user_question}

Provide a helpful, contextual response that:
- Draws from relevant vault content
- Maintains conversational flow
- Offers specific examples when possible
- Suggests follow-up questions or exploration paths

Be natural and engaging while being informative.""",
            required_vars=["user_question"],
            validators={"user_question": lambda x: len(str(x).strip()) > 0}
        )
        
        self.add_template(
            "SUMMARIZE_NOTES",
            """Create a comprehensive summary of the specified content:

Content scope: {content_scope}
[if:note_count]Number of notes: {note_count}[/if]
[if:time_range]Time range: {time_range}[/if]

Include:
1. **Key Points**: Main ideas and findings
2. **Important Details**: Critical information that shouldn't be lost
3. **Connections**: How ideas relate to each other
4. **Action Items**: Any tasks or follow-ups mentioned
5. **Questions**: Unresolved issues or areas for exploration

[if:summary_length]Target length: {summary_length}[/if]
[if:focus_areas]Emphasize these areas: {focus_areas}[/if]""",
            required_vars=["content_scope"],
            validators={"content_scope": lambda x: len(str(x).strip()) > 0}
        )
    
    def add_template(self, name: str, template: str, required_vars: Optional[List[str]] = None,
                    validators: Optional[Dict[str, Callable]] = None) -> None:
        """Add a new prompt template."""
        self.templates[name] = PromptTemplate(name, template, required_vars, validators)
    
    def get_template(self, name: str) -> PromptTemplate:
        """Get a prompt template by name."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        return self.templates[name]
    
    def get_system_prompt(self, prompt_type: Union[PromptType, str], 
                         output_format: OutputFormat = OutputFormat.CONVERSATIONAL,
                         custom_instructions: Optional[str] = None) -> SystemPrompt:
        """Get a system prompt for the specified type."""
        if isinstance(prompt_type, str):
            prompt_type = PromptType(prompt_type)
        
        cache_key = f"{prompt_type.value}_{output_format.value}"
        
        if cache_key not in self.system_prompts:
            self.system_prompts[cache_key] = SystemPrompt(prompt_type, custom_instructions)
        
        return self.system_prompts[cache_key]
    
    def format_user_prompt(self, template: str, vault_context: Optional[VaultContext] = None,
                          user_query: Optional[str] = None, **variables) -> UserPrompt:
        """Format a user prompt using a template."""
        if template in self.templates:
            # Use template-based approach
            template_obj = self.get_template(template)
            rendered_query = template_obj.render(variables)
            return UserPrompt(rendered_query, vault_context, template)
        else:
            # Use direct query approach
            query = user_query or template
            return UserPrompt(query, vault_context)
    
    def create_complete_prompt(self, prompt_type: Union[PromptType, str],
                              user_template: str, vault_context: Optional[VaultContext] = None,
                              output_format: OutputFormat = OutputFormat.CONVERSATIONAL,
                              **variables) -> str:
        """Create a complete prompt combining system and user prompts."""
        # Get system prompt
        system_prompt = self.get_system_prompt(prompt_type, output_format)
        system_rendered = system_prompt.render(output_format=output_format)
        
        # Get user prompt
        user_prompt = self.format_user_prompt(user_template, vault_context, **variables)
        user_rendered = user_prompt.render(**variables)
        
        # Combine prompts
        complete_prompt = f"{system_rendered}\n\n---\n\n{user_rendered}"
        
        return complete_prompt
    
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    
    def list_prompt_types(self) -> List[str]:
        """List all available prompt types."""
        return [pt.value for pt in PromptType]
    
    def validate_prompt(self, prompt: str, max_tokens: int = 8000) -> Dict[str, Any]:
        """Validate a prompt for common issues."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "token_estimate": len(prompt.split()) * 1.3  # Rough token estimation
        }
        
        # Check length
        if validation_result["token_estimate"] > max_tokens:
            validation_result["warnings"].append(f"Prompt may exceed {max_tokens} tokens")
        
        # Check for empty sections
        if not prompt.strip():
            validation_result["errors"].append("Prompt is empty")
            validation_result["valid"] = False
        
        # Check for unresolved variables
        unresolved_vars = re.findall(r'\{(\w+)\}', prompt)
        if unresolved_vars:
            validation_result["warnings"].append(f"Unresolved variables: {', '.join(unresolved_vars)}")
        
        return validation_result


# Convenience functions for quick prompt creation
def create_vault_analyzer(vault_context: VaultContext, specific_analysis: str = "") -> str:
    """Quick function to create a vault analysis prompt."""
    manager = PromptManager()
    return manager.create_complete_prompt(
        PromptType.VAULT_ANALYZER,
        "ANALYZE_PATTERNS",
        vault_context=vault_context,
        output_format=OutputFormat.STRUCTURED,
        specific_focus=specific_analysis
    )

def create_chat_prompt(user_question: str, vault_context: Optional[VaultContext] = None,
                      conversation_history: str = "") -> str:
    """Quick function to create a chat prompt."""
    manager = PromptManager()
    return manager.create_complete_prompt(
        PromptType.CHAT_ASSISTANT,
        "CHAT_RESPONSE",
        vault_context=vault_context,
        output_format=OutputFormat.CONVERSATIONAL,
        user_question=user_question,
        conversation_history=conversation_history
    )

def create_insight_extractor(vault_context: VaultContext, time_period: str = "",
                           specific_topics: str = "") -> str:
    """Quick function to create an insight extraction prompt."""
    manager = PromptManager()
    return manager.create_complete_prompt(
        PromptType.INSIGHT_EXTRACTOR,
        "EXTRACT_INSIGHTS",
        vault_context=vault_context,
        output_format=OutputFormat.STRUCTURED,
        time_period=time_period,
        specific_topics=specific_topics
    )

def create_summarizer(content_scope: str, vault_context: Optional[VaultContext] = None,
                     note_count: int = 0, summary_length: str = "detailed") -> str:
    """Quick function to create a summarization prompt."""
    manager = PromptManager()
    variables = {
        "content_scope": content_scope,
        "summary_length": summary_length
    }
    if note_count > 0:
        variables["note_count"] = note_count
    
    return manager.create_complete_prompt(
        PromptType.SUMMARIZER,
        "SUMMARIZE_NOTES",
        vault_context=vault_context,
        output_format=OutputFormat.MARKDOWN,
        **variables
    )


# Example usage and testing
if __name__ == "__main__":
    # Example vault context
    sample_vault = VaultContext(
        total_notes=120,
        note_types=["journal", "research", "project"],
        main_themes=["AI", "productivity", "learning"],
        date_range="2024-01-01 to 2024-08-22",
        vault_size_mb=15.2,
        tags=["#ai", "#productivity", "#learning", "#research"]
    )
    
    # Example usage
    print("=== VaultMind Prompt System Demo ===\n")
    
    # 1. Create a vault analyzer prompt
    analyzer_prompt = create_vault_analyzer(
        sample_vault, 
        "productivity patterns and learning progression"
    )
    print("1. Vault Analyzer Prompt (truncated):")
    print(analyzer_prompt[:500] + "...\n")
    
    # 2. Create a chat prompt
    chat_prompt = create_chat_prompt(
        "What are my most productive times for writing?",
        sample_vault,
        "User previously asked about writing habits."
    )
    print("2. Chat Prompt (truncated):")
    print(chat_prompt[:500] + "...\n")
    
    # 3. Test prompt manager directly
    manager = PromptManager()
    print("3. Available Templates:")
    print(manager.list_templates())
    print("\n4. Available Prompt Types:")
    print(manager.list_prompt_types())
