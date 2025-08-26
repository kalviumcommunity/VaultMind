"""
VaultMind Zero-Shot Prompting Implementation

This module implements zero-shot prompting for note analysis without requiring
training examples or labeled data. The system can analyze any note type using
only task descriptions and the model's pre-trained knowledge.

Zero-shot prompting leverages the model's inherent understanding to perform
tasks without seeing examples, making it highly flexible for diverse note types.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import re


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


class AnalysisTask(Enum):
    """Types of analysis tasks for zero-shot prompting."""
    EXTRACT_KEY_POINTS = "extract_key_points"
    IDENTIFY_THEMES = "identify_themes"
    SUMMARIZE_CONTENT = "summarize_content"
    FIND_ACTION_ITEMS = "find_action_items"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    CATEGORIZE_TOPICS = "categorize_topics"
    EXTRACT_ENTITIES = "extract_entities"
    IDENTIFY_PATTERNS = "identify_patterns"


class OutputStructure(Enum):
    """Output structure formats for analysis results."""
    LIST = "list"
    JSON = "json"
    PARAGRAPH = "paragraph"
    BULLET_POINTS = "bullet_points"
    STRUCTURED_DATA = "structured_data"
    NARRATIVE = "narrative"


@dataclass
class NoteContent:
    """Represents a note with its content and metadata."""
    title: str
    content: str
    note_type: Optional[NoteType] = None
    created_date: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_word_count(self) -> int:
        """Get the word count of the note content."""
        return len(self.content.split())

    def get_preview(self, words: int = 50) -> str:
        """Get a preview of the note content."""
        content_words = self.content.split()
        if len(content_words) <= words:
            return self.content
        return " ".join(content_words[:words]) + "..."


@dataclass
class ZeroShotPrompt:
    """Represents a zero-shot prompt with task instructions."""
    task_description: str
    context_instructions: str
    output_format: str
    constraints: List[str] = field(default_factory=list)
    examples_description: str = ""

    def build_prompt(self, note_content: str) -> str:
        """Build the complete zero-shot prompt."""
        prompt_parts = [
            f"**Task**: {self.task_description}",
            f"**Context**: {self.context_instructions}",
            f"**Output Format**: {self.output_format}"
        ]

        if self.constraints:
            constraints_text = "\n".join(f"- {constraint}" for constraint in self.constraints)
            prompt_parts.append(f"**Constraints**:\n{constraints_text}")

        if self.examples_description:
            prompt_parts.append(f"**What to look for**: {self.examples_description}")

        prompt_parts.extend([
            "**Content to analyze**:",
            note_content,
            "",
            "Provide your analysis following the specified format:"
        ])

        return "\n\n".join(prompt_parts)


class ZeroShotAnalyzer:
    """
    Main class for zero-shot note analysis without training examples.

    This analyzer can handle any note type and analysis task by leveraging
    the model's pre-trained knowledge and carefully crafted prompts.
    """

    def __init__(self):
        self.task_prompts = self._initialize_task_prompts()
        self.note_type_contexts = self._initialize_note_contexts()

    def _initialize_task_prompts(self) -> Dict[AnalysisTask, Dict[str, str]]:
        """Initialize task-specific prompt templates."""
        return {
            AnalysisTask.EXTRACT_KEY_POINTS: {
                "description": "Extract the most important points and insights from the content",
                "context": "You are analyzing personal notes to identify the core information that would be most valuable for future reference",
                "output_format": "A numbered list of key points, each with a brief explanation",
                "examples_description": "Main arguments, important facts, critical decisions, key insights, or significant observations"
            },

            AnalysisTask.IDENTIFY_THEMES: {
                "description": "Identify recurring themes, topics, and conceptual patterns in the content",
                "context": "You are analyzing content to understand the underlying themes and conceptual relationships",
                "output_format": "A list of themes with brief descriptions and relevance to the content",
                "examples_description": "Overarching topics, repeated concepts, philosophical themes, or subject matter categories"
            },

            AnalysisTask.SUMMARIZE_CONTENT: {
                "description": "Create a comprehensive yet concise summary of the content",
                "context": "You are creating a summary that captures the essential information while being significantly shorter than the original",
                "output_format": "A structured summary with main points organized logically",
                "examples_description": "Core message, key supporting points, important details, and conclusions"
            },

            AnalysisTask.FIND_ACTION_ITEMS: {
                "description": "Identify actionable tasks, decisions to be made, or follow-up items mentioned in the content",
                "context": "You are looking for concrete actions that need to be taken based on this content",
                "output_format": "A list of action items with priority levels and context",
                "examples_description": "Tasks to complete, decisions to make, people to contact, research to conduct, or deadlines to meet"
            },

            AnalysisTask.ANALYZE_SENTIMENT: {
                "description": "Analyze the emotional tone and sentiment expressed in the content",
                "context": "You are evaluating the emotional context and attitudes expressed in personal writing",
                "output_format": "Overall sentiment classification with supporting evidence and emotional indicators",
                "examples_description": "Positive/negative emotions, confidence levels, stress indicators, enthusiasm, or concern"
            },

            AnalysisTask.CATEGORIZE_TOPICS: {
                "description": "Categorize the content into relevant topic areas and subject domains",
                "context": "You are organizing content into logical categories for better knowledge management",
                "output_format": "A hierarchical categorization with primary and secondary topics",
                "examples_description": "Subject domains, knowledge areas, project categories, or thematic classifications"
            },

            AnalysisTask.EXTRACT_ENTITIES: {
                "description": "Extract important entities such as people, places, concepts, dates, and organizations",
                "context": "You are identifying specific entities that could be important for cross-referencing and connections",
                "output_format": "Categorized lists of entities with their context and relevance",
                "examples_description": "Names, locations, dates, organizations, concepts, tools, methods, or resources mentioned"
            },

            AnalysisTask.IDENTIFY_PATTERNS: {
                "description": "Identify patterns, relationships, and connections within the content",
                "context": "You are looking for structural patterns, logical relationships, and recurring elements",
                "output_format": "Description of patterns found with examples and significance",
                "examples_description": "Cause-effect relationships, temporal patterns, logical structures, or recurring elements"
            }
        }

    def _initialize_note_contexts(self) -> Dict[NoteType, str]:
        """Initialize note-type specific context information."""
        return {
            NoteType.JOURNAL: "This is a personal journal entry that may contain reflections, thoughts, daily experiences, and personal insights",
            NoteType.RESEARCH: "This is a research note containing academic or investigative content, findings, hypotheses, or scholarly information",
            NoteType.MEETING: "This is a meeting note containing discussion points, decisions made, action items, and participant interactions",
            NoteType.PROJECT: "This is a project-related note containing planning information, progress updates, tasks, or project-specific content",
            NoteType.IDEA: "This is an idea note containing creative thoughts, brainstorming content, concepts, or innovative thinking",
            NoteType.REFERENCE: "This is a reference note containing factual information, definitions, procedures, or reference material",
            NoteType.PERSONAL: "This is a personal note containing private thoughts, personal planning, or individual reflection",
            NoteType.ACADEMIC: "This is an academic note containing educational content, study materials, or learning-related information"
        }

    def analyze_note(self, note: NoteContent, task: AnalysisTask,
                    output_structure: OutputStructure = OutputStructure.LIST,
                    custom_constraints: Optional[List[str]] = None) -> ZeroShotPrompt:
        """
        Analyze a note using zero-shot prompting for the specified task.

        Args:
            note: The note content to analyze
            task: The type of analysis to perform
            output_structure: How to structure the output
            custom_constraints: Additional constraints for the analysis

        Returns:
            ZeroShotPrompt: A complete prompt ready for the AI model
        """
        # Get task-specific prompt template
        task_template = self.task_prompts.get(task)
        if not task_template:
            raise ValueError(f"Unknown analysis task: {task}")

        # Build context with note type information
        context_parts = [task_template["context"]]

        if note.note_type and note.note_type in self.note_type_contexts:
            context_parts.append(self.note_type_contexts[note.note_type])

        # Add note metadata context
        if note.tags:
            context_parts.append(f"The note is tagged with: {', '.join(note.tags)}")

        if note.created_date:
            context_parts.append(f"This note was created on: {note.created_date}")

        # Build output format instructions
        output_format = self._build_output_format(output_structure, task_template["output_format"])

        # Combine default and custom constraints
        constraints = [
            "Focus on the most relevant and important information",
            "Be specific and provide concrete examples when possible",
            "Maintain objectivity while being thorough"
        ]

        if custom_constraints:
            constraints.extend(custom_constraints)

        # Create the zero-shot prompt
        return ZeroShotPrompt(
            task_description=task_template["description"],
            context_instructions=". ".join(context_parts),
            output_format=output_format,
            constraints=constraints,
            examples_description=task_template["examples_description"]
        )

    def _build_output_format(self, structure: OutputStructure, base_format: str) -> str:
        """Build output format instructions based on structure preference."""
        structure_instructions = {
            OutputStructure.LIST: "Organize as a clear numbered or bulleted list",
            OutputStructure.JSON: "Structure as valid JSON with appropriate fields",
            OutputStructure.PARAGRAPH: "Present as coherent paragraphs with logical flow",
            OutputStructure.BULLET_POINTS: "Use bullet points with clear hierarchical organization",
            OutputStructure.STRUCTURED_DATA: "Organize as structured data with clear categories and subcategories",
            OutputStructure.NARRATIVE: "Present as a narrative explanation with logical progression"
        }

        structure_instruction = structure_instructions.get(structure, "")
        return f"{base_format}. {structure_instruction}".strip()

    def batch_analyze(self, notes: List[NoteContent], task: AnalysisTask,
                     output_structure: OutputStructure = OutputStructure.LIST) -> List[ZeroShotPrompt]:
        """
        Analyze multiple notes with the same task.

        Args:
            notes: List of notes to analyze
            task: The analysis task to perform on all notes
            output_structure: Output structure preference

        Returns:
            List of ZeroShotPrompt objects for batch processing
        """
        return [self.analyze_note(note, task, output_structure) for note in notes]

    def get_comparative_prompt(self, notes: List[NoteContent], task: AnalysisTask) -> ZeroShotPrompt:
        """
        Create a prompt for comparative analysis across multiple notes.

        Args:
            notes: List of notes to compare
            task: The analysis task to perform

        Returns:
            ZeroShotPrompt for comparative analysis
        """
        # Combine all note contents
        combined_content = "\n\n---NOTE SEPARATOR---\n\n".join([
            f"**Note {i+1}: {note.title}**\n{note.content}"
            for i, note in enumerate(notes)
        ])

        # Get base task template
        task_template = self.task_prompts.get(task)
        if not task_template:
            raise ValueError(f"Unknown analysis task: {task}")

        # Modify for comparative analysis
        comparative_description = f"Compare and analyze the following notes to {task_template['description'].lower()}"
        comparative_context = f"{task_template['context']}. You are performing comparative analysis across multiple related notes to identify patterns, differences, and connections."
        comparative_output = f"Provide a comparative analysis showing similarities, differences, and cross-note insights. {task_template['output_format']}"

        comparative_constraints = [
            "Identify patterns that span across multiple notes",
            "Highlight unique aspects of each note",
            "Show connections and relationships between notes",
            "Provide insights that emerge from the comparison"
        ]

        return ZeroShotPrompt(
            task_description=comparative_description,
            context_instructions=comparative_context,
            output_format=comparative_output,
            constraints=comparative_constraints,
            examples_description=f"Cross-note patterns, comparative insights, and relational analysis for: {task_template['examples_description']}"
        )

    def create_custom_task(self, task_name: str, description: str, context: str,
                          output_format: str, examples_description: str = "") -> AnalysisTask:
        """
        Create a custom analysis task for specialized needs.

        Args:
            task_name: Name for the custom task
            description: Task description
            context: Context instructions
            output_format: Expected output format
            examples_description: Description of what to look for

        Returns:
            Custom AnalysisTask that can be used with analyze_note
        """
        # Create a new task dynamically (in a real implementation, you might want to extend the enum)
        custom_task = f"CUSTOM_{task_name.upper()}"

        # Add to task prompts
        self.task_prompts[custom_task] = {
            "description": description,
            "context": context,
            "output_format": output_format,
            "examples_description": examples_description
        }

        return custom_task

    def get_task_suggestions(self, note: NoteContent) -> List[AnalysisTask]:
        """
        Suggest appropriate analysis tasks based on note type and content.

        Args:
            note: The note to analyze

        Returns:
            List of suggested analysis tasks
        """
        suggestions = []

        # Base suggestions for all notes
        suggestions.extend([
            AnalysisTask.EXTRACT_KEY_POINTS,
            AnalysisTask.IDENTIFY_THEMES,
            AnalysisTask.SUMMARIZE_CONTENT
        ])

        # Note type specific suggestions
        if note.note_type == NoteType.MEETING:
            suggestions.extend([
                AnalysisTask.FIND_ACTION_ITEMS,
                AnalysisTask.EXTRACT_ENTITIES
            ])
        elif note.note_type == NoteType.JOURNAL:
            suggestions.extend([
                AnalysisTask.ANALYZE_SENTIMENT,
                AnalysisTask.IDENTIFY_PATTERNS
            ])
        elif note.note_type == NoteType.RESEARCH:
            suggestions.extend([
                AnalysisTask.CATEGORIZE_TOPICS,
                AnalysisTask.EXTRACT_ENTITIES,
                AnalysisTask.IDENTIFY_PATTERNS
            ])
        elif note.note_type in [NoteType.PROJECT, NoteType.IDEA]:
            suggestions.extend([
                AnalysisTask.FIND_ACTION_ITEMS,
                AnalysisTask.CATEGORIZE_TOPICS
            ])

        # Content-based suggestions
        content_lower = note.content.lower()
        if any(word in content_lower for word in ["todo", "action", "follow up", "next steps"]):
            suggestions.append(AnalysisTask.FIND_ACTION_ITEMS)

        if any(word in content_lower for word in ["feeling", "think", "believe", "happy", "sad", "excited"]):
            suggestions.append(AnalysisTask.ANALYZE_SENTIMENT)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))


# Convenience functions for common use cases
def analyze_journal_entry(content: str, title: str = "Journal Entry",
                         created_date: str = None) -> ZeroShotPrompt:
    """Quick function to analyze a journal entry."""
    note = NoteContent(
        title=title,
        content=content,
        note_type=NoteType.JOURNAL,
        created_date=created_date or datetime.now().strftime("%Y-%m-%d")
    )

    analyzer = ZeroShotAnalyzer()
    return analyzer.analyze_note(note, AnalysisTask.ANALYZE_SENTIMENT, OutputStructure.NARRATIVE)


def analyze_meeting_notes(content: str, title: str = "Meeting Notes") -> ZeroShotPrompt:
    """Quick function to extract action items from meeting notes."""
    note = NoteContent(
        title=title,
        content=content,
        note_type=NoteType.MEETING
    )

    analyzer = ZeroShotAnalyzer()
    return analyzer.analyze_note(note, AnalysisTask.FIND_ACTION_ITEMS, OutputStructure.BULLET_POINTS)


def analyze_research_content(content: str, title: str = "Research Note") -> ZeroShotPrompt:
    """Quick function to extract key points from research content."""
    note = NoteContent(
        title=title,
        content=content,
        note_type=NoteType.RESEARCH
    )

    analyzer = ZeroShotAnalyzer()
    return analyzer.analyze_note(note, AnalysisTask.EXTRACT_KEY_POINTS, OutputStructure.STRUCTURED_DATA)


# Example usage and demonstration
if __name__ == "__main__":
    # Example usage of the ZeroShotAnalyzer
    print("=== VaultMind Zero-Shot Prompting Demo ===\n")

    # Sample notes for demonstration
    sample_journal = NoteContent(
        title="Daily Reflection - Project Progress",
        content="""
        Had a really productive day working on the VaultMind project. Finally got the zero-shot prompting 
        implementation working properly. The key insight was realizing that we don't need training examples 
        if we can describe the task clearly enough. Feeling confident about the direction we're taking.
        
        Still need to work on the UI components and test the integration with different note types. 
        Tomorrow I should focus on the meeting notes analysis feature.
        
        One concern is whether the prompts are too verbose - might need to optimize for token usage.
        """,
        note_type=NoteType.JOURNAL,
        created_date="2025-08-26",
        tags=["development", "reflection", "progress"]
    )

    sample_meeting = NoteContent(
        title="Team Standup - August 26",
        content="""
        Attendees: Alice, Bob, Charlie, Diana
        
        Alice: Completed the user authentication module. Blocked on database schema review.
        Bob: Working on API endpoints. Need clarification on error handling requirements.
        Charlie: Finished UI mockups. Will start implementation after design approval.
        Diana: Researching integration options. Meeting with vendor next week.
        
        Action items:
        - Alice to schedule database review meeting by Thursday
        - Bob to document error handling requirements
        - Charlie to get design approval from stakeholders
        - Diana to prepare vendor meeting agenda
        
        Next standup: Tomorrow at 9 AM
        """,
        note_type=NoteType.MEETING,
        tags=["standup", "team", "development"]
    )

    # Initialize analyzer
    analyzer = ZeroShotAnalyzer()

    # Demo 1: Analyze journal sentiment
    print("1. Journal Sentiment Analysis:")
    journal_prompt = analyzer.analyze_note(
        sample_journal,
        AnalysisTask.ANALYZE_SENTIMENT,
        OutputStructure.NARRATIVE
    )
    print(f"Task: {journal_prompt.task_description}")
    print(f"Output format: {journal_prompt.output_format}")
    print(f"Constraints: {len(journal_prompt.constraints)} constraints defined\n")

    # Demo 2: Extract meeting action items
    print("2. Meeting Action Items Extraction:")
    meeting_prompt = analyzer.analyze_note(
        sample_meeting,
        AnalysisTask.FIND_ACTION_ITEMS,
        OutputStructure.BULLET_POINTS
    )
    print(f"Task: {meeting_prompt.task_description}")
    print(f"Context includes: Meeting-specific analysis instructions")
    print(f"Full prompt length: {len(meeting_prompt.build_prompt(sample_meeting.content))} characters\n")

    # Demo 3: Task suggestions
    print("3. Suggested Tasks for Journal Note:")
    journal_suggestions = analyzer.get_task_suggestions(sample_journal)
    for task in journal_suggestions[:5]:  # Show first 5 suggestions
        print(f"   - {task.value}")

    print("\n4. Suggested Tasks for Meeting Note:")
    meeting_suggestions = analyzer.get_task_suggestions(sample_meeting)
    for task in meeting_suggestions[:5]:  # Show first 5 suggestions
        print(f"   - {task.value}")

    # Demo 4: Comparative analysis
    print("\n5. Comparative Analysis Setup:")
    comparative_prompt = analyzer.get_comparative_prompt(
        [sample_journal, sample_meeting],
        AnalysisTask.IDENTIFY_THEMES
    )
    print(f"Comparative task: {comparative_prompt.task_description}")
    print(f"Cross-note analysis: {len(comparative_prompt.constraints)} specialized constraints")

    print("\n=== Zero-Shot Prompting System Ready ===")
