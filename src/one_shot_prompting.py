"""
VaultMind One-Shot Prompting Implementation

This module implements one-shot prompting for note analysis using single example
templates. Unlike zero-shot prompting, one-shot provides the AI with one concrete
example to establish the pattern and desired output format.

One-shot prompting bridges the gap between zero-shot (no examples) and few-shot
(multiple examples) approaches, providing clear guidance with minimal overhead.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json


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
    """Types of analysis tasks for one-shot prompting."""
    EXTRACT_KEY_POINTS = "extract_key_points"
    IDENTIFY_THEMES = "identify_themes"
    SUMMARIZE_CONTENT = "summarize_content"
    FIND_ACTION_ITEMS = "find_action_items"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    CATEGORIZE_TOPICS = "categorize_topics"
    EXTRACT_ENTITIES = "extract_entities"
    CREATE_OUTLINE = "create_outline"


class OutputFormat(Enum):
    """Output format specifications."""
    JSON = "json"
    MARKDOWN = "markdown"
    BULLET_POINTS = "bullet_points"
    STRUCTURED_LIST = "structured_list"
    NARRATIVE = "narrative"
    TABLE = "table"


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
class ExampleTemplate:
    """Represents an example template for one-shot prompting."""
    example_input: str
    example_output: str
    explanation: str
    note_type: NoteType
    analysis_task: AnalysisTask
    output_format: OutputFormat

    def format_example(self) -> str:
        """Format the example for inclusion in prompts."""
        return f"""**Example Input:**
{self.example_input}

**Example Output:**
{self.example_output}

**Explanation:** {self.explanation}"""


@dataclass
class OneShotPrompt:
    """Represents a one-shot prompt with task instructions and one example."""
    task_description: str
    context_instructions: str
    example_template: ExampleTemplate
    target_content: str
    output_format: str
    constraints: List[str] = field(default_factory=list)

    def build_prompt(self) -> str:
        """Build the complete one-shot prompt."""
        prompt_parts = [
            f"**Task**: {self.task_description}",
            f"**Context**: {self.context_instructions}",
            "",
            "**Example of the desired analysis:**",
            self.example_template.format_example(),
            "",
            f"**Output Format**: {self.output_format}"
        ]

        if self.constraints:
            constraints_text = "\n".join(f"- {constraint}" for constraint in self.constraints)
            prompt_parts.append(f"**Constraints**:\n{constraints_text}")

        prompt_parts.extend([
            "",
            "**Now analyze this content following the same pattern:**",
            self.target_content,
            "",
            "Provide your analysis:"
        ])

        return "\n".join(prompt_parts)


class OneShotAnalyzer:
    """
    Main class for one-shot note analysis using single example templates.

    This analyzer provides concrete examples to guide AI analysis, ensuring
    consistent output format and quality across different note types.
    """

    def __init__(self):
        self.example_templates = self._initialize_example_templates()
        self.task_descriptions = self._initialize_task_descriptions()
        self.note_type_contexts = self._initialize_note_contexts()

    def _initialize_example_templates(self) -> Dict[Tuple[NoteType, AnalysisTask, OutputFormat], ExampleTemplate]:
        """Initialize example templates for different combinations."""
        templates = {}

        # Journal + Sentiment Analysis + Narrative
        templates[(NoteType.JOURNAL, AnalysisTask.ANALYZE_SENTIMENT, OutputFormat.NARRATIVE)] = ExampleTemplate(
            example_input="""Daily Reflection - March 15th
Today was challenging but rewarding. Started the morning feeling overwhelmed by the project deadlines, but after breaking down the tasks into smaller chunks, I felt much more in control. The team meeting went really well - everyone was collaborative and supportive. I'm excited about the new direction we're taking, though slightly nervous about the technical challenges ahead. Overall feeling optimistic and motivated for tomorrow.""",

            example_output="""**Overall Sentiment: Positive with Mixed Emotions**

The entry shows a clear emotional journey from initial stress to confidence and optimism. Key emotional indicators:

**Positive Elements:**
- Sense of accomplishment ("challenging but rewarding")  
- Problem-solving confidence ("felt much more in control")
- Social satisfaction ("team meeting went really well")
- Future optimism ("excited about the new direction")

**Stress Indicators:**
- Initial overwhelm ("feeling overwhelmed by project deadlines")
- Mild anxiety ("slightly nervous about technical challenges")

**Emotional Trajectory:** The author demonstrates healthy emotional processing, moving from stress to empowerment through practical problem-solving. The closing sentiment is distinctly positive and forward-looking.""",

            explanation="This analysis identifies the emotional progression, specific sentiment markers, and overall psychological state while providing concrete evidence from the text.",
            note_type=NoteType.JOURNAL,
            analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
            output_format=OutputFormat.NARRATIVE
        )

        # Meeting + Action Items + Bullet Points
        templates[(NoteType.MEETING, AnalysisTask.FIND_ACTION_ITEMS, OutputFormat.BULLET_POINTS)] = ExampleTemplate(
            example_input="""Weekly Team Sync - March 10, 2025
Attendees: Sarah (PM), Mike (Dev), Lisa (Design), Tom (QA)

Sarah: Budget review complete. Need approval from finance by Friday. Will schedule follow-up.
Mike: API integration 80% done. Blocked on third-party documentation. Reaching out to vendor support.
Lisa: New mockups ready for review. Team should provide feedback by Wednesday.
Tom: Found critical bug in payment flow. Needs immediate attention from dev team.

Decisions:
- Moving sprint review to next Tuesday
- Hiring additional frontend developer approved
- Lisa will lead the design system initiative

Next meeting: Same time next week. Tom to demo the bug fix.""",

            example_output="""**Immediate Action Items (High Priority):**
• Sarah: Schedule and complete finance approval meeting by Friday
• Mike: Contact vendor support for API documentation (blocking issue)
• Tom: Fix critical payment flow bug immediately
• Dev team: Address payment flow bug found by Tom

**This Week Actions:**
• Lisa: Share new mockups with team for review
• All team members: Provide mockup feedback to Lisa by Wednesday
• Tom: Prepare bug fix demo for next meeting

**Ongoing/Future Actions:**
• Sarah: Begin hiring process for additional frontend developer
• Lisa: Start leading design system initiative
• Team: Prepare for sprint review (moved to next Tuesday)

**Follow-up Items:**
• Next weekly sync: Same time next week
• Sprint review: Moved to next Tuesday""",

            explanation="Action items are categorized by urgency and ownership, with clear deadlines and dependencies identified.",
            note_type=NoteType.MEETING,
            analysis_task=AnalysisTask.FIND_ACTION_ITEMS,
            output_format=OutputFormat.BULLET_POINTS
        )

        # Research + Key Points + Structured List
        templates[(NoteType.RESEARCH, AnalysisTask.EXTRACT_KEY_POINTS, OutputFormat.STRUCTURED_LIST)] = ExampleTemplate(
            example_input="""Research Notes: Transformer Architecture in Natural Language Processing

The Transformer model, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized NLP by eliminating recurrence and convolution entirely. Instead, it relies solely on attention mechanisms to draw global dependencies between input and output.

Key architectural components include multi-head self-attention, which allows the model to jointly attend to information from different representation subspaces. The positional encoding provides sequence order information since the model has no inherent notion of position. The encoder-decoder structure processes sequences in parallel rather than sequentially, enabling much faster training.

The attention mechanism computes attention weights as A(Q,K,V) = softmax(QK^T/√d_k)V, where Q, K, and V represent queries, keys, and values respectively. This allows direct modeling of dependencies regardless of distance in the sequence.

Performance improvements over RNNs include better parallelization, reduced training time, and superior handling of long-range dependencies. State-of-the-art results achieved on machine translation tasks, with BLEU scores significantly exceeding previous models.""",

            example_output="""**1. Core Innovation**
   - Eliminates recurrence and convolution entirely
   - Relies solely on attention mechanisms for dependencies
   - Processes sequences in parallel vs. sequentially

**2. Key Architectural Elements**
   - Multi-head self-attention for different representation subspaces
   - Positional encoding to provide sequence order information
   - Encoder-decoder structure for parallel processing

**3. Technical Foundation**
   - Attention formula: A(Q,K,V) = softmax(QK^T/√d_k)V
   - Q, K, V represent queries, keys, and values
   - Direct modeling of dependencies regardless of sequence distance

**4. Performance Advantages**
   - Better parallelization capabilities
   - Reduced training time compared to RNNs
   - Superior long-range dependency handling
   - State-of-the-art BLEU scores on machine translation

**5. Research Impact**
   - Introduced in "Attention Is All You Need" (Vaswani et al., 2017)
   - Revolutionized NLP field
   - Foundation for subsequent transformer-based models""",

            explanation="Key points are organized hierarchically with technical details preserved and structured for easy reference and further study.",
            note_type=NoteType.RESEARCH,
            analysis_task=AnalysisTask.EXTRACT_KEY_POINTS,
            output_format=OutputFormat.STRUCTURED_LIST
        )

        # Project + Themes + JSON
        templates[(NoteType.PROJECT, AnalysisTask.IDENTIFY_THEMES, OutputFormat.JSON)] = ExampleTemplate(
            example_input="""VaultMind Project Update - Week 12

Progress this week focused on three main areas. First, the prompt engineering system is nearly complete with both zero-shot and one-shot implementations finished. The modular design allows easy extension for new prompting strategies. Second, user interface development has accelerated with the CLI framework fully functional and preliminary GUI mockups ready for review.

Technical challenges emerged around vector database integration. Performance bottlenecks appeared when indexing large vaults, requiring optimization of the embedding pipeline. The team decided to implement incremental indexing and caching strategies to address scalability concerns.

User feedback from beta testing has been overwhelmingly positive. Testers particularly appreciated the intelligent task suggestions and the clean, intuitive interface. Some requested additional export formats and batch processing capabilities, which we've added to the roadmap.

Looking ahead, the focus shifts to documentation, testing coverage, and deployment preparation. The goal is to have a production-ready release by month-end with comprehensive user guides and API documentation.""",

            example_output="""{
  "major_themes": [
    {
      "theme": "Technical Development Progress",
      "description": "Core system components reaching completion",
      "evidence": ["prompt engineering system nearly complete", "CLI framework fully functional", "GUI mockups ready"],
      "significance": "Shows strong development velocity and systematic approach"
    },
    {
      "theme": "Performance and Scalability",
      "description": "Addressing system performance challenges",
      "evidence": ["vector database integration challenges", "performance bottlenecks with large vaults", "incremental indexing implementation"],
      "significance": "Critical for production readiness and user adoption"
    },
    {
      "theme": "User-Centered Design",
      "description": "Focus on user experience and feedback integration",
      "evidence": ["positive beta testing feedback", "intelligent task suggestions", "clean, intuitive interface"],
      "significance": "Validates product-market fit and design decisions"
    },
    {
      "theme": "Product Roadmap and Planning",
      "description": "Strategic planning for launch and future features",
      "evidence": ["additional export formats requested", "batch processing capabilities", "production-ready release timeline"],
      "significance": "Shows clear path to market and feature prioritization"
    }
  ],
  "recurring_concepts": ["user feedback", "system performance", "modular design", "production readiness"],
  "temporal_focus": "current_sprint_and_near_future",
  "primary_concerns": ["scalability", "user_experience", "deployment_timeline"]
}""",

            explanation="Themes are identified with supporting evidence and significance, structured as JSON for programmatic processing and analysis.",
            note_type=NoteType.PROJECT,
            analysis_task=AnalysisTask.IDENTIFY_THEMES,
            output_format=OutputFormat.JSON
        )

        # Idea + Outline + Markdown
        templates[(NoteType.IDEA, AnalysisTask.CREATE_OUTLINE, OutputFormat.MARKDOWN)] = ExampleTemplate(
            example_input="""Idea: Personal Knowledge Graph Visualization Tool

What if we could visualize personal knowledge as an interactive 3D graph where concepts, people, projects, and ideas are nodes connected by relationships? Users could explore their thoughts spatially, discovering unexpected connections.

The interface would allow zooming from high-level themes down to specific notes. Color coding could represent different types of content - blue for factual knowledge, green for personal reflections, red for urgent items. Edge thickness could show connection strength.

Implementation could use WebGL for performance with D3.js for interaction. The backend would analyze note content to suggest connections, but users could manually add relationships too. Time-based animations could show knowledge evolution.

This could help with writer's block by showing related concepts. Research workflows would benefit from visual connection discovery. Students could see how their learning builds over time. The tool might reveal knowledge gaps or overconnected areas needing balance.""",

            example_output="""# Personal Knowledge Graph Visualization Tool

## I. Core Concept
   ### A. Interactive 3D Knowledge Representation
   - Spatial exploration of personal knowledge
   - Nodes represent concepts, people, projects, ideas
   - Connections show relationships between elements
   
   ### B. Discovery Through Visualization
   - Unexpected connection identification
   - Knowledge pattern recognition
   - Spatial knowledge navigation

## II. User Interface Design
   ### A. Multi-Level Exploration
   - High-level theme overview
   - Zoom to specific note details
   - Seamless scale transitions
   
   ### B. Visual Coding System
   - Color-coded content types (blue: facts, green: reflections, red: urgent)
   - Edge thickness indicates connection strength
   - Visual hierarchy and organization

## III. Technical Implementation
   ### A. Frontend Technologies
   - WebGL for performance optimization
   - D3.js for interactive controls
   - Responsive 3D rendering
   
   ### B. Backend Intelligence
   - Automated connection suggestion
   - Manual relationship addition
   - Time-based evolution tracking

## IV. Use Cases and Benefits
   ### A. Creative Applications
   - Writer's block resolution
   - Related concept discovery
   
   ### B. Academic and Professional Uses
   - Research workflow enhancement
   - Student learning progression visualization
   - Knowledge gap identification""",

            explanation="The outline structures the idea hierarchically, organizing concepts logically while preserving the creative vision and practical considerations.",
            note_type=NoteType.IDEA,
            analysis_task=AnalysisTask.CREATE_OUTLINE,
            output_format=OutputFormat.MARKDOWN
        )

        return templates

    def _initialize_task_descriptions(self) -> Dict[AnalysisTask, str]:
        """Initialize task descriptions for one-shot prompting."""
        return {
            AnalysisTask.EXTRACT_KEY_POINTS: "Extract the most important points, insights, and information from the content, organizing them in a clear, hierarchical structure",
            AnalysisTask.IDENTIFY_THEMES: "Identify recurring themes, conceptual patterns, and major topics within the content, showing how they relate to each other",
            AnalysisTask.SUMMARIZE_CONTENT: "Create a comprehensive yet concise summary that captures the essential meaning and important details of the content",
            AnalysisTask.FIND_ACTION_ITEMS: "Identify actionable tasks, decisions to be made, deadlines, and follow-up items mentioned in the content",
            AnalysisTask.ANALYZE_SENTIMENT: "Analyze the emotional tone, attitudes, and sentiment expressed in the content, identifying emotional patterns and progression",
            AnalysisTask.CATEGORIZE_TOPICS: "Categorize the content into relevant topic areas and subject domains for better organization and retrieval",
            AnalysisTask.EXTRACT_ENTITIES: "Extract important entities such as people, places, dates, organizations, concepts, and tools mentioned in the content",
            AnalysisTask.CREATE_OUTLINE: "Create a structured outline that organizes the content into logical sections with clear hierarchy and flow"
        }

    def _initialize_note_contexts(self) -> Dict[NoteType, str]:
        """Initialize note-type specific context information."""
        return {
            NoteType.JOURNAL: "This is a personal journal entry containing thoughts, reflections, daily experiences, and personal insights. Focus on emotional content and personal development themes.",
            NoteType.RESEARCH: "This is academic or investigative research content containing findings, hypotheses, methodologies, and scholarly information. Maintain technical accuracy and academic rigor.",
            NoteType.MEETING: "This is meeting documentation containing discussions, decisions, action items, and participant interactions. Focus on actionable outcomes and clear responsibilities.",
            NoteType.PROJECT: "This is project-related content containing planning information, progress updates, tasks, and project-specific details. Emphasize deliverables and timeline information.",
            NoteType.IDEA: "This is creative ideation content containing brainstorming, concepts, innovative thinking, and creative exploration. Preserve creative vision while adding structure.",
            NoteType.REFERENCE: "This is reference material containing factual information, procedures, definitions, or documentation. Maintain accuracy and clear organization.",
            NoteType.PERSONAL: "This is personal content containing private thoughts, personal planning, goals, or individual reflection. Handle with sensitivity and focus on personal growth themes.",
            NoteType.ACADEMIC: "This is educational content containing learning materials, study notes, or academic coursework. Structure for effective learning and review."
        }

    def analyze_note(self, note: NoteContent, task: AnalysisTask,
                    output_format: OutputFormat = OutputFormat.STRUCTURED_LIST,
                    custom_constraints: Optional[List[str]] = None) -> OneShotPrompt:
        """
        Analyze a note using one-shot prompting with a relevant example.

        Args:
            note: The note content to analyze
            task: The type of analysis to perform
            output_format: How to structure the output
            custom_constraints: Additional constraints for the analysis

        Returns:
            OneShotPrompt: A complete prompt with example ready for the AI model
        """
        # Get the best matching example template
        example_template = self._get_best_example(note.note_type, task, output_format)

        # Get task description
        task_description = self.task_descriptions.get(task, "Analyze the content as specified")

        # Build context
        context_parts = [task_description]
        if note.note_type and note.note_type in self.note_type_contexts:
            context_parts.append(self.note_type_contexts[note.note_type])

        # Add note metadata context
        if note.tags:
            context_parts.append(f"Note tags: {', '.join(note.tags)}")

        if note.created_date:
            context_parts.append(f"Created: {note.created_date}")

        # Build output format instructions
        output_instructions = self._get_output_format_instructions(output_format)

        # Default constraints
        constraints = [
            "Follow the same structure and style as the example",
            "Be specific and provide concrete details",
            "Maintain consistency with the example format",
            "Focus on the most relevant and important information"
        ]

        if custom_constraints:
            constraints.extend(custom_constraints)

        return OneShotPrompt(
            task_description=task_description,
            context_instructions=". ".join(context_parts),
            example_template=example_template,
            target_content=note.content,
            output_format=output_instructions,
            constraints=constraints
        )

    def _get_best_example(self, note_type: Optional[NoteType], task: AnalysisTask,
                         output_format: OutputFormat) -> ExampleTemplate:
        """Get the best matching example template."""
        # Try exact match first
        if note_type:
            key = (note_type, task, output_format)
            if key in self.example_templates:
                return self.example_templates[key]

        # Try to find same task and format with different note type
        for (nt, at, of), template in self.example_templates.items():
            if at == task and of == output_format:
                return template

        # Try to find same task with different format
        for (nt, at, of), template in self.example_templates.items():
            if at == task:
                return template

        # Fallback to first available template
        return next(iter(self.example_templates.values()))

    def _get_output_format_instructions(self, output_format: OutputFormat) -> str:
        """Get specific output format instructions."""
        format_instructions = {
            OutputFormat.JSON: "Structure the output as valid JSON with clear field names and proper data types",
            OutputFormat.MARKDOWN: "Use proper Markdown formatting with headers, lists, and emphasis where appropriate",
            OutputFormat.BULLET_POINTS: "Use bullet points with clear hierarchy and consistent formatting",
            OutputFormat.STRUCTURED_LIST: "Use numbered lists and sublists with clear organization and hierarchy",
            OutputFormat.NARRATIVE: "Present as flowing narrative text with logical paragraph structure",
            OutputFormat.TABLE: "Organize information in a clear table format with appropriate columns and rows"
        }
        return format_instructions.get(output_format, "Follow the example format exactly")

    def get_available_examples(self, note_type: Optional[NoteType] = None,
                              task: Optional[AnalysisTask] = None) -> List[ExampleTemplate]:
        """Get all available example templates, optionally filtered."""
        examples = list(self.example_templates.values())

        if note_type:
            examples = [ex for ex in examples if ex.note_type == note_type]

        if task:
            examples = [ex for ex in examples if ex.analysis_task == task]

        return examples

    def add_custom_example(self, example: ExampleTemplate) -> None:
        """Add a custom example template to the analyzer."""
        key = (example.note_type, example.analysis_task, example.output_format)
        self.example_templates[key] = example

    def batch_analyze(self, notes: List[NoteContent], task: AnalysisTask,
                     output_format: OutputFormat = OutputFormat.STRUCTURED_LIST) -> List[OneShotPrompt]:
        """Analyze multiple notes using the same task and format."""
        return [self.analyze_note(note, task, output_format) for note in notes]

    def compare_with_zero_shot(self, note: NoteContent, task: AnalysisTask) -> Dict[str, OneShotPrompt]:
        """Compare one-shot approach with a zero-shot equivalent for the same task."""
        # One-shot version
        one_shot = self.analyze_note(note, task)

        # Zero-shot version (simplified, without example)
        zero_shot_prompt = OneShotPrompt(
            task_description=self.task_descriptions[task],
            context_instructions=self.note_type_contexts.get(note.note_type, "Analyze the following content"),
            example_template=ExampleTemplate("", "", "No example provided - zero-shot approach", note.note_type or NoteType.PERSONAL, task, OutputFormat.STRUCTURED_LIST),
            target_content=note.content,
            output_format="Provide analysis in a clear, organized format",
            constraints=["Be thorough and specific", "Use clear organization"]
        )

        return {
            "one_shot": one_shot,
            "zero_shot": zero_shot_prompt
        }


# Convenience functions for common use cases
def analyze_with_example(content: str, note_type: NoteType, task: AnalysisTask,
                        output_format: OutputFormat = OutputFormat.STRUCTURED_LIST,
                        title: str = "Note") -> OneShotPrompt:
    """Quick function to analyze content with appropriate example."""
    note = NoteContent(title=title, content=content, note_type=note_type)
    analyzer = OneShotAnalyzer()
    return analyzer.analyze_note(note, task, output_format)


def analyze_journal_with_example(content: str, title: str = "Journal Entry") -> OneShotPrompt:
    """Analyze journal entry sentiment with example guidance."""
    return analyze_with_example(
        content, NoteType.JOURNAL, AnalysisTask.ANALYZE_SENTIMENT,
        OutputFormat.NARRATIVE, title
    )


def analyze_meeting_with_example(content: str, title: str = "Meeting Notes") -> OneShotPrompt:
    """Extract meeting action items with example guidance."""
    return analyze_with_example(
        content, NoteType.MEETING, AnalysisTask.FIND_ACTION_ITEMS,
        OutputFormat.BULLET_POINTS, title
    )


def analyze_research_with_example(content: str, title: str = "Research Notes") -> OneShotPrompt:
    """Extract research key points with example guidance."""
    return analyze_with_example(
        content, NoteType.RESEARCH, AnalysisTask.EXTRACT_KEY_POINTS,
        OutputFormat.STRUCTURED_LIST, title
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind One-Shot Prompting Demo ===\n")

    # Sample note for demonstration
    sample_note = NoteContent(
        title="Team Retrospective - Sprint 8",
        content="""
        Sprint 8 Retrospective - August 25, 2025
        
        What went well:
        - Delivered all planned features on time
        - Great collaboration between design and development
        - Automated testing coverage improved significantly
        - Customer feedback was overwhelmingly positive
        
        What could be improved:
        - Communication gaps during the middle of sprint
        - Some technical debt accumulated in the payment module
        - Documentation updates fell behind development pace
        
        Action items for next sprint:
        - Schedule daily check-ins for first week
        - Dedicate 20% of sprint to technical debt reduction
        - Assign documentation ownership for each feature
        - Set up automated documentation generation
        
        Team sentiment is high. Everyone feels good about the direction and pace.
        """,
        note_type=NoteType.MEETING,
        created_date="2025-08-25",
        tags=["retrospective", "sprint", "team"]
    )

    # Initialize analyzer
    analyzer = OneShotAnalyzer()

    # Demo 1: Extract action items with example
    print("1. One-Shot Action Item Extraction:")
    action_prompt = analyzer.analyze_note(
        sample_note,
        AnalysisTask.FIND_ACTION_ITEMS,
        OutputFormat.BULLET_POINTS
    )
    print(f"Task: {action_prompt.task_description}")
    print(f"Uses example: {action_prompt.example_template.note_type.value} + {action_prompt.example_template.analysis_task.value}")
    print(f"Example explanation: {action_prompt.example_template.explanation[:100]}...")
    print(f"Full prompt length: {len(action_prompt.build_prompt())} characters\n")

    # Demo 2: Compare one-shot vs zero-shot
    print("2. One-Shot vs Zero-Shot Comparison:")
    comparison = analyzer.compare_with_zero_shot(sample_note, AnalysisTask.FIND_ACTION_ITEMS)

    one_shot_length = len(comparison["one_shot"].build_prompt())
    zero_shot_length = len(comparison["zero_shot"].build_prompt())

    print(f"One-shot prompt: {one_shot_length} characters (includes example)")
    print(f"Zero-shot prompt: {zero_shot_length} characters (no example)")
    print(f"Difference: {one_shot_length - zero_shot_length} characters for example guidance\n")

    # Demo 3: Available examples
    print("3. Available Example Templates:")
    examples = analyzer.get_available_examples()
    for i, example in enumerate(examples[:3], 1):  # Show first 3
        print(f"   {i}. {example.note_type.value} + {example.analysis_task.value} → {example.output_format.value}")

    print(f"\nTotal examples available: {len(examples)}")

    # Demo 4: Custom example addition
    print("\n4. Custom Example Capability:")
    custom_example = ExampleTemplate(
        example_input="Sample custom input",
        example_output="Sample custom output",
        explanation="Custom example for specialized analysis",
        note_type=NoteType.PERSONAL,
        analysis_task=AnalysisTask.SUMMARIZE_CONTENT,
        output_format=OutputFormat.TABLE
    )

    print(f"Can add custom examples: {custom_example.note_type.value} analysis")
    print(f"Before adding: {len(analyzer.example_templates)} templates")
    analyzer.add_custom_example(custom_example)
    print(f"After adding: {len(analyzer.example_templates)} templates")

    print("\n=== One-Shot Prompting System Ready ===")
