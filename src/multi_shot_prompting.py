"""
VaultMind Multi-Shot Prompting Implementation

This module implements multi-shot prompting for note analysis using 3-5 diverse
examples per analysis type. Multi-shot prompting provides multiple examples to
show pattern variations, edge cases, and progressive complexity levels.

Multi-shot prompting excels when tasks require understanding of nuanced patterns,
handling diverse input types, or demonstrating complex analysis techniques that
benefit from seeing multiple approaches.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import random


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
    """Types of analysis tasks for multi-shot prompting."""
    EXTRACT_KEY_POINTS = "extract_key_points"
    IDENTIFY_THEMES = "identify_themes"
    SUMMARIZE_CONTENT = "summarize_content"
    FIND_ACTION_ITEMS = "find_action_items"
    ANALYZE_SENTIMENT = "analyze_sentiment"
    CATEGORIZE_TOPICS = "categorize_topics"
    EXTRACT_ENTITIES = "extract_entities"
    IDENTIFY_PATTERNS = "identify_patterns"
    CREATE_CONNECTIONS = "create_connections"


class ComplexityLevel(Enum):
    """Complexity levels for progressive example selection."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


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
    complexity_level: Optional[ComplexityLevel] = None

    def get_word_count(self) -> int:
        """Get the word count of the note content."""
        return len(self.content.split())

    def estimate_complexity(self) -> ComplexityLevel:
        """Estimate the complexity level of the note content."""
        word_count = self.get_word_count()
        unique_concepts = len(set(word.lower() for word in self.content.split() if len(word) > 4))

        if word_count < 100 and unique_concepts < 20:
            return ComplexityLevel.SIMPLE
        elif word_count < 300 and unique_concepts < 50:
            return ComplexityLevel.MODERATE
        elif word_count < 600 and unique_concepts < 100:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXPERT


@dataclass
class MultiShotExample:
    """Represents a single example in a multi-shot prompt."""
    example_input: str
    example_output: str
    explanation: str
    complexity_level: ComplexityLevel
    note_type: NoteType
    analysis_task: AnalysisTask
    output_format: OutputFormat
    tags: List[str] = field(default_factory=list)

    def format_example(self, example_number: int) -> str:
        """Format the example for inclusion in prompts."""
        return f"""**Example {example_number} ({self.complexity_level.value} level):**

*Input:*
{self.example_input}

*Output:*
{self.example_output}

*Why this works:* {self.explanation}"""


@dataclass
class MultiShotPrompt:
    """Represents a multi-shot prompt with task instructions and multiple examples."""
    task_description: str
    context_instructions: str
    examples: List[MultiShotExample]
    target_content: str
    output_format: str
    constraints: List[str] = field(default_factory=list)
    example_selection_strategy: str = "progressive"

    def build_prompt(self) -> str:
        """Build the complete multi-shot prompt."""
        prompt_parts = [
            f"**Task**: {self.task_description}",
            f"**Context**: {self.context_instructions}",
            "",
            f"**Learning from {len(self.examples)} examples:**"
        ]

        # Add examples
        for i, example in enumerate(self.examples, 1):
            prompt_parts.extend([
                "",
                example.format_example(i),
                ""
            ])

        prompt_parts.extend([
            f"**Output Format**: {self.output_format}",
            ""
        ])

        if self.constraints:
            constraints_text = "\n".join(f"- {constraint}" for constraint in self.constraints)
            prompt_parts.append(f"**Guidelines**:\n{constraints_text}\n")

        prompt_parts.extend([
            f"**Pattern Recognition**: Notice how the examples show different approaches and complexity levels. Apply these patterns to analyze the following content:",
            "",
            "**Content to analyze:**",
            self.target_content,
            "",
            "**Your analysis (following the patterns above):**"
        ])

        return "\n".join(prompt_parts)

    def get_example_summary(self) -> Dict[str, Any]:
        """Get a summary of the examples used."""
        return {
            "total_examples": len(self.examples),
            "complexity_levels": [ex.complexity_level.value for ex in self.examples],
            "selection_strategy": self.example_selection_strategy,
            "example_types": [ex.note_type.value for ex in self.examples],
            "prompt_length": len(self.build_prompt())
        }


class MultiShotAnalyzer:
    """
    Main class for multi-shot note analysis using diverse example sets.

    This analyzer uses 3-5 carefully curated examples per task to demonstrate
    pattern variations, complexity progression, and edge case handling.
    """

    def __init__(self):
        self.example_repository = self._initialize_example_repository()
        self.task_descriptions = self._initialize_task_descriptions()
        self.note_type_contexts = self._initialize_note_contexts()
        self.selection_strategies = self._initialize_selection_strategies()

    def _initialize_example_repository(self) -> Dict[Tuple[AnalysisTask, NoteType], List[MultiShotExample]]:
        """Initialize comprehensive example repository."""
        repository = {}

        # Sentiment Analysis Examples for Journal Notes
        journal_sentiment_examples = [
            MultiShotExample(
                example_input="Quick check-in: Feeling pretty good today. Got some work done, had lunch with Sarah. Weather's nice.",
                example_output="**Overall Sentiment: Positive (Mild)**\n- Contentment indicators: 'feeling pretty good', 'got some work done'\n- Social connection: positive lunch experience\n- Environmental satisfaction: 'weather's nice'\n- Tone: Casual, satisfied, low-energy positive",
                explanation="Simple, straightforward positive sentiment with clear indicators and casual tone analysis.",
                complexity_level=ComplexityLevel.SIMPLE,
                note_type=NoteType.JOURNAL,
                analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
                output_format=OutputFormat.STRUCTURED_LIST
            ),
            MultiShotExample(
                example_input="Mixed feelings today. The presentation went really well - felt confident and got great feedback from the team. But then I got some concerning news about the budget cuts. Trying to stay optimistic but honestly feeling a bit anxious about what this means for our projects.",
                example_output="**Overall Sentiment: Mixed (Positive-Negative Tension)**\n\n**Positive Elements:**\n- Achievement satisfaction: 'presentation went really well'\n- Confidence boost: 'felt confident'\n- External validation: 'great feedback from the team'\n\n**Negative Elements:**\n- Worry source: 'concerning news about budget cuts'\n- Future anxiety: 'feeling a bit anxious about what this means'\n\n**Emotional Processing:**\n- Active coping: 'trying to stay optimistic'\n- Realistic acknowledgment of concerns\n- Balanced perspective despite stress",
                explanation="Demonstrates handling complex mixed emotions with clear categorization and emotional processing recognition.",
                complexity_level=ComplexityLevel.MODERATE,
                note_type=NoteType.JOURNAL,
                analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
                output_format=OutputFormat.STRUCTURED_LIST
            ),
            MultiShotExample(
                example_input="Today marks exactly one year since I started this journey of self-improvement. Looking back, I can see how much I've grown, yet I'm also acutely aware of how far I still have to go. There's this strange paradox of feeling proud of my progress while simultaneously being frustrated with my current limitations. The meditation practice has definitely helped with emotional regulation, but I still find myself spiraling into self-doubt when facing new challenges. It's like I'm caught between two versions of myself - the one who's made tremendous strides and the one who's terrified of not being enough.",
                example_output="**Overall Sentiment: Complex Introspective (Growth-Oriented with Self-Doubt)**\n\n**Temporal Perspective Analysis:**\n- Reflective milestone recognition (one-year marker)\n- Dual temporal awareness: past progress vs. future gaps\n- Sophisticated self-assessment capabilities\n\n**Emotional Complexity Indicators:**\n- Paradoxical feelings: simultaneous pride and frustration\n- Meta-cognitive awareness: 'strange paradox' recognition\n- Identity tension: 'caught between two versions of myself'\n\n**Growth Mindset Markers:**\n- Progress acknowledgment: 'how much I've grown'\n- Skill development recognition: 'meditation practice has helped'\n- Realistic goal-setting: 'aware of how far I still have to go'\n\n**Vulnerability and Challenge Areas:**\n- Persistent patterns: 'spiraling into self-doubt'\n- Performance anxiety: 'terrified of not being enough'\n- Situational triggers: 'when facing new challenges'\n\n**Psychological Sophistication:**\n- Advanced emotional vocabulary and self-analysis\n- Recognition of internal contradictions\n- Balanced perspective on personal development journey",
                explanation="Complex psychological analysis showing sophisticated emotional processing, temporal perspective, and growth mindset assessment.",
                complexity_level=ComplexityLevel.COMPLEX,
                note_type=NoteType.JOURNAL,
                analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
                output_format=OutputFormat.STRUCTURED_LIST
            ),
            MultiShotExample(
                example_input="Terrible day. Everything went wrong from the moment I woke up. Overslept, missed the meeting, spilled coffee on my laptop. Feel like giving up.",
                example_output="**Overall Sentiment: Strongly Negative (Crisis Response)**\n- Cascading failure pattern: multiple sequential problems\n- Overwhelm indicators: 'everything went wrong'\n- Defeat response: 'feel like giving up'\n- Acute stress markers: immediate, intense negative reaction\n- Temporal focus: crisis-oriented, present-moment distress",
                explanation="Shows how to analyze acute negative sentiment with crisis patterns and overwhelm indicators.",
                complexity_level=ComplexityLevel.SIMPLE,
                note_type=NoteType.JOURNAL,
                analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
                output_format=OutputFormat.STRUCTURED_LIST
            ),
            MultiShotExample(
                example_input="I've been thinking a lot about Dad's advice before he passed - 'Don't let perfect be the enemy of good.' It's taken me three years to really understand what he meant. Today, while working on this project that I've been paralyzed by for weeks, I finally just started. Not perfectly, not with all the answers, just started. And you know what? It felt like a small victory against the perfectionist demon that's been holding me back since childhood. There's something profound about honoring his memory by actually living his wisdom rather than just remembering his words. I think he'd be proud of this messy, imperfect progress.",
                example_output="**Overall Sentiment: Profound Positive (Grief-Integrated Growth)**\n\n**Grief Processing Sophistication:**\n- Temporal integration: three-year meaning-making journey\n- Wisdom inheritance: transforming advice into lived experience\n- Memorial honoring: 'honoring his memory by actually living his wisdom'\n- Healthy attachment: maintaining connection while moving forward\n\n**Personal Growth Trajectory:**\n- Pattern recognition: identifying 'perfectionist demon' from 'childhood'\n- Behavioral breakthrough: moving from paralysis to action\n- Cognitive reframing: 'small victory' perspective on imperfect action\n- Self-compassion development: accepting 'messy, imperfect progress'\n\n**Emotional Sophistication:**\n- Integrated processing: grief + growth + pride\n- Generational wisdom: connecting past guidance to present action\n- Identity evolution: shifting from perfectionist paralysis to growth-oriented action\n- Meaning-making capacity: finding profound significance in simple actions\n\n**Motivational Psychology:**\n- Intrinsic motivation: personal meaning rather than external validation\n- Legacy consciousness: imagining parental approval as growth metric\n- Present-moment application: translating reflection into immediate action",
                explanation="Expert-level analysis showing grief integration, generational wisdom processing, and sophisticated personal growth patterns.",
                complexity_level=ComplexityLevel.EXPERT,
                note_type=NoteType.JOURNAL,
                analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
                output_format=OutputFormat.STRUCTURED_LIST
            )
        ]

        repository[(AnalysisTask.ANALYZE_SENTIMENT, NoteType.JOURNAL)] = journal_sentiment_examples

        # Action Items Examples for Meeting Notes
        meeting_action_examples = [
            MultiShotExample(
                example_input="Team standup: John will update the documentation. Sarah needs to fix the login bug. Meeting ends.",
                example_output="**Action Items:**\n• John: Update documentation (Owner: John, Priority: Standard, Due: Not specified)\n• Sarah: Fix login bug (Owner: Sarah, Priority: Standard, Due: Not specified)",
                explanation="Basic action item extraction with clear ownership identification.",
                complexity_level=ComplexityLevel.SIMPLE,
                note_type=NoteType.MEETING,
                analysis_task=AnalysisTask.FIND_ACTION_ITEMS,
                output_format=OutputFormat.BULLET_POINTS
            ),
            MultiShotExample(
                example_input="Q3 Planning Meeting - Marketing & Engineering Sync\nAttendees: Lisa (Marketing Dir), Mike (Eng Lead), Sarah (PM), Tom (Designer)\n\nLisa: We need the new feature demo ready for the client presentation next Tuesday. This is critical for closing the Q4 deal.\nMike: I can have a working prototype by Monday, but we'll need Tom's latest designs integrated.\nTom: Designs are 90% complete. I'll finish by Thursday and hand off to Mike.\nSarah: I'll coordinate the demo logistics and prepare backup materials in case of technical issues.\nLisa: Also, we should prepare a competitive analysis document. Sarah, can you own this?\nSarah: Yes, I'll have it ready by Friday.\nMike: One concern - the server capacity might not handle demo load. Should I provision additional resources?\nLisa: Yes, definitely. Budget approved. Tom, please also prepare static mockups as backup.\nDecision: If prototype isn't stable by Monday afternoon, we'll use static demo with Tom's mockups.",
                example_output="**Critical Actions (Next Tuesday Deadline):**\n• Mike: Complete working prototype (Due: Monday, Priority: Critical, Dependencies: Tom's designs)\n• Tom: Finish design completion and handoff (Due: Thursday, Priority: High, Blocks: Mike's prototype)\n• Mike: Provision additional server resources for demo (Due: Before Monday, Priority: High, Budget: Pre-approved)\n• Tom: Prepare static mockups as backup (Due: Monday, Priority: Medium, Contingency planning)\n\n**Supporting Actions:**\n• Sarah: Coordinate demo logistics and prepare backup materials (Due: Next Tuesday, Priority: High)\n• Sarah: Create competitive analysis document (Due: Friday, Priority: Medium)\n\n**Contingency Decision:**\n• If prototype unstable by Monday afternoon → Switch to static demo with mockups (Decision maker: Team consensus)",
                explanation="Complex action extraction showing priority levels, dependencies, contingency planning, and budget considerations.",
                complexity_level=ComplexityLevel.MODERATE,
                note_type=NoteType.MEETING,
                analysis_task=AnalysisTask.FIND_ACTION_ITEMS,
                output_format=OutputFormat.BULLET_POINTS
            ),
            MultiShotExample(
                example_input="Board Meeting - Strategic Planning Session\nBoard Members: Chairman Wilson, CEO Martinez, CFO Chen, External Directors: Dr. Patel, Ms. Johnson\nExecutive Team: CTO Kumar, CMO Thompson, COO Davis\n\nChairman Wilson opens with concerns about market position deterioration in Q3. CEO Martinez presents turnaround strategy focusing on three pillars: product innovation acceleration, market expansion into EMEA, and operational efficiency improvements.\n\nDr. Patel questions the feasibility of simultaneous expansion while cutting costs. CFO Chen confirms $2.3M budget allocation for EMEA entry, contingent on achieving 15% cost reduction in operations by Q1.\n\nCOO Davis proposes phased approach: Phase 1 - operational optimization (Q4), Phase 2 - EMEA market research and partnership development (Q1), Phase 3 - market entry execution (Q2). Ms. Johnson suggests engaging McKinsey for operational audit.\n\nCTO Kumar raises technical debt concerns - estimates $1.8M investment needed for infrastructure modernization to support expansion. CEO Martinez approves preliminary budget pending detailed technical assessment.\n\nCMO Thompson requests approval for $500K marketing research budget for EMEA. Board approves with requirement for monthly progress reports.\n\nChairman Wilson mandates formation of Expansion Oversight Committee including Dr. Patel (Chair), CFO Chen, and COO Davis. First committee meeting scheduled within two weeks.\n\nDecision: Proceed with phased expansion plan contingent on Q4 operational targets. All department heads to submit detailed implementation timelines by month-end.",
                example_output="**Executive Leadership Actions:**\n• CEO Martinez: Finalize turnaround strategy documentation and communication plan (Due: Within 1 week, Priority: Critical)\n• CFO Chen: Structure $2.3M EMEA budget with 15% cost reduction contingencies (Due: Q1, Priority: Critical, Dependency: Operational targets)\n• COO Davis: Execute Phase 1 operational optimization program (Due: Q4, Priority: Critical, Success metric: 15% cost reduction)\n• CTO Kumar: Complete detailed technical infrastructure assessment (Due: 3 weeks, Priority: High, Budget impact: $1.8M)\n• CMO Thompson: Execute EMEA marketing research program (Due: Ongoing Q1, Priority: High, Budget: $500K approved)\n\n**Board Governance Actions:**\n• Dr. Patel: Chair Expansion Oversight Committee and schedule inaugural meeting (Due: Within 2 weeks, Priority: High)\n• Dr. Patel, CFO Chen, COO Davis: Form Expansion Oversight Committee structure and charter (Due: 2 weeks, Priority: High)\n• Chairman Wilson: Oversee committee formation and strategic alignment (Due: 2 weeks, Priority: Medium)\n\n**All Department Heads:**\n• Submit detailed implementation timelines for phased expansion plan (Due: Month-end, Priority: Critical, Scope: All departments)\n\n**Conditional Actions (Pending Board Decisions):**\n• Ms. Johnson's McKinsey engagement proposal: Pending operational audit approval (Status: Under consideration)\n• CTO Kumar's $1.8M infrastructure investment: Pending technical assessment completion (Status: Preliminary approval)\n\n**Reporting Requirements:**\n• CMO Thompson: Monthly EMEA research progress reports to board (Ongoing, Starting Q1)\n• All Committee Members: Regular Expansion Oversight Committee reporting (Frequency: TBD at first meeting)\n\n**Strategic Contingencies:**\n• Entire EMEA expansion contingent on Q4 operational optimization success (15% cost reduction target)\n• Phase 2 and 3 execution dependent on Phase 1 completion and board review",
                explanation="Expert-level extraction showing multi-stakeholder governance, conditional dependencies, budget approvals, and strategic contingency planning.",
                complexity_level=ComplexityLevel.EXPERT,
                note_type=NoteType.MEETING,
                analysis_task=AnalysisTask.FIND_ACTION_ITEMS,
                output_format=OutputFormat.BULLET_POINTS
            )
        ]

        repository[(AnalysisTask.FIND_ACTION_ITEMS, NoteType.MEETING)] = meeting_action_examples

        # Key Points Examples for Research Notes
        research_keypoints_examples = [
            MultiShotExample(
                example_input="Study: The Impact of Sleep on Memory Consolidation\nResearchers found that people who sleep 8+ hours retain 40% more information than those who sleep less than 6 hours. REM sleep appears crucial for memory formation.",
                example_output="**Key Research Findings:**\n1. Sleep duration directly correlates with memory retention\n2. 8+ hours sleep → 40% better information retention vs <6 hours\n3. REM sleep phase is crucial for memory formation process",
                explanation="Simple research summary focusing on core quantitative findings and key mechanisms.",
                complexity_level=ComplexityLevel.SIMPLE,
                note_type=NoteType.RESEARCH,
                analysis_task=AnalysisTask.EXTRACT_KEY_POINTS,
                output_format=OutputFormat.STRUCTURED_LIST
            ),
            MultiShotExample(
                example_input="Longitudinal Study: Remote Work Impact on Productivity and Well-being (2020-2024)\n\nMethodology: 2,847 knowledge workers across 12 industries, quarterly surveys + productivity metrics\n\nProductivity Findings:\n- Individual productivity increased 13% on average\n- Team collaboration effectiveness decreased 8%\n- Creative problem-solving sessions 23% less effective remotely\n- Routine task completion improved 18%\n\nWell-being Results:\n- Work-life balance satisfaction up 31%\n- Professional isolation reported by 67% of participants\n- Career advancement anxiety increased 22%\n- Commute stress elimination saved avg 1.2 hours daily\n\nHybrid Model Performance:\n- 2-3 days remote optimal for productivity + collaboration balance\n- Companies with structured hybrid policies showed 15% better retention\n- Informal learning opportunities decreased 34% in full-remote settings",
                example_output="**Methodology Significance:**\n1. Robust sample: 2,847 knowledge workers across 12 industries (4-year longitudinal design)\n2. Multi-metric approach: subjective surveys + objective productivity data\n3. Comprehensive scope: individual, team, and organizational levels\n\n**Productivity Key Findings:**\n4. Individual work improvement: +13% average productivity gain\n5. Collaboration trade-offs: -8% team effectiveness, -23% creative problem-solving\n6. Task-type differentiation: +18% routine tasks vs decreased collaborative creativity\n\n**Well-being Critical Insights:**\n7. Work-life balance paradox: +31% satisfaction but +22% career advancement anxiety\n8. Social isolation epidemic: 67% report professional isolation despite productivity gains\n9. Time recapture value: 1.2 hours daily saved from commute elimination\n\n**Hybrid Model Optimization:**\n10. Sweet spot identification: 2-3 remote days optimal for balanced outcomes\n11. Policy structure importance: 15% better retention with structured hybrid approaches\n12. Learning opportunity gap: -34% informal learning in full-remote environments\n\n**Strategic Implications:**\n13. Task-based work design: optimize location based on work type requirements\n14. Intentional collaboration design: structured approaches needed for remote creative work\n15. Career development innovation: new models needed to address advancement anxiety",
                explanation="Comprehensive research analysis showing methodology assessment, multi-dimensional findings synthesis, and strategic implications extraction.",
                complexity_level=ComplexityLevel.COMPLEX,
                note_type=NoteType.RESEARCH,
                analysis_task=AnalysisTask.EXTRACT_KEY_POINTS,
                output_format=OutputFormat.STRUCTURED_LIST
            )
        ]

        repository[(AnalysisTask.EXTRACT_KEY_POINTS, NoteType.RESEARCH)] = research_keypoints_examples

        # Theme Identification Examples for Project Notes
        project_theme_examples = [
            MultiShotExample(
                example_input="Sprint Retrospective: We delivered all features but had several bugs in production. Team communication was good. Need better testing process.",
                example_output="**Primary Themes:**\n• Delivery Success: All planned features completed on time\n• Quality Concerns: Production bugs indicate testing gaps\n• Process Improvement: Need for enhanced testing procedures\n• Team Dynamics: Positive communication noted",
                explanation="Basic theme identification from straightforward project content.",
                complexity_level=ComplexityLevel.SIMPLE,
                note_type=NoteType.PROJECT,
                analysis_task=AnalysisTask.IDENTIFY_THEMES,
                output_format=OutputFormat.BULLET_POINTS
            ),
            MultiShotExample(
                example_input="VaultMind Development Diary - Week 24\n\nThis week revealed interesting tensions between our technical ambitions and practical constraints. The team is incredibly capable and motivated, but we're hitting some hard truths about scope management. The AI integration is more complex than anticipated - not just technically, but in terms of user experience design.\n\nDavid's prototype for the prompt engineering system shows real promise, but it's clear we need to make some tough decisions about feature prioritization. Do we go deep on a few advanced features or broad on basic functionality?\n\nUser feedback from our alpha testers has been surprisingly positive about the core concept, but they're asking for integrations we hadn't planned - Notion, Roam, other knowledge management tools. This validates our market hypothesis but also expands our development scope significantly.\n\nThe technical debt from our rapid prototyping phase is starting to show. We need to balance new feature development with code cleanup and testing infrastructure. Sarah's right that we can't keep building on shaky foundations.\n\nFinancially, we're in a good position for another 8 months, but that timeline assumes we can ship a beta by month 6. The pressure is mounting, but it's productive pressure - everyone feels the urgency but remains confident in our direction.",
                example_output="**Strategic Tensions:**\n• Ambition vs. Constraints: Technical capabilities exceeding practical execution capacity\n• Feature Strategy Dilemma: Depth vs. breadth decision point for product development\n• Scope Expansion Pressure: User demand driving unplanned feature requirements\n\n**Technical Development Themes:**\n• AI Integration Complexity: Beyond technical challenges to UX design complexity\n• Prototype Promise: David's prompt engineering breakthrough showing potential\n• Technical Debt Recognition: Rapid prototyping legacy requiring attention\n• Infrastructure Investment: Need for testing and code quality foundations\n\n**Market Validation Patterns:**\n• Core Concept Validation: Alpha tester positive response to fundamental approach\n• Integration Demand: User requests for ecosystem connectivity (Notion, Roam)\n• Market Hypothesis Confirmation: Demand exists but with expanded expectations\n\n**Team Dynamics & Leadership:**\n• Collective Capability: High team skill and motivation levels\n• Collaborative Decision-Making: Distributed ownership (David's prototyping, Sarah's infrastructure wisdom)\n• Productive Pressure: Urgency creating focus rather than stress\n\n**Business Sustainability:**\n• Financial Runway: 8-month buffer with conditional beta timeline\n• Timeline Dependencies: Month 6 beta ship date as critical milestone\n• Confidence Maintenance: Team belief in direction despite mounting pressures",
                explanation="Multi-dimensional theme analysis showing strategic, technical, market, and organizational patterns with interconnected relationships.",
                complexity_level=ComplexityLevel.MODERATE,
                note_type=NoteType.PROJECT,
                analysis_task=AnalysisTask.IDENTIFY_THEMES,
                output_format=OutputFormat.BULLET_POINTS
            )
        ]

        repository[(AnalysisTask.IDENTIFY_THEMES, NoteType.PROJECT)] = project_theme_examples

        return repository

    def _initialize_task_descriptions(self) -> Dict[AnalysisTask, str]:
        """Initialize comprehensive task descriptions."""
        return {
            AnalysisTask.EXTRACT_KEY_POINTS: "Extract the most important points, insights, and information from the content, identifying critical details and organizing them hierarchically by significance",
            AnalysisTask.IDENTIFY_THEMES: "Identify recurring themes, conceptual patterns, and underlying motifs within the content, showing how different themes interconnect and evolve",
            AnalysisTask.SUMMARIZE_CONTENT: "Create a comprehensive yet concise summary that captures essential meaning, important details, and key takeaways while maintaining the original context",
            AnalysisTask.FIND_ACTION_ITEMS: "Identify actionable tasks, decisions to be made, deadlines, responsibilities, and follow-up items, including dependencies and priority levels",
            AnalysisTask.ANALYZE_SENTIMENT: "Analyze emotional tone, attitudes, sentiment progression, and psychological patterns expressed in the content, including complexity and nuance",
            AnalysisTask.CATEGORIZE_TOPICS: "Categorize content into relevant topic areas, subject domains, and thematic categories for improved organization and knowledge management",
            AnalysisTask.EXTRACT_ENTITIES: "Extract important entities including people, places, dates, organizations, concepts, tools, and resources mentioned in the content",
            AnalysisTask.IDENTIFY_PATTERNS: "Identify recurring patterns, relationships, trends, and structural elements within the content that reveal deeper insights",
            AnalysisTask.CREATE_CONNECTIONS: "Identify and create meaningful connections between ideas, concepts, and themes, showing relationships and potential knowledge links"
        }

    def _initialize_note_contexts(self) -> Dict[NoteType, str]:
        """Initialize detailed note-type contexts."""
        return {
            NoteType.JOURNAL: "Personal journal entries containing thoughts, reflections, experiences, emotions, and personal development insights. Focus on emotional intelligence, growth patterns, and self-awareness.",
            NoteType.RESEARCH: "Academic or investigative research content with findings, methodologies, hypotheses, data analysis, and scholarly information. Maintain scientific rigor and evidence-based analysis.",
            NoteType.MEETING: "Meeting documentation including discussions, decisions, action items, participant interactions, and organizational dynamics. Emphasize actionable outcomes and clear accountability.",
            NoteType.PROJECT: "Project-related content with planning, progress updates, challenges, strategic decisions, and team dynamics. Focus on deliverables, timelines, and project success factors.",
            NoteType.IDEA: "Creative ideation and brainstorming content with innovative concepts, problem-solving approaches, and creative exploration. Preserve creative potential while adding analytical structure.",
            NoteType.REFERENCE: "Reference material with factual information, procedures, best practices, documentation, and knowledge resources. Maintain accuracy and clear categorization.",
            NoteType.PERSONAL: "Personal content including private thoughts, life planning, goals, relationships, and individual development. Handle with sensitivity while providing meaningful insights.",
            NoteType.ACADEMIC: "Educational content with learning materials, study notes, course work, and academic progress. Structure for effective learning, retention, and knowledge application."
        }

    def _initialize_selection_strategies(self) -> Dict[str, str]:
        """Initialize example selection strategies."""
        return {
            "progressive": "Select examples that demonstrate increasing complexity levels to show pattern evolution",
            "diverse": "Select examples that show maximum variation in approaches and styles",
            "contextual": "Select examples most similar to the target content's context and complexity",
            "comprehensive": "Select examples that cover all major variations and edge cases",
            "balanced": "Select examples that balance complexity levels and approach diversity"
        }

    def analyze_note(self, note: NoteContent, task: AnalysisTask,
                    output_format: OutputFormat = OutputFormat.STRUCTURED_LIST,
                    selection_strategy: str = "progressive",
                    max_examples: int = 4,
                    custom_constraints: Optional[List[str]] = None) -> MultiShotPrompt:
        """
        Analyze a note using multi-shot prompting with diverse examples.

        Args:
            note: The note content to analyze
            task: The type of analysis to perform
            output_format: How to structure the output
            selection_strategy: Strategy for selecting examples
            max_examples: Maximum number of examples to include (3-5 recommended)
            custom_constraints: Additional constraints for the analysis

        Returns:
            MultiShotPrompt: A complete prompt with multiple examples
        """
        # Get available examples for this task and note type
        examples = self._get_examples_for_task(task, note.note_type)

        # Select examples based on strategy
        selected_examples = self._select_examples(
            examples, note, selection_strategy, max_examples
        )

        # Get task description
        task_description = self.task_descriptions.get(task, "Analyze the content as specified")

        # Build comprehensive context
        context_parts = [
            task_description,
            "You will learn from multiple examples showing different approaches and complexity levels."
        ]

        if note.note_type and note.note_type in self.note_type_contexts:
            context_parts.append(self.note_type_contexts[note.note_type])

        # Add note metadata context
        if note.tags:
            context_parts.append(f"Target note tags: {', '.join(note.tags)}")

        if note.created_date:
            context_parts.append(f"Created: {note.created_date}")

        # Estimate target complexity and mention it
        target_complexity = note.complexity_level or note.estimate_complexity()
        context_parts.append(f"Target content appears to be {target_complexity.value} complexity level.")

        # Build output format instructions
        output_instructions = self._get_output_format_instructions(output_format)

        # Comprehensive constraints
        constraints = [
            "Learn from the pattern variations shown in the examples",
            "Match the complexity level appropriate for the target content",
            "Apply the most relevant techniques demonstrated in the examples",
            "Maintain consistency with the demonstrated quality standards",
            "Adapt the approaches to fit the specific content characteristics"
        ]

        if custom_constraints:
            constraints.extend(custom_constraints)

        return MultiShotPrompt(
            task_description=task_description,
            context_instructions=". ".join(context_parts),
            examples=selected_examples,
            target_content=note.content,
            output_format=output_instructions,
            constraints=constraints,
            example_selection_strategy=selection_strategy
        )

    def _get_examples_for_task(self, task: AnalysisTask, note_type: Optional[NoteType]) -> List[MultiShotExample]:
        """Get all available examples for a specific task."""
        examples = []

        # First try to find examples for exact note type match
        if note_type:
            key = (task, note_type)
            if key in self.example_repository:
                examples.extend(self.example_repository[key])

        # Then add examples from other note types for the same task
        for (repo_task, repo_note_type), repo_examples in self.example_repository.items():
            if repo_task == task and (not note_type or repo_note_type != note_type):
                examples.extend(repo_examples)

        return examples

    def _select_examples(self, available_examples: List[MultiShotExample],
                        target_note: NoteContent, strategy: str,
                        max_examples: int) -> List[MultiShotExample]:
        """Select examples based on the specified strategy."""
        if not available_examples:
            return []

        target_complexity = target_note.complexity_level or target_note.estimate_complexity()

        if strategy == "progressive":
            return self._select_progressive_examples(available_examples, target_complexity, max_examples)
        elif strategy == "diverse":
            return self._select_diverse_examples(available_examples, max_examples)
        elif strategy == "contextual":
            return self._select_contextual_examples(available_examples, target_note, max_examples)
        elif strategy == "comprehensive":
            return self._select_comprehensive_examples(available_examples, max_examples)
        else:  # balanced
            return self._select_balanced_examples(available_examples, target_complexity, max_examples)

    def _select_progressive_examples(self, examples: List[MultiShotExample],
                                   target_complexity: ComplexityLevel,
                                   max_examples: int) -> List[MultiShotExample]:
        """Select examples showing progressive complexity leading to target level."""
        # Sort by complexity level
        complexity_order = [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE,
                          ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT]

        target_index = complexity_order.index(target_complexity)
        selected = []

        # Include examples up to and including target complexity
        for complexity in complexity_order[:target_index + 1]:
            complexity_examples = [ex for ex in examples if ex.complexity_level == complexity]
            if complexity_examples:
                selected.append(complexity_examples[0])  # Take first example of each level

        # If we need more examples and target is complex/expert, add one higher level
        if len(selected) < max_examples and target_index < len(complexity_order) - 1:
            higher_complexity = complexity_order[target_index + 1]
            higher_examples = [ex for ex in examples if ex.complexity_level == higher_complexity]
            if higher_examples:
                selected.append(higher_examples[0])

        return selected[:max_examples]

    def _select_diverse_examples(self, examples: List[MultiShotExample],
                               max_examples: int) -> List[MultiShotExample]:
        """Select examples with maximum diversity in approaches."""
        if len(examples) <= max_examples:
            return examples

        # Group by note type and complexity, then select one from each group
        groups = {}
        for ex in examples:
            key = (ex.note_type, ex.complexity_level)
            if key not in groups:
                groups[key] = []
            groups[key].append(ex)

        selected = []
        for group_examples in groups.values():
            selected.append(group_examples[0])
            if len(selected) >= max_examples:
                break

        return selected[:max_examples]

    def _select_contextual_examples(self, examples: List[MultiShotExample],
                                  target_note: NoteContent,
                                  max_examples: int) -> List[MultiShotExample]:
        """Select examples most similar to target content."""
        target_complexity = target_note.complexity_level or target_note.estimate_complexity()

        # Score examples based on similarity to target
        scored_examples = []
        for ex in examples:
            score = 0
            # Same note type gets high score
            if ex.note_type == target_note.note_type:
                score += 3
            # Same complexity level gets high score
            if ex.complexity_level == target_complexity:
                score += 2
            # Similar tags get bonus points
            if target_note.tags:
                common_tags = set(ex.tags) & set(target_note.tags)
                score += len(common_tags)

            scored_examples.append((score, ex))

        # Sort by score and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_examples[:max_examples]]

    def _select_comprehensive_examples(self, examples: List[MultiShotExample],
                                     max_examples: int) -> List[MultiShotExample]:
        """Select examples covering all major variations."""
        if len(examples) <= max_examples:
            return examples

        # Ensure we cover all complexity levels if possible
        complexity_levels = list(set(ex.complexity_level for ex in examples))
        selected = []

        # Take one example from each complexity level
        for complexity in complexity_levels:
            complexity_examples = [ex for ex in examples if ex.complexity_level == complexity]
            selected.append(complexity_examples[0])
            if len(selected) >= max_examples:
                break

        # Fill remaining slots with diverse examples
        remaining = max_examples - len(selected)
        if remaining > 0:
            remaining_examples = [ex for ex in examples if ex not in selected]
            selected.extend(remaining_examples[:remaining])

        return selected

    def _select_balanced_examples(self, examples: List[MultiShotExample],
                                target_complexity: ComplexityLevel,
                                max_examples: int) -> List[MultiShotExample]:
        """Select examples balancing complexity and diversity."""
        # Combine progressive and diverse strategies
        progressive = self._select_progressive_examples(examples, target_complexity, max_examples // 2 + 1)
        diverse = self._select_diverse_examples(examples, max_examples)

        # Merge without duplicates
        selected = progressive.copy()
        for ex in diverse:
            if ex not in selected and len(selected) < max_examples:
                selected.append(ex)

        return selected[:max_examples]

    def _get_output_format_instructions(self, output_format: OutputFormat) -> str:
        """Get detailed output format instructions."""
        format_instructions = {
            OutputFormat.JSON: "Structure as valid JSON with clear field names, proper data types, and logical hierarchy",
            OutputFormat.MARKDOWN: "Use proper Markdown formatting with headers, lists, emphasis, and code blocks where appropriate",
            OutputFormat.BULLET_POINTS: "Use bullet points with clear hierarchy, consistent formatting, and logical grouping",
            OutputFormat.STRUCTURED_LIST: "Use numbered lists and sublists with clear organization, hierarchy, and detailed explanations",
            OutputFormat.NARRATIVE: "Present as flowing narrative text with logical paragraph structure and smooth transitions",
            OutputFormat.TABLE: "Organize information in clear table format with appropriate columns, rows, and headers"
        }
        return format_instructions.get(output_format, "Follow the format demonstrated in the examples")

    def get_example_statistics(self) -> Dict[str, Any]:
        """Get statistics about the example repository."""
        stats = {
            "total_examples": 0,
            "by_task": {},
            "by_note_type": {},
            "by_complexity": {},
            "by_output_format": {}
        }

        for examples in self.example_repository.values():
            stats["total_examples"] += len(examples)
            for ex in examples:
                # Count by task
                task_name = ex.analysis_task.value
                stats["by_task"][task_name] = stats["by_task"].get(task_name, 0) + 1

                # Count by note type
                note_type_name = ex.note_type.value
                stats["by_note_type"][note_type_name] = stats["by_note_type"].get(note_type_name, 0) + 1

                # Count by complexity
                complexity_name = ex.complexity_level.value
                stats["by_complexity"][complexity_name] = stats["by_complexity"].get(complexity_name, 0) + 1

                # Count by output format
                format_name = ex.output_format.value
                stats["by_output_format"][format_name] = stats["by_output_format"].get(format_name, 0) + 1

        return stats

    def add_custom_example(self, example: MultiShotExample) -> None:
        """Add a custom example to the repository."""
        key = (example.analysis_task, example.note_type)
        if key not in self.example_repository:
            self.example_repository[key] = []
        self.example_repository[key].append(example)

    def batch_analyze(self, notes: List[NoteContent], task: AnalysisTask,
                     output_format: OutputFormat = OutputFormat.STRUCTURED_LIST,
                     selection_strategy: str = "progressive") -> List[MultiShotPrompt]:
        """Analyze multiple notes using multi-shot prompting."""
        return [
            self.analyze_note(note, task, output_format, selection_strategy)
            for note in notes
        ]

    def compare_strategies(self, note: NoteContent, task: AnalysisTask) -> Dict[str, MultiShotPrompt]:
        """Compare different example selection strategies for the same note."""
        strategies = ["progressive", "diverse", "contextual", "comprehensive", "balanced"]
        return {
            strategy: self.analyze_note(note, task, selection_strategy=strategy)
            for strategy in strategies
        }


# Convenience functions for common use cases
def analyze_with_examples(content: str, note_type: NoteType, task: AnalysisTask,
                         max_examples: int = 4, title: str = "Note") -> MultiShotPrompt:
    """Quick function to analyze content with multiple examples."""
    note = NoteContent(title=title, content=content, note_type=note_type)
    analyzer = MultiShotAnalyzer()
    return analyzer.analyze_note(note, task, max_examples=max_examples)


def analyze_journal_multishot(content: str, title: str = "Journal Entry") -> MultiShotPrompt:
    """Analyze journal sentiment with multiple examples."""
    return analyze_with_examples(
        content, NoteType.JOURNAL, AnalysisTask.ANALYZE_SENTIMENT,
        max_examples=4, title=title
    )


def analyze_meeting_multishot(content: str, title: str = "Meeting Notes") -> MultiShotPrompt:
    """Extract meeting action items with multiple examples."""
    return analyze_with_examples(
        content, NoteType.MEETING, AnalysisTask.FIND_ACTION_ITEMS,
        max_examples=3, title=title
    )


def analyze_research_multishot(content: str, title: str = "Research Notes") -> MultiShotPrompt:
    """Extract research key points with multiple examples."""
    return analyze_with_examples(
        content, NoteType.RESEARCH, AnalysisTask.EXTRACT_KEY_POINTS,
        max_examples=4, title=title
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Multi-Shot Prompting Demo ===\n")

    # Sample note for demonstration
    sample_note = NoteContent(
        title="Personal Development Journey - 6 Month Review",
        content="""
        Six months into this intentional growth journey, I'm seeing patterns I hadn't expected. 
        The meditation practice has become almost automatic - 20 minutes each morning feels natural now, 
        though I still struggle with evening sessions when I'm tired.
        
        Professionally, I've been taking on more complex projects and actually enjoying the challenge 
        rather than feeling overwhelmed. The imposter syndrome hasn't disappeared, but it's quieter. 
        I notice it more as a background hum than the loud alarm it used to be.
        
        Relationships have shifted too. I'm setting boundaries more clearly with family, which has 
        created some tension but also more authentic connections. Sarah mentioned last week that I 
        seem more present in our conversations.
        
        The biggest surprise has been how much energy I have for creative projects. Started writing 
        again after a 3-year hiatus. Nothing groundbreaking, but the act of creating feels vital again.
        
        Areas for growth: Still procrastinating on financial planning. Need to address the tendency 
        to say yes to everything. And I'm realizing that self-compassion is harder work than 
        self-discipline ever was.
        
        Overall trajectory feels positive and sustainable. The changes feel integrated rather than forced.
        """,
        note_type=NoteType.JOURNAL,
        created_date="2025-08-26",
        tags=["personal-development", "reflection", "growth"],
        complexity_level=ComplexityLevel.COMPLEX
    )

    # Initialize analyzer
    analyzer = MultiShotAnalyzer()

    # Demo 1: Multi-shot sentiment analysis with progressive examples
    print("1. Multi-Shot Sentiment Analysis (Progressive Strategy):")
    sentiment_prompt = analyzer.analyze_note(
        sample_note,
        AnalysisTask.ANALYZE_SENTIMENT,
        OutputFormat.STRUCTURED_LIST,
        "progressive",
        max_examples=4
    )

    example_summary = sentiment_prompt.get_example_summary()
    print(f"Examples used: {example_summary['total_examples']}")
    print(f"Complexity progression: {' → '.join(example_summary['complexity_levels'])}")
    print(f"Selection strategy: {example_summary['selection_strategy']}")
    print(f"Total prompt length: {example_summary['prompt_length']:,} characters\n")

    # Demo 2: Strategy comparison
    print("2. Comparing Example Selection Strategies:")
    strategy_comparison = analyzer.compare_strategies(sample_note, AnalysisTask.ANALYZE_SENTIMENT)

    for strategy, prompt in strategy_comparison.items():
        summary = prompt.get_example_summary()
        print(f"   {strategy}: {summary['total_examples']} examples, "
              f"{summary['prompt_length']:,} chars, "
              f"levels: {' + '.join(set(summary['complexity_levels']))}")

    print()

    # Demo 3: Repository statistics
    print("3. Example Repository Statistics:")
    stats = analyzer.get_example_statistics()
    print(f"Total examples: {stats['total_examples']}")
    print(f"Tasks covered: {list(stats['by_task'].keys())}")
    print(f"Complexity distribution: {stats['by_complexity']}")
    print(f"Note type coverage: {list(stats['by_note_type'].keys())}\n")

    # Demo 4: Custom example addition
    print("4. Custom Example Capability:")
    print(f"Before adding: {stats['total_examples']} examples")

    custom_example = MultiShotExample(
        example_input="Custom analysis example input",
        example_output="Custom analysis example output",
        explanation="This demonstrates adding domain-specific examples",
        complexity_level=ComplexityLevel.MODERATE,
        note_type=NoteType.PERSONAL,
        analysis_task=AnalysisTask.ANALYZE_SENTIMENT,
        output_format=OutputFormat.NARRATIVE
    )

    analyzer.add_custom_example(custom_example)
    new_stats = analyzer.get_example_statistics()
    print(f"After adding: {new_stats['total_examples']} examples")

    # Demo 5: Complexity estimation
    print(f"\n5. Automatic Complexity Estimation:")
    estimated_complexity = sample_note.estimate_complexity()
    print(f"Sample note estimated complexity: {estimated_complexity.value}")
    print(f"Word count: {sample_note.get_word_count()}")
    print(f"Note preview: {sample_note.get_preview(30)}")

    print("\n=== Multi-Shot Prompting System Ready ===")
    print("Advantages over zero-shot and one-shot:")
    print("• Shows pattern variations and edge cases")
    print("• Demonstrates complexity progression")
    print("• Provides multiple valid approaches")
    print("• Handles nuanced analysis requirements")
    print("• Adapts to content complexity levels")
