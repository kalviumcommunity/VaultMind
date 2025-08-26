"""
VaultMind Chain-of-Thought Prompting Implementation

This module implements chain-of-thought (CoT) prompting for complex vault analysis,
breaking down reasoning into explicit, step-by-step processes that improve accuracy
and provide transparent decision-making paths.

Chain-of-thought prompting excels at complex reasoning tasks by encouraging the AI
to "think out loud" through multi-step problems, leading to more accurate and
explainable analysis results.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import re
from collections import defaultdict


class ReasoningComplexity(Enum):
    """Levels of reasoning complexity for different analysis tasks."""
    SIMPLE = "simple"           # 2-3 reasoning steps
    MODERATE = "moderate"       # 4-6 reasoning steps
    COMPLEX = "complex"         # 7-10 reasoning steps
    EXPERT = "expert"           # 10+ reasoning steps with sub-analyses


class AnalysisTask(Enum):
    """Types of analysis tasks suitable for chain-of-thought reasoning."""
    CROSS_NOTE_SYNTHESIS = "cross_note_synthesis"
    PATTERN_EVOLUTION_TRACKING = "pattern_evolution_tracking"
    MULTI_PERSPECTIVE_ANALYSIS = "multi_perspective_analysis"
    CAUSAL_RELATIONSHIP_MAPPING = "causal_relationship_mapping"
    KNOWLEDGE_GAP_IDENTIFICATION = "knowledge_gap_identification"
    STRATEGIC_INSIGHT_GENERATION = "strategic_insight_generation"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    TEMPORAL_TREND_ANALYSIS = "temporal_trend_analysis"
    CONCEPT_HIERARCHY_BUILDING = "concept_hierarchy_building"
    DECISION_FRAMEWORK_CONSTRUCTION = "decision_framework_construction"


class ReasoningStep(Enum):
    """Types of reasoning steps in chain-of-thought analysis."""
    OBSERVATION = "observation"           # What do we see?
    HYPOTHESIS = "hypothesis"             # What might this mean?
    EVIDENCE_GATHERING = "evidence_gathering"  # What supports this?
    PATTERN_RECOGNITION = "pattern_recognition"  # What patterns emerge?
    COMPARISON = "comparison"             # How does this relate to other data?
    INFERENCE = "inference"               # What can we conclude?
    VALIDATION = "validation"             # Does this conclusion hold up?
    SYNTHESIS = "synthesis"               # How does this fit together?
    IMPLICATION = "implication"           # What does this mean for the user?
    RECOMMENDATION = "recommendation"     # What should be done about it?


class OutputFormat(Enum):
    """Output formats for chain-of-thought analysis."""
    STRUCTURED_REASONING = "structured_reasoning"
    NARRATIVE_FLOW = "narrative_flow"
    STEP_BY_STEP_BREAKDOWN = "step_by_step_breakdown"
    DECISION_TREE = "decision_tree"
    EVIDENCE_CONCLUSION = "evidence_conclusion"


@dataclass
class ReasoningStepData:
    """Represents a single step in the chain of thought process."""
    step_type: ReasoningStep
    step_number: int
    description: str
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0
    sub_steps: List['ReasoningStepData'] = field(default_factory=list)
    connections_to_previous: List[int] = field(default_factory=list)
    
    def format_step(self, depth: int = 0) -> str:
        """Format the reasoning step for display."""
        indent = "  " * depth
        step_text = f"{indent}**Step {self.step_number} ({self.step_type.value.title()})**: {self.description}"
        
        if self.reasoning:
            step_text += f"\n{indent}*Reasoning*: {self.reasoning}"
        
        if self.evidence:
            step_text += f"\n{indent}*Evidence*:"
            for evidence_item in self.evidence:
                step_text += f"\n{indent}  - {evidence_item}"
        
        if self.confidence > 0:
            step_text += f"\n{indent}*Confidence*: {self.confidence:.1%}"
        
        # Add sub-steps if they exist
        if self.sub_steps:
            step_text += f"\n{indent}*Sub-analysis*:"
            for sub_step in self.sub_steps:
                step_text += f"\n{sub_step.format_step(depth + 1)}"
        
        return step_text


@dataclass
class ChainOfThoughtPrompt:
    """Represents a complete chain-of-thought prompt with reasoning structure."""
    task_description: str
    reasoning_framework: str
    example_reasoning_chain: List[str] = field(default_factory=list)
    step_templates: Dict[ReasoningStep, str] = field(default_factory=dict)
    target_content: str = ""
    complexity_level: ReasoningComplexity = ReasoningComplexity.MODERATE
    output_format: OutputFormat = OutputFormat.STRUCTURED_REASONING
    
    def build_prompt(self) -> str:
        """Build the complete chain-of-thought prompt."""
        prompt_parts = [
            f"**Complex Analysis Task**: {self.task_description}",
            "",
            f"**Reasoning Framework**: {self.reasoning_framework}",
            ""
        ]
        
        # Add example reasoning chain if provided
        if self.example_reasoning_chain:
            prompt_parts.extend([
                "**Example of step-by-step reasoning:**",
                ""
            ])
            for i, example_step in enumerate(self.example_reasoning_chain, 1):
                prompt_parts.append(f"Step {i}: {example_step}")
            prompt_parts.append("")
        
        # Add step-by-step instructions
        prompt_parts.extend([
            "**Your task**: Apply this same systematic reasoning approach to analyze the following content.",
            "",
            "**Required reasoning process:**",
            "1. **Observe**: What key information do you notice?",
            "2. **Hypothesize**: What initial theories or patterns emerge?",
            "3. **Gather Evidence**: What specific details support your hypotheses?",
            "4. **Recognize Patterns**: What deeper patterns become clear?",
            "5. **Compare**: How does this relate to other information or contexts?",
            "6. **Infer**: What logical conclusions can you draw?",
            "7. **Validate**: Do your conclusions hold up to scrutiny?",
            "8. **Synthesize**: How do all pieces fit together?",
            "9. **Identify Implications**: What does this mean for the user?",
            "10. **Recommend**: What actionable insights or next steps emerge?",
            "",
            "**Content to analyze:**",
            self.target_content,
            "",
            "**Instructions:**",
            "- Think through each step explicitly",
            "- Show your reasoning process clearly",
            "- Build each step on the previous ones",
            "- Be specific about evidence and logic",
            "- Arrive at well-supported conclusions",
            "",
            "**Begin your step-by-step analysis:**"
        ])
        
        return "\n".join(prompt_parts)
    
    def get_complexity_guidance(self) -> str:
        """Get guidance text based on complexity level."""
        guidance_map = {
            ReasoningComplexity.SIMPLE: "Focus on 2-3 clear reasoning steps with direct logic",
            ReasoningComplexity.MODERATE: "Use 4-6 reasoning steps with evidence gathering and pattern recognition",
            ReasoningComplexity.COMPLEX: "Apply 7-10 detailed reasoning steps with sub-analyses and validation",
            ReasoningComplexity.EXPERT: "Conduct comprehensive multi-layered analysis with 10+ steps and extensive sub-reasoning"
        }
        return guidance_map.get(self.complexity_level, "Apply systematic step-by-step reasoning")


@dataclass
class AnalysisResult:
    """Represents the result of a chain-of-thought analysis."""
    reasoning_chain: List[ReasoningStepData] = field(default_factory=list)
    final_conclusion: str = ""
    confidence_score: float = 0.0
    key_insights: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    potential_limitations: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    def format_complete_analysis(self) -> str:
        """Format the complete analysis with reasoning chain."""
        sections = [
            "# Chain-of-Thought Analysis Result",
            "",
            "## Reasoning Process:"
        ]
        
        for step in self.reasoning_chain:
            sections.append(step.format_step())
            sections.append("")
        
        sections.extend([
            "## Final Conclusion:",
            self.final_conclusion,
            "",
            f"**Overall Confidence**: {self.confidence_score:.1%}",
            ""
        ])
        
        if self.key_insights:
            sections.extend([
                "## Key Insights:",
                ""
            ])
            for insight in self.key_insights:
                sections.append(f"- {insight}")
            sections.append("")
        
        if self.recommended_actions:
            sections.extend([
                "## Recommended Actions:",
                ""
            ])
            for action in self.recommended_actions:
                sections.append(f"- {action}")
        
        return "\n".join(sections)


class ChainOfThoughtAnalyzer:
    """
    Main class for chain-of-thought analysis of vault content.
    
    This analyzer breaks down complex reasoning tasks into explicit steps,
    improving accuracy and providing transparent decision-making processes.
    """
    
    def __init__(self):
        self.reasoning_templates = self._initialize_reasoning_templates()
        self.task_frameworks = self._initialize_task_frameworks()
        self.example_chains = self._initialize_example_chains()
        self.complexity_adaptations = self._initialize_complexity_adaptations()
    
    def _initialize_reasoning_templates(self) -> Dict[AnalysisTask, Dict[str, Any]]:
        """Initialize reasoning templates for different analysis tasks."""
        return {
            AnalysisTask.CROSS_NOTE_SYNTHESIS: {
                "description": "Synthesize insights across multiple notes to identify overarching themes and connections",
                "framework": "Multi-source analysis with pattern integration and cross-validation",
                "complexity": ReasoningComplexity.COMPLEX,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.PATTERN_RECOGNITION,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.EVIDENCE_GATHERING,
                    ReasoningStep.SYNTHESIS,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.IMPLICATION
                ]
            },
            
            AnalysisTask.PATTERN_EVOLUTION_TRACKING: {
                "description": "Track how patterns, themes, or concepts evolve over time across the vault",
                "framework": "Temporal analysis with trend identification and change point detection",
                "complexity": ReasoningComplexity.COMPLEX,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.PATTERN_RECOGNITION,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.EVIDENCE_GATHERING,
                    ReasoningStep.INFERENCE,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.IMPLICATION,
                    ReasoningStep.RECOMMENDATION
                ]
            },
            
            AnalysisTask.CAUSAL_RELATIONSHIP_MAPPING: {
                "description": "Identify and map causal relationships between concepts, events, or decisions",
                "framework": "Causal inference with evidence triangulation and logical validation",
                "complexity": ReasoningComplexity.EXPERT,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.HYPOTHESIS,
                    ReasoningStep.EVIDENCE_GATHERING,
                    ReasoningStep.PATTERN_RECOGNITION,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.INFERENCE,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.SYNTHESIS,
                    ReasoningStep.IMPLICATION,
                    ReasoningStep.RECOMMENDATION
                ]
            },
            
            AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION: {
                "description": "Identify gaps, inconsistencies, or missing elements in the knowledge base",
                "framework": "Gap analysis with systematic coverage evaluation and priority assessment",
                "complexity": ReasoningComplexity.MODERATE,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.PATTERN_RECOGNITION,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.INFERENCE,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.IMPLICATION,
                    ReasoningStep.RECOMMENDATION
                ]
            },
            
            AnalysisTask.STRATEGIC_INSIGHT_GENERATION: {
                "description": "Generate strategic insights and actionable recommendations from vault content",
                "framework": "Strategic analysis with insight extraction and actionability assessment",
                "complexity": ReasoningComplexity.EXPERT,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.PATTERN_RECOGNITION,
                    ReasoningStep.HYPOTHESIS,
                    ReasoningStep.EVIDENCE_GATHERING,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.INFERENCE,
                    ReasoningStep.SYNTHESIS,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.IMPLICATION,
                    ReasoningStep.RECOMMENDATION
                ]
            },
            
            AnalysisTask.CONTRADICTION_RESOLUTION: {
                "description": "Identify and resolve contradictions or conflicts within the vault content",
                "framework": "Conflict analysis with evidence weighing and resolution synthesis",
                "complexity": ReasoningComplexity.COMPLEX,
                "key_steps": [
                    ReasoningStep.OBSERVATION,
                    ReasoningStep.COMPARISON,
                    ReasoningStep.EVIDENCE_GATHERING,
                    ReasoningStep.HYPOTHESIS,
                    ReasoningStep.VALIDATION,
                    ReasoningStep.INFERENCE,
                    ReasoningStep.SYNTHESIS,
                    ReasoningStep.RECOMMENDATION
                ]
            }
        }
    
    def _initialize_task_frameworks(self) -> Dict[AnalysisTask, str]:
        """Initialize detailed frameworks for each analysis task."""
        return {
            AnalysisTask.CROSS_NOTE_SYNTHESIS: """
Multi-Note Synthesis Framework:
1. Inventory all relevant notes and their key themes
2. Identify recurring patterns across different notes
3. Map connections and relationships between concepts
4. Gather supporting evidence for each connection
5. Synthesize insights that emerge from the combination
6. Validate synthesis against individual note contexts
7. Extract implications and actionable insights
            """.strip(),
            
            AnalysisTask.PATTERN_EVOLUTION_TRACKING: """
Pattern Evolution Framework:
1. Establish baseline patterns from early content
2. Track pattern manifestations across time periods
3. Identify inflection points and change moments
4. Compare pattern strength and characteristics over time
5. Analyze factors that may have influenced changes
6. Validate evolution hypothesis with evidence
7. Project implications for future development
8. Recommend pattern-based actions
            """.strip(),
            
            AnalysisTask.CAUSAL_RELATIONSHIP_MAPPING: """
Causal Analysis Framework:
1. Identify potential cause-effect relationships
2. Form testable hypotheses about causal mechanisms
3. Gather temporal and contextual evidence
4. Apply causal inference criteria (timing, mechanism, alternative explanations)
5. Compare against alternative causal explanations
6. Validate causal claims with multiple evidence types
7. Map complex multi-causal relationships
8. Synthesize causal network understanding
9. Identify leverage points for intervention
10. Recommend causal-based strategies
            """.strip(),
            
            AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION: """
Gap Identification Framework:
1. Map existing knowledge coverage areas
2. Identify patterns in knowledge distribution
3. Compare coverage against expected domains
4. Detect underexplored or missing topic areas
5. Assess gap significance and priority
6. Validate gaps against user goals and interests
7. Recommend gap-filling strategies
            """.strip(),
            
            AnalysisTask.STRATEGIC_INSIGHT_GENERATION: """
Strategic Insight Framework:
1. Survey all available information comprehensively
2. Identify high-level patterns and meta-themes
3. Generate multiple strategic hypotheses
4. Evaluate each hypothesis against evidence
5. Compare strategic options and trade-offs
6. Synthesize key strategic insights
7. Validate insights against practical constraints
8. Assess actionability and implementation paths
9. Prioritize insights by impact and feasibility
10. Formulate specific strategic recommendations
            """.strip()
        }
    
    def _initialize_example_chains(self) -> Dict[AnalysisTask, List[str]]:
        """Initialize example reasoning chains for different tasks."""
        return {
            AnalysisTask.CROSS_NOTE_SYNTHESIS: [
                "**Observe**: I notice three notes about productivity, two about stress management, and four about project planning",
                "**Recognize Patterns**: All productivity notes mention morning routines, stress notes focus on breathing techniques, project notes emphasize breaking down large tasks",
                "**Find Connections**: Morning routines appear to reduce stress, which improves project focus - there's a chain relationship",
                "**Gather Evidence**: Specific examples include '6 AM start improved my focus' and 'breathing exercises before big meetings help clarity'",
                "**Synthesize**: The user has developed an integrated approach: structured mornings → stress management → effective project execution",
                "**Validate**: This pattern appears consistently across 3 months of notes with concrete results mentioned",
                "**Conclude**: User's optimal productivity system combines temporal structure, stress regulation, and task management"
            ],
            
            AnalysisTask.PATTERN_EVOLUTION_TRACKING: [
                "**Establish Baseline**: Early notes (Jan-Mar) show scattered thoughts about 'work-life balance' with frustration",
                "**Track Changes**: Mid-period (Apr-Jun) introduces specific techniques like 'time blocking' and 'boundary setting'",
                "**Identify Shift**: July marks a clear change - less frustration language, more implementation-focused content",
                "**Analyze Factors**: Introduction of new job role in June appears to be the catalyst for systematic approach",
                "**Validate Evolution**: Recent notes (Aug) show confident mastery language and helping others with similar challenges",
                "**Project Forward**: Pattern suggests user is moving from learning to teaching phase in work-life integration",
                "**Recommend**: Consider formalizing this knowledge into a guide or mentoring opportunity"
            ],
            
            AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION: [
                "**Map Coverage**: Strong representation in technology (40%), personal development (30%), but weak in relationships (5%)",
                "**Identify Patterns**: Technical notes are detailed and systematic, personal notes are reflective, relationship notes are sparse and surface-level",
                "**Compare Expectations**: For someone in leadership role, relationship and communication skills knowledge seems underdeveloped",
                "**Assess Significance**: Gap is significant - leadership effectiveness heavily depends on relationship skills",
                "**Validate Priority**: Recent work challenges mentioned in notes seem related to team dynamics and communication",
                "**Recommend Strategy**: Focus learning on interpersonal skills, team leadership, and communication frameworks"
            ]
        }
    
    def _initialize_complexity_adaptations(self) -> Dict[ReasoningComplexity, Dict[str, Any]]:
        """Initialize adaptations based on reasoning complexity levels."""
        return {
            ReasoningComplexity.SIMPLE: {
                "step_count": "2-3 steps",
                "depth": "Surface-level reasoning with clear logic",
                "evidence": "Direct, obvious evidence",
                "validation": "Basic consistency checks"
            },
            ReasoningComplexity.MODERATE: {
                "step_count": "4-6 steps", 
                "depth": "Multi-layer reasoning with pattern recognition",
                "evidence": "Multiple evidence sources with some inference",
                "validation": "Cross-referencing and consistency validation"
            },
            ReasoningComplexity.COMPLEX: {
                "step_count": "7-10 steps",
                "depth": "Deep analysis with sub-reasoning chains",
                "evidence": "Comprehensive evidence gathering with triangulation",
                "validation": "Multiple validation approaches and alternative hypothesis testing"
            },
            ReasoningComplexity.EXPERT: {
                "step_count": "10+ steps",
                "depth": "Expert-level multi-dimensional analysis with nested reasoning",
                "evidence": "Exhaustive evidence analysis with nuanced interpretation",
                "validation": "Rigorous validation with scenario testing and limitation analysis"
            }
        }
    
    def create_chain_of_thought_prompt(self, 
                                     task: AnalysisTask,
                                     content: str,
                                     complexity: ReasoningComplexity = ReasoningComplexity.MODERATE,
                                     include_examples: bool = True,
                                     custom_framework: str = None) -> ChainOfThoughtPrompt:
        """
        Create a chain-of-thought prompt for complex analysis.
        
        Args:
            task: The type of analysis to perform
            content: The content to analyze
            complexity: The reasoning complexity level
            include_examples: Whether to include example reasoning chains
            custom_framework: Optional custom reasoning framework
        
        Returns:
            ChainOfThoughtPrompt: A complete chain-of-thought prompt
        """
        if task not in self.reasoning_templates:
            raise ValueError(f"Unknown analysis task: {task}")
        
        template = self.reasoning_templates[task]
        
        # Use custom framework or get default
        framework = custom_framework or self.task_frameworks.get(task, "Systematic step-by-step reasoning approach")
        
        # Get example chain if requested
        example_chain = self.example_chains.get(task, []) if include_examples else []
        
        return ChainOfThoughtPrompt(
            task_description=template["description"],
            reasoning_framework=framework,
            example_reasoning_chain=example_chain,
            target_content=content,
            complexity_level=complexity,
            output_format=OutputFormat.STRUCTURED_REASONING
        )
    
    def analyze_with_chain_of_thought(self,
                                    task: AnalysisTask,
                                    content: str,
                                    complexity: ReasoningComplexity = ReasoningComplexity.MODERATE,
                                    custom_steps: List[ReasoningStep] = None) -> ChainOfThoughtPrompt:
        """
        Perform chain-of-thought analysis with explicit reasoning steps.
        
        Args:
            task: The analysis task to perform
            content: Content to analyze
            complexity: Complexity level of reasoning
            custom_steps: Custom reasoning steps sequence
        
        Returns:
            ChainOfThoughtPrompt: Complete prompt ready for AI processing
        """
        prompt = self.create_chain_of_thought_prompt(task, content, complexity)
        
        # Add complexity-specific adaptations
        complexity_info = self.complexity_adaptations[complexity]
        
        # Enhance prompt with complexity guidance
        enhanced_framework = f"""
{prompt.reasoning_framework}

**Complexity Level**: {complexity.value.title()} ({complexity_info['step_count']})
**Reasoning Depth**: {complexity_info['depth']}
**Evidence Standard**: {complexity_info['evidence']}
**Validation Level**: {complexity_info['validation']}
        """.strip()
        
        prompt.reasoning_framework = enhanced_framework
        
        return prompt
    
    def create_multi_perspective_analysis(self,
                                        content: str,
                                        perspectives: List[str],
                                        synthesis_approach: str = "convergent") -> ChainOfThoughtPrompt:
        """
        Create a multi-perspective chain-of-thought analysis.
        
        Args:
            content: Content to analyze from multiple perspectives
            perspectives: List of different analytical perspectives
            synthesis_approach: How to synthesize multiple perspectives
        
        Returns:
            ChainOfThoughtPrompt: Multi-perspective analysis prompt
        """
        framework = f"""
Multi-Perspective Analysis Framework:

**Perspectives to Consider**: {', '.join(perspectives)}
**Synthesis Approach**: {synthesis_approach}

**Reasoning Process**:
1. **Individual Perspective Analysis**: Analyze from each perspective separately
   - What unique insights does each perspective reveal?
   - What evidence supports each perspective's conclusions?
   - What are the strengths and limitations of each view?

2. **Cross-Perspective Comparison**: Compare insights across perspectives
   - Where do perspectives align or conflict?
   - What complementary insights emerge?
   - Which perspective provides the most compelling evidence?

3. **Synthesis Integration**: Integrate perspectives into coherent understanding
   - How can different perspectives be reconciled?
   - What meta-insights emerge from the combination?
   - What is the most comprehensive interpretation?

4. **Validation and Implications**: Validate synthesis and extract implications
   - Does the integrated view make logical sense?
   - What actionable insights emerge?
   - What recommendations follow from this analysis?
        """.strip()
        
        return ChainOfThoughtPrompt(
            task_description=f"Analyze content from multiple perspectives: {', '.join(perspectives)}",
            reasoning_framework=framework,
            target_content=content,
            complexity_level=ReasoningComplexity.COMPLEX,
            output_format=OutputFormat.STRUCTURED_REASONING
        )
    
    def create_causal_reasoning_prompt(self,
                                     content: str,
                                     potential_causes: List[str] = None,
                                     potential_effects: List[str] = None) -> ChainOfThoughtPrompt:
        """
        Create a specialized prompt for causal reasoning analysis.
        
        Args:
            content: Content to analyze for causal relationships
            potential_causes: Hypothesized causes to investigate
            potential_effects: Hypothesized effects to investigate
        
        Returns:
            ChainOfThoughtPrompt: Causal reasoning analysis prompt
        """
        cause_guidance = f"\n**Potential Causes to Investigate**: {', '.join(potential_causes)}" if potential_causes else ""
        effect_guidance = f"\n**Potential Effects to Investigate**: {', '.join(potential_effects)}" if potential_effects else ""
        
        framework = f"""
Causal Reasoning Framework:

**Objective**: Identify and validate causal relationships in the content
{cause_guidance}
{effect_guidance}

**Systematic Causal Analysis**:
1. **Temporal Sequencing**: What events or factors came before others?
2. **Mechanism Identification**: What plausible mechanisms could link causes to effects?
3. **Alternative Explanations**: What other factors could explain the observed effects?
4. **Evidence Strength**: How strong is the evidence for each causal claim?
5. **Confounding Factors**: What other variables might influence the relationship?
6. **Causal Network**: How do multiple causes and effects interact?
7. **Intervention Points**: Where could changes most effectively influence outcomes?

**Validation Criteria**:
- Temporal precedence (cause before effect)
- Plausible mechanism (how cause leads to effect)
- Elimination of alternatives (ruling out other explanations)
- Strength of association (consistent relationship)
- Dose-response relationship (stronger cause → stronger effect)
        """.strip()
        
        return ChainOfThoughtPrompt(
            task_description="Identify and analyze causal relationships within the content",
            reasoning_framework=framework,
            target_content=content,
            complexity_level=ReasoningComplexity.EXPERT,
            output_format=OutputFormat.STRUCTURED_REASONING
        )
    
    def create_strategic_reasoning_prompt(self,
                                        content: str,
                                        strategic_context: Dict[str, Any] = None,
                                        decision_timeframe: str = "medium-term") -> ChainOfThoughtPrompt:
        """
        Create a strategic reasoning prompt for high-level analysis.
        
        Args:
            content: Content to analyze strategically
            strategic_context: Additional strategic context information
            decision_timeframe: Timeframe for strategic decisions
        
        Returns:
            ChainOfThoughtPrompt: Strategic reasoning analysis prompt
        """
        context_info = ""
        if strategic_context:
            context_items = []
            for key, value in strategic_context.items():
                context_items.append(f"- {key.title()}: {value}")
            context_info = f"\n**Strategic Context**:\n" + "\n".join(context_items)
        
        framework = f"""
Strategic Reasoning Framework:

**Objective**: Generate strategic insights and actionable recommendations
**Decision Timeframe**: {decision_timeframe}
{context_info}

**Strategic Analysis Process**:
1. **Situational Assessment**: What is the current state and key factors?
2. **Trend Analysis**: What patterns and trajectories are evident?
3. **Opportunity Identification**: What opportunities are emerging or available?
4. **Risk Assessment**: What risks and challenges need consideration?
5. **Resource Evaluation**: What capabilities and constraints exist?
6. **Strategic Options**: What different strategic approaches are possible?
7. **Option Evaluation**: How do strategic options compare on key criteria?
8. **Integration**: What integrated strategic approach makes most sense?
9. **Implementation**: What are the key implementation considerations?
10. **Success Metrics**: How will success be measured and monitored?

**Strategic Thinking Principles**:
- Long-term perspective with short-term actions
- Systems thinking and interconnection awareness
- Scenario planning and contingency consideration
- Stakeholder impact and alignment analysis
- Resource optimization and constraint management
        """.strip()
        
        return ChainOfThoughtPrompt(
            task_description="Conduct strategic analysis and generate actionable strategic insights",
            reasoning_framework=framework,
            target_content=content,
            complexity_level=ReasoningComplexity.EXPERT,
            output_format=OutputFormat.STRUCTURED_REASONING
        )
    
    def get_reasoning_quality_metrics(self, prompt: ChainOfThoughtPrompt) -> Dict[str, Any]:
        """
        Get quality metrics for a chain-of-thought prompt.
        
        Args:
            prompt: The prompt to evaluate
        
        Returns:
            Dict containing quality metrics
        """
        metrics = {
            "complexity_level": prompt.complexity_level.value,
            "framework_completeness": len(prompt.reasoning_framework) > 200,
            "has_examples": len(prompt.example_reasoning_chain) > 0,
            "step_count_estimate": len(prompt.example_reasoning_chain) if prompt.example_reasoning_chain else "variable",
            "prompt_length": len(prompt.build_prompt()),
            "reasoning_depth": "high" if prompt.complexity_level in [ReasoningComplexity.COMPLEX, ReasoningComplexity.EXPERT] else "moderate"
        }
        
        return metrics
    
    def suggest_complexity_level(self, 
                                content_length: int,
                                task_type: AnalysisTask,
                                available_context: Dict[str, Any] = None) -> ReasoningComplexity:
        """
        Suggest appropriate complexity level based on content and task characteristics.
        
        Args:
            content_length: Length of content to analyze (word count)
            task_type: Type of analysis task
            available_context: Additional context information
        
        Returns:
            Recommended ReasoningComplexity level
        """
        # Base complexity from task type
        task_complexities = {
            AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION: ReasoningComplexity.MODERATE,
            AnalysisTask.CROSS_NOTE_SYNTHESIS: ReasoningComplexity.COMPLEX,
            AnalysisTask.PATTERN_EVOLUTION_TRACKING: ReasoningComplexity.COMPLEX,
            AnalysisTask.CAUSAL_RELATIONSHIP_MAPPING: ReasoningComplexity.EXPERT,
            AnalysisTask.STRATEGIC_INSIGHT_GENERATION: ReasoningComplexity.EXPERT,
            AnalysisTask.CONTRADICTION_RESOLUTION: ReasoningComplexity.COMPLEX
        }
        
        base_complexity = task_complexities.get(task_type, ReasoningComplexity.MODERATE)
        
        # Adjust based on content length
        if content_length < 100:  # Very short content
            if base_complexity == ReasoningComplexity.EXPERT:
                return ReasoningComplexity.COMPLEX
            elif base_complexity == ReasoningComplexity.COMPLEX:
                return ReasoningComplexity.MODERATE
        elif content_length > 1000:  # Very long content
            if base_complexity == ReasoningComplexity.SIMPLE:
                return ReasoningComplexity.MODERATE
            elif base_complexity == ReasoningComplexity.MODERATE:
                return ReasoningComplexity.COMPLEX
        
        # Adjust based on available context
        if available_context:
            context_richness = len(available_context.get("related_notes", [])) + len(available_context.get("background_info", []))
            if context_richness > 5:  # Rich context available
                complexity_levels = list(ReasoningComplexity)
                current_index = complexity_levels.index(base_complexity)
                if current_index < len(complexity_levels) - 1:
                    return complexity_levels[current_index + 1]
        
        return base_complexity


# Convenience functions for common chain-of-thought scenarios
def analyze_cross_note_synthesis(notes_content: List[str], 
                               synthesis_focus: str = "themes and patterns") -> ChainOfThoughtPrompt:
    """Quick function for cross-note synthesis with chain-of-thought reasoning."""
    combined_content = f"""
**Synthesis Focus**: {synthesis_focus}

**Notes to Synthesize**:
""" + "\n\n---\n\n".join([f"**Note {i+1}**: {content}" for i, content in enumerate(notes_content)])
    
    analyzer = ChainOfThoughtAnalyzer()
    return analyzer.analyze_with_chain_of_thought(
        AnalysisTask.CROSS_NOTE_SYNTHESIS,
        combined_content,
        ReasoningComplexity.COMPLEX
    )


def track_pattern_evolution(temporal_content: Dict[str, str], 
                          pattern_focus: str = "behavioral and thematic changes") -> ChainOfThoughtPrompt:
    """Quick function for tracking pattern evolution over time."""
    temporal_analysis_content = f"""
**Pattern Focus**: {pattern_focus}

**Temporal Content Analysis**:
"""
    
    for time_period, content in sorted(temporal_content.items()):
        temporal_analysis_content += f"\n**{time_period}**: {content}\n"
    
    analyzer = ChainOfThoughtAnalyzer()
    return analyzer.analyze_with_chain_of_thought(
        AnalysisTask.PATTERN_EVOLUTION_TRACKING,
        temporal_analysis_content,
        ReasoningComplexity.COMPLEX
    )


def identify_knowledge_gaps(vault_summary: str, 
                          expertise_domain: str = "general knowledge work") -> ChainOfThoughtPrompt:
    """Quick function for knowledge gap identification."""
    gap_analysis_content = f"""
**Expertise Domain**: {expertise_domain}
**Current Vault Summary**: {vault_summary}

**Gap Analysis Context**: Identify areas where knowledge coverage is insufficient, inconsistent, or missing entirely relative to the stated domain expertise.
"""
    
    analyzer = ChainOfThoughtAnalyzer()
    return analyzer.analyze_with_chain_of_thought(
        AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION,
        gap_analysis_content,
        ReasoningComplexity.MODERATE
    )


def generate_strategic_insights(strategic_content: str,
                              context: Dict[str, Any] = None,
                              timeframe: str = "6-month") -> ChainOfThoughtPrompt:
    """Quick function for strategic insight generation."""
    analyzer = ChainOfThoughtAnalyzer()
    return analyzer.create_strategic_reasoning_prompt(
        strategic_content,
        strategic_context=context,
        decision_timeframe=timeframe
    )


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Chain-of-Thought Prompting Demo ===\n")
    
    # Sample complex analysis scenario
    sample_notes = [
        """
        Project Update - Week 1: Started the VaultMind implementation with high energy. 
        Team seems motivated but I'm concerned about scope creep. The AI integration 
        is more complex than initially estimated. Need to balance feature richness 
        with delivery timeline.
        """,
        """
        Project Update - Week 4: Hit our first major milestone but behind on documentation. 
        The team is working well together, though Mike seems overwhelmed with the backend 
        complexity. Sarah's design work is excellent but we need better communication 
        between frontend and backend teams.
        """,
        """
        Project Update - Week 8: Significant breakthrough with the prompting system. 
        The chain-of-thought approach is showing real promise. However, we're definitely 
        behind schedule and need to make some tough decisions about feature prioritization. 
        Team morale is still good but I can sense some stress building.
        """,
        """
        Project Update - Week 12: Final sprint toward MVP. The core functionality is solid 
        and early user feedback is very positive. We had to cut some advanced features 
        but the foundation is strong. Planning to do a proper retrospective to capture 
        lessons learned for future iterations.
        """
    ]
    
    sample_strategic_content = """
    VaultMind Strategic Assessment - Q3 2025
    
    Market Position: We've successfully launched our MVP with positive user feedback. 
    Key differentiators are the adaptive prompting system and vault-specific analysis 
    capabilities. However, competition is increasing with three new entrants this quarter.
    
    Technical Capabilities: Core AI integration is solid, prompting framework is 
    innovative and extensible. Main technical debt is in the user interface and 
    documentation system. Team has strong AI/ML expertise but needs UX design support.
    
    User Adoption: 150+ beta users with 73% weekly retention. Power users love the 
    advanced features, but onboarding is challenging for newcomers. Feature requests 
    focus on integrations (Notion, Roam) and mobile access.
    
    Resource Situation: 8 months runway remaining. Team of 4 core developers plus 
    2 part-time contributors. Need to decide between scaling team vs. focusing on 
    efficiency improvements.
    
    Strategic Questions: Should we prioritize user acquisition, feature development, 
    or technical infrastructure? How do we balance advanced capabilities with 
    accessibility? What's our defensible competitive moat?
    """
    
    # Initialize analyzer
    analyzer = ChainOfThoughtAnalyzer()
    
    print("1. Cross-Note Synthesis Analysis:")
    synthesis_prompt = analyze_cross_note_synthesis(
        sample_notes, 
        "project management patterns and team dynamics"
    )
    
    quality_metrics = analyzer.get_reasoning_quality_metrics(synthesis_prompt)
    print(f"   Complexity Level: {quality_metrics['complexity_level']}")
    print(f"   Has Examples: {quality_metrics['has_examples']}")
    print(f"   Framework Complete: {quality_metrics['framework_completeness']}")
    print(f"   Prompt Length: {quality_metrics['prompt_length']:,} characters")
    print()
    
    print("2. Strategic Reasoning Analysis:")
    strategic_context = {
        "industry": "AI/Knowledge Management Tools",
        "stage": "Post-MVP, Pre-Scale",
        "team_size": "6 people",
        "runway": "8 months"
    }
    
    strategic_prompt = generate_strategic_insights(
        sample_strategic_content,
        context=strategic_context,
        timeframe="12-month"
    )
    
    print(f"   Task: {strategic_prompt.task_description[:80]}...")
    print(f"   Complexity: {strategic_prompt.complexity_level.value}")
    print(f"   Framework Length: {len(strategic_prompt.reasoning_framework)} characters")
    print()
    
    print("3. Pattern Evolution Tracking:")
    temporal_data = {
        "Week 1": sample_notes[0],
        "Week 4": sample_notes[1], 
        "Week 8": sample_notes[2],
        "Week 12": sample_notes[3]
    }
    
    evolution_prompt = track_pattern_evolution(
        temporal_data,
        "team dynamics and project management approach evolution"
    )
    
    print(f"   Analysis Focus: Pattern evolution over 12-week period")
    print(f"   Example Steps: {len(evolution_prompt.example_reasoning_chain)} reasoning examples")
    print(f"   Content Sections: {len(temporal_data)} time periods")
    print()
    
    print("4. Complexity Level Suggestions:")
    test_scenarios = [
        ("Short meeting note", 50, AnalysisTask.KNOWLEDGE_GAP_IDENTIFICATION),
        ("Comprehensive project retrospective", 800, AnalysisTask.CAUSAL_RELATIONSHIP_MAPPING),
        ("Multi-note synthesis", 1200, AnalysisTask.CROSS_NOTE_SYNTHESIS),
        ("Strategic planning document", 1500, AnalysisTask.STRATEGIC_INSIGHT_GENERATION)
    ]
    
    for description, word_count, task in test_scenarios:
        suggested = analyzer.suggest_complexity_level(word_count, task)
        print(f"   {description} ({word_count} words, {task.value}): {suggested.value}")
    print()
    
    print("5. Multi-Perspective Analysis:")
    perspectives = ["project manager", "team member", "user advocate", "business strategist"]
    multi_perspective_prompt = analyzer.create_multi_perspective_analysis(
        sample_strategic_content,
        perspectives,
        "convergent"
    )
    
    print(f"   Perspectives: {len(perspectives)} different viewpoints")
    print(f"   Synthesis Approach: Convergent integration")
    print(f"   Framework: Multi-layered perspective analysis")
    print()
    
    print("6. Causal Reasoning Analysis:")
    potential_causes = ["team communication patterns", "scope management decisions", "technical complexity"]
    potential_effects = ["timeline delays", "team stress levels", "feature quality"]
    
    causal_prompt = analyzer.create_causal_reasoning_prompt(
        "\n".join(sample_notes),
        potential_causes=potential_causes,
        potential_effects=potential_effects
    )
    
    print(f"   Investigating {len(potential_causes)} potential causes")
    print(f"   Analyzing {len(potential_effects)} potential effects")
    print(f"   Validation: Expert-level causal inference criteria")
    
    print("\n=== Chain-of-Thought Analysis System Ready ===")
    print("Key capabilities:")
    print("• Step-by-step reasoning for complex analysis")
    print("• Multiple specialized reasoning frameworks")
    print("• Complexity-adaptive reasoning depth")
    print("• Multi-perspective and causal analysis")
    print("• Strategic insight generation with validation")
    print("• Transparent, explainable analysis process")
