"""
VaultMind Evaluation and Testing Framework

This module implements a comprehensive testing framework for evaluating different
prompting strategies with standardized datasets, judge prompts, and automated
scoring systems to measure effectiveness and reliability.

The framework enables systematic comparison of zero-shot, one-shot, multi-shot,
dynamic, and chain-of-thought prompting approaches using objective metrics
and AI-assisted evaluation.

Author: VaultMind Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import statistics
import re
from collections import defaultdict, Counter
import uuid


class PromptingStrategy(Enum):
    """Different prompting strategies to evaluate."""
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    MULTI_SHOT = "multi_shot"
    DYNAMIC = "dynamic"
    CHAIN_OF_THOUGHT = "chain_of_thought"


class EvaluationMetric(Enum):
    """Metrics for evaluating prompting strategy effectiveness."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    CONSISTENCY = "consistency"
    ACTIONABILITY = "actionability"
    INSIGHT_DEPTH = "insight_depth"
    ERROR_RATE = "error_rate"
    PROCESSING_TIME = "processing_time"
    TOKEN_EFFICIENCY = "token_efficiency"


class TestCategory(Enum):
    """Categories of test cases for comprehensive evaluation."""
    SIMPLE_EXTRACTION = "simple_extraction"
    COMPLEX_ANALYSIS = "complex_analysis"
    MULTI_NOTE_SYNTHESIS = "multi_note_synthesis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    STRATEGIC_REASONING = "strategic_reasoning"
    PATTERN_RECOGNITION = "pattern_recognition"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"


class ScoringApproach(Enum):
    """Different approaches for scoring test results."""
    RULE_BASED = "rule_based"
    AI_JUDGE = "ai_judge"
    HUMAN_REFERENCE = "human_reference"
    HYBRID = "hybrid"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class TestSample:
    """Represents a single test sample with input, expected output, and metadata."""
    test_id: str
    category: TestCategory
    difficulty_level: str  # "simple", "moderate", "complex", "expert"
    input_content: str
    task_description: str
    expected_output: str
    reference_reasoning: str = ""
    evaluation_criteria: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_complexity_score(self) -> float:
        """Get numerical complexity score for the test sample."""
        complexity_map = {
            "simple": 1.0,
            "moderate": 2.5,
            "complex": 4.0,
            "expert": 5.0
        }
        return complexity_map.get(self.difficulty_level, 2.5)


@dataclass
class TestResult:
    """Represents the result of testing a prompting strategy on a sample."""
    test_id: str
    strategy: PromptingStrategy
    actual_output: str
    scores: Dict[EvaluationMetric, float] = field(default_factory=dict)
    processing_time: float = 0.0
    token_count: int = 0
    error_occurred: bool = False
    error_message: str = ""
    judge_reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_overall_score(self) -> float:
        """Calculate overall weighted score across all metrics."""
        if not self.scores:
            return 0.0

        # Weight different metrics based on importance
        weights = {
            EvaluationMetric.ACCURACY: 0.25,
            EvaluationMetric.COMPLETENESS: 0.20,
            EvaluationMetric.RELEVANCE: 0.15,
            EvaluationMetric.CLARITY: 0.15,
            EvaluationMetric.INSIGHT_DEPTH: 0.15,
            EvaluationMetric.ACTIONABILITY: 0.10
        }

        weighted_score = 0.0
        total_weight = 0.0

        for metric, score in self.scores.items():
            weight = weights.get(metric, 0.1)
            weighted_score += score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report comparing different strategies."""
    test_suite_name: str
    strategies_tested: List[PromptingStrategy]
    total_tests: int
    results_by_strategy: Dict[PromptingStrategy, List[TestResult]] = field(default_factory=dict)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def calculate_strategy_rankings(self) -> List[Tuple[PromptingStrategy, float]]:
        """Calculate and rank strategies by overall performance."""
        strategy_scores = {}

        for strategy, results in self.results_by_strategy.items():
            if results:
                scores = [r.get_overall_score() for r in results if not r.error_occurred]
                strategy_scores[strategy] = statistics.mean(scores) if scores else 0.0

        return sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)


class JudgePromptBuilder:
    """Builds judge prompts for AI-assisted evaluation of test results."""

    def __init__(self):
        self.judge_templates = self._initialize_judge_templates()

    def _initialize_judge_templates(self) -> Dict[EvaluationMetric, str]:
        """Initialize judge prompt templates for different metrics."""
        return {
            EvaluationMetric.ACCURACY: """
You are an expert evaluator assessing the accuracy of AI-generated analysis.

**Task**: Compare the AI's output against the expected reference output and evaluate accuracy.

**Evaluation Criteria**:
- Factual correctness: Are the facts and claims accurate?
- Logical consistency: Do the conclusions follow logically from the evidence?
- Error identification: Are there any clear mistakes or inaccuracies?

**Input Content**: {input_content}
**Expected Output**: {expected_output}
**AI Generated Output**: {actual_output}

Rate accuracy on a scale of 0-10 where:
- 0-2: Major inaccuracies, incorrect conclusions
- 3-4: Some accuracy but significant errors
- 5-6: Generally accurate with minor errors
- 7-8: Highly accurate with minimal errors
- 9-10: Exceptional accuracy, no detectable errors

Provide your rating and brief reasoning:
            """.strip(),

            EvaluationMetric.COMPLETENESS: """
You are an expert evaluator assessing the completeness of AI-generated analysis.

**Task**: Evaluate how thoroughly the AI addressed all aspects of the analysis task.

**Evaluation Criteria**:
- Coverage: Did the AI address all key points mentioned in the expected output?
- Depth: Is the level of detail appropriate for the task?
- Missing elements: What important aspects were overlooked?

**Task Description**: {task_description}
**Expected Output**: {expected_output}
**AI Generated Output**: {actual_output}

Rate completeness on a scale of 0-10 where:
- 0-2: Major gaps, many important points missed
- 3-4: Some coverage but significant omissions
- 5-6: Adequate coverage with some gaps
- 7-8: Comprehensive coverage with minor gaps
- 9-10: Exceptional completeness, all key points addressed

Provide your rating and brief reasoning:
            """.strip(),

            EvaluationMetric.CLARITY: """
You are an expert evaluator assessing the clarity and understandability of AI-generated analysis.

**Task**: Evaluate how clear, well-structured, and easy to understand the AI's output is.

**Evaluation Criteria**:
- Organization: Is the information well-structured and logically organized?
- Language: Is the language clear, precise, and appropriate?
- Readability: Is the output easy to follow and understand?

**AI Generated Output**: {actual_output}

Rate clarity on a scale of 0-10 where:
- 0-2: Confusing, poorly organized, difficult to understand
- 3-4: Some clarity issues, structure problems
- 5-6: Generally clear with some confusing aspects
- 7-8: Well-organized and clear communication
- 9-10: Exceptionally clear, perfectly structured

Provide your rating and brief reasoning:
            """.strip(),

            EvaluationMetric.INSIGHT_DEPTH: """
You are an expert evaluator assessing the depth and quality of insights in AI-generated analysis.

**Task**: Evaluate the depth, sophistication, and value of insights provided by the AI.

**Evaluation Criteria**:
- Insight quality: Are the insights meaningful and valuable?
- Analytical depth: Does the analysis go beyond surface-level observations?
- Novel perspectives: Does the AI provide fresh or unexpected insights?

**Input Content**: {input_content}
**AI Generated Output**: {actual_output}

Rate insight depth on a scale of 0-10 where:
- 0-2: Superficial, obvious observations only
- 3-4: Some insights but mostly surface-level
- 5-6: Moderate depth with some valuable insights
- 7-8: Deep analysis with meaningful insights
- 9-10: Exceptional depth, profound and valuable insights

Provide your rating and brief reasoning:
            """.strip(),

            EvaluationMetric.ACTIONABILITY: """
You are an expert evaluator assessing the actionability of AI-generated analysis.

**Task**: Evaluate how actionable and practical the AI's recommendations and insights are.

**Evaluation Criteria**:
- Practical value: Can the insights be acted upon effectively?
- Specificity: Are recommendations specific enough to implement?
- Feasibility: Are suggested actions realistic and achievable?

**AI Generated Output**: {actual_output}

Rate actionability on a scale of 0-10 where:
- 0-2: No actionable insights, purely theoretical
- 3-4: Limited actionability, vague recommendations
- 5-6: Some actionable elements with moderate specificity
- 7-8: Highly actionable with specific, practical recommendations
- 9-10: Exceptional actionability, clear implementation guidance

Provide your rating and brief reasoning:
            """.strip()
        }

    def build_judge_prompt(self,
                          metric: EvaluationMetric,
                          test_sample: TestSample,
                          actual_output: str) -> str:
        """Build a judge prompt for evaluating a specific metric."""
        if metric not in self.judge_templates:
            raise ValueError(f"No judge template available for metric: {metric}")

        template = self.judge_templates[metric]

        return template.format(
            input_content=test_sample.input_content,
            task_description=test_sample.task_description,
            expected_output=test_sample.expected_output,
            actual_output=actual_output
        )


class TestFramework:
    """
    Main testing framework for evaluating prompting strategies.

    Provides comprehensive evaluation capabilities including standardized test samples,
    AI-assisted judging, and detailed performance analytics.
    """

    def __init__(self):
        self.test_samples = self._initialize_test_samples()
        self.judge_builder = JudgePromptBuilder()
        self.evaluation_history = []
        self.scoring_functions = self._initialize_scoring_functions()

    def _initialize_test_samples(self) -> List[TestSample]:
        """Initialize comprehensive test sample dataset."""
        samples = []

        # Simple Extraction Test
        samples.append(TestSample(
            test_id="SE001",
            category=TestCategory.SIMPLE_EXTRACTION,
            difficulty_level="simple",
            input_content="""
Daily Standup - March 15, 2025
Team: Alice, Bob, Charlie

Alice: Completed user authentication module yesterday. Today working on password reset functionality. No blockers.
Bob: Finished API endpoint testing. Moving to database optimization. Need help with query performance.
Charlie: Wrapped up UI mockups. Starting frontend implementation. Waiting for design approval from Sarah.

Action items:
- Bob to consult with database team about performance
- Charlie to follow up with Sarah on design approval
- Next standup: Same time tomorrow
            """,
            task_description="Extract all action items from this standup meeting note",
            expected_output="""
**Action Items:**
• Bob: Consult with database team about query performance optimization
• Charlie: Follow up with Sarah to get design approval for UI mockups
• Team: Attend next standup at the same time tomorrow

**Responsible Parties:** Bob (database consultation), Charlie (design approval follow-up)
**Timeline:** Database consultation and design approval needed before next standup
            """,
            reference_reasoning="Simple extraction task requiring identification of explicit action items with clear ownership and context",
            evaluation_criteria={
                "completeness": "Must identify all 3 action items",
                "accuracy": "Must correctly assign ownership",
                "format": "Should organize items clearly"
            }
        ))

        # Complex Analysis Test
        samples.append(TestSample(
            test_id="CA001",
            category=TestCategory.COMPLEX_ANALYSIS,
            difficulty_level="complex",
            input_content="""
Project Retrospective - VaultMind Development Q3 2025

**What Went Well:**
- Technical architecture decisions proved solid under stress testing
- Team collaboration improved significantly after implementing daily check-ins
- User feedback loop with beta testers provided valuable insights
- AI integration exceeded performance expectations
- Documentation quality improved with new peer review process

**What Could Be Improved:**
- Initial scope estimation was overly optimistic leading to timeline pressures
- Communication gaps between frontend and backend teams caused integration delays
- Testing infrastructure was built reactively rather than proactively
- Resource allocation decisions were made too late in the sprint
- Technical debt accumulated faster than planned due to delivery pressure

**Key Insights:**
- Early investment in team communication pays compound dividends
- AI performance optimization requires more experimentation time than traditional features
- User feedback quality improves dramatically with structured feedback mechanisms
- Technical architecture flexibility becomes critical during rapid iteration phases

**Looking Forward:**
- Need better upfront planning with realistic timeline buffers
- Should implement automated testing pipeline before next major feature
- Consider hiring additional DevOps support for scaling infrastructure
- Establish regular technical debt reduction cycles
            """,
            task_description="Analyze this project retrospective to identify systemic patterns, root causes of issues, and strategic recommendations for future development cycles",
            expected_output="""
**Systemic Patterns Identified:**
1. **Planning vs. Execution Gap**: Overly optimistic scope estimation suggests systematic underestimation of complexity, particularly for AI features
2. **Communication Evolution**: Team collaboration improvement shows learning curve - early problems were process-based, not people-based
3. **Technical vs. Process Balance**: Strong technical decisions but weak process infrastructure (testing, debt management)

**Root Cause Analysis:**
- **Timeline Pressures**: Stem from poor initial estimation, creating cascading effects on quality and technical debt
- **Integration Delays**: Communication gaps indicate insufficient cross-team coordination processes
- **Reactive Infrastructure**: Focus on feature delivery over foundational systems

**Strategic Recommendations:**
1. **Process Improvements**: 
   - Implement planning poker with AI complexity multipliers
   - Establish cross-team integration checkpoints
   - Build automated testing pipeline before next sprint
2. **Resource Strategy**:
   - Hire DevOps engineer to prevent infrastructure bottlenecks
   - Allocate 20% of sprint capacity to technical debt reduction
3. **Learning Integration**:
   - Formalize user feedback structured collection process
   - Create technical architecture review cycles

**Success Indicators to Track:**
- Estimation accuracy improvement over next 2 sprints
- Cross-team integration issue frequency reduction
- Technical debt accumulation rate stabilization
            """,
            reference_reasoning="Complex analysis requiring pattern recognition, causal analysis, and strategic synthesis across multiple dimensions of project management",
            evaluation_criteria={
                "depth": "Must identify systemic patterns beyond surface observations",
                "reasoning": "Should show causal connections between problems",
                "actionability": "Recommendations must be specific and implementable"
            }
        ))

        # Multi-Note Synthesis Test
        samples.append(TestSample(
            test_id="MS001",
            category=TestCategory.MULTI_NOTE_SYNTHESIS,
            difficulty_level="complex",
            input_content="""
**Note 1 - Personal Development Goal (Jan 2025):**
Decided to focus on improving public speaking skills this year. Have always been nervous about presentations, but career growth requires better communication. Signed up for Toastmasters and planning to speak at 2 industry conferences.

**Note 2 - Toastmasters Experience (Mar 2025):**
Third Toastmasters meeting today. Getting more comfortable with impromptu speaking exercises. Key insight: preparation reduces anxiety dramatically, but over-preparation can make me sound robotic. Finding the balance.

**Note 3 - Conference Proposal (May 2025):**
Submitted proposal for DevCon 2025 on "AI-Assisted Knowledge Management." Topic combines my technical expertise with vault analysis work. Feeling nervous but excited. Using storytelling framework learned from Toastmasters.

**Note 4 - DevCon Presentation (Aug 2025):**
Nailed the DevCon presentation! 200+ attendees, great Q&A session, multiple people approached afterward. Used personal VaultMind journey as case study. Confidence level completely different from January. Already accepted invitation to speak at TechSummit in November.

**Note 5 - Reflection on Growth (Aug 2025):**
Looking back at January's goal, can't believe the transformation. Public speaking went from biggest fear to genuine strength. Key factors: consistent practice, real-world application, choosing topics I'm passionate about, and accepting that nervousness never fully disappears but becomes manageable.
            """,
            task_description="Synthesize insights across these 5 notes to understand the complete personal development journey and extract transferable principles for skill development",
            expected_output="""
**Personal Development Journey Analysis:**

**Transformation Arc:**
- **Starting Point**: Public speaking as "biggest fear" limiting career growth
- **Systematic Approach**: Toastmasters training + real-world application (conferences)
- **Outcome**: Complete reversal - speaking became "genuine strength" with ongoing opportunities

**Key Success Factors Identified:**
1. **Strategic Practice Environment**: Toastmasters provided safe space for skill development
2. **Real-World Application**: Conference speaking forced practical application under pressure
3. **Content-Passion Alignment**: Speaking about VaultMind work leveraged existing expertise
4. **Expectation Management**: Accepting nervousness as manageable rather than eliminatable

**Transferable Development Principles:**
1. **Structured Learning + Practical Application**: Combine formal training with real-world challenges
2. **Progressive Difficulty**: Toastmasters exercises → Conference proposal → Major presentation
3. **Leverage Existing Strengths**: Use domain expertise as foundation for new skill development
4. **Mindset Shift**: Reframe anxiety as energy rather than obstacle
5. **Compound Growth**: Success creates opportunities (DevCon → TechSummit invitation)

**Meta-Insights on Skill Development:**
- **Timeline**: 8-month transformation from goal-setting to mastery demonstration
- **Turning Points**: Conference proposal (May) marked shift from learning to applying
- **Success Metrics**: External validation (audience engagement, speaking invitations) confirmed internal confidence growth
- **Sustainable Practice**: Framework established for continued speaking opportunities

**Actionable Framework for Other Skills:**
1. Identify specific skill gap limiting growth
2. Find structured learning environment (equivalent to Toastmasters)
3. Commit to real-world application with deadline pressure
4. Align practice content with existing expertise
5. Reframe discomfort as growth indicator rather than stop signal
            """,
            reference_reasoning="Multi-note synthesis requiring temporal analysis, pattern recognition, and principle extraction across a complete development journey",
            evaluation_criteria={
                "synthesis": "Must connect insights across all 5 notes coherently",
                "pattern_recognition": "Should identify systematic success factors",
                "transferability": "Must extract principles applicable beyond public speaking"
            }
        ))

        # Temporal Analysis Test
        samples.append(TestSample(
            test_id="TA001",
            category=TestCategory.TEMPORAL_ANALYSIS,
            difficulty_level="moderate",
            input_content="""
**Week 1 (Jan 2025)**: Started morning routine experiment. 5:30 AM wake up, meditation, exercise, journaling. Feeling motivated but tired. Productivity seems higher in mornings.

**Week 4 (Jan 2025)**: Routine is becoming habit. Less resistance to early wake-up. Energy levels more consistent throughout day. Noticed better decision-making in afternoon meetings.

**Week 8 (Feb 2025)**: Hit first major challenge - got sick and routine broke down for a week. Struggled to restart. Realized routine was fragile, tied too closely to perfect conditions.

**Week 12 (Mar 2025)**: Adapted routine to be more flexible. If I miss morning workout, do evening walk. If too tired for 20-min meditation, do 5 minutes. Consistency over perfection approach working better.

**Week 16 (Apr 2025)**: Routine feels automatic now. Don't need willpower to maintain it. Started adding creative writing to morning block. The quiet morning time has become my most productive period.

**Week 20 (May 2025)**: Evaluating impact: sleeping better, more creative output, better emotional regulation. The biggest change is feeling like I control my day rather than reacting to it. Planning to maintain long-term.
            """,
            task_description="Track the evolution of this morning routine experiment, identifying key phases, turning points, and lessons learned about habit formation",
            expected_output="""
**Habit Formation Evolution Timeline:**

**Phase 1: Motivated Initiation (Weeks 1-4)**
- *Characteristics*: High motivation, rigid structure, initial resistance
- *Key Insight*: Early productivity gains provided positive reinforcement
- *Challenge*: Physical adaptation period (tiredness, adjustment)

**Phase 2: Fragility Exposure (Weeks 5-8)**
- *Turning Point*: Illness disrupted routine, revealed system weakness
- *Key Learning*: Rigid systems are vulnerable to disruption
- *Problem Identified*: All-or-nothing approach unsustainable

**Phase 3: Adaptive Restructuring (Weeks 9-12)**
- *Strategic Shift*: "Consistency over perfection" philosophy
- *System Evolution*: Built in flexibility and alternatives
- *Resilience Building*: Prepared for imperfect conditions

**Phase 4: Automatization (Weeks 13-16)**  
- *Milestone*: Routine became automatic, willpower no longer required
- *Expansion*: Added creative writing, leveraging established foundation
- *Optimization*: Identified morning time as peak productivity window

**Phase 5: Sustainable Integration (Weeks 17-20)**
- *Outcome Assessment*: Multiple life areas improved (sleep, creativity, emotional regulation)
- *Psychological Shift*: From reactive to proactive daily experience
- *Long-term Commitment*: Decision to maintain based on proven benefits

**Critical Success Factors:**
1. **Early Win Recognition**: Productivity improvements provided motivation fuel
2. **Failure Learning**: Used disruption as system improvement opportunity
3. **Flexibility Integration**: Adapted structure without abandoning core principles
4. **Progressive Complexity**: Added elements only after base routine was automatic
5. **Holistic Benefits**: Recognized system-wide life improvements beyond original goals

**Habit Formation Principles Demonstrated:**
- **21-Day Myth Debunked**: True automatization took 13-16 weeks
- **Disruption Resilience**: Systems must account for imperfect conditions
- **Incremental Expansion**: Build complexity on stable foundations
- **Identity Integration**: Success correlated with shift from doing to being
            """,
            reference_reasoning="Temporal analysis requiring phase identification, turning point recognition, and habit formation principle extraction",
            evaluation_criteria={
                "phase_identification": "Must clearly delineate different phases of development",
                "turning_points": "Should identify key moments that changed trajectory",
                "principle_extraction": "Must derive transferable habit formation insights"
            }
        ))

        # Strategic Reasoning Test
        samples.append(TestSample(
            test_id="SR001",
            category=TestCategory.STRATEGIC_REASONING,
            difficulty_level="expert",
            input_content="""
**VaultMind Strategic Crossroads - September 2025**

**Current Position:**
- 500 active users, 40% weekly retention, $15K MRR
- Team of 6: 4 engineers, 1 designer, 1 PM
- 12 months runway at current burn rate
- Strong technical product with 85% user satisfaction
- Core differentiation: adaptive AI prompting for personal knowledge

**Market Dynamics:**
- Personal knowledge management space heating up (Notion raised $275M, Obsidian at 1M users)
- AI integration becoming table stakes, not differentiator
- Enterprise customers asking for team features (we're individual-focused)
- New competitors launching weekly with VC backing

**Growth Options Under Consideration:**

**Option A: Enterprise Pivot**
- Build team collaboration features, admin dashboards, SSO
- Target 10-100 person knowledge teams
- Potential for $500-5000/month contracts
- Requires 6+ months development, hiring sales team

**Option B: Consumer AI Assistant**
- Double-down on AI personalization, add voice interface, mobile app
- Target broader consumer market beyond knowledge workers
- Freemium model with AI-powered features as premium
- Leverage existing AI expertise, faster time to market

**Option C: Vertical Specialization**
- Focus on specific professional niches (researchers, consultants, writers)
- Build industry-specific templates, integrations, workflows
- Premium pricing for specialized value
- Requires domain expertise acquisition

**Resource Constraints:**
- Limited runway requires revenue growth within 9 months
- Team expertise heavily weighted toward AI/ML, light on sales/marketing
- Existing user base loves current product but growth has plateaued
- Competing priorities: new features vs. technical debt vs. growth initiatives

**Key Uncertainties:**
- Will AI commoditization erode our competitive advantage?
- Can we successfully sell to enterprises with current team composition?
- Is the consumer market big enough for sustainable growth?
- How defensible is our position against well-funded competitors?
            """,
            task_description="Conduct comprehensive strategic analysis to recommend the optimal path forward, including risk assessment, resource allocation, and implementation roadmap",
            expected_output="""
**Strategic Analysis & Recommendation:**

**Market Position Assessment:**
- **Strengths**: Strong technical foundation, proven AI differentiation, satisfied user base
- **Vulnerabilities**: Resource constraints, plateau growth, increasing competition
- **Critical Window**: 9-month revenue growth requirement creates urgency

**Option Analysis:**

**Option A (Enterprise): High Risk, High Reward**
- *Pros*: Large contract values, defensible moat, less consumer competition
- *Cons*: Major capability gap (sales, enterprise features), longest time-to-revenue
- *Resource Fit*: Poor - requires skills/roles we don't have
- *Risk Level*: High - betting entire runway on unproven enterprise capability

**Option B (Consumer AI): Moderate Risk, Moderate Reward** 
- *Pros*: Leverages core AI strengths, broader market, faster deployment
- *Cons*: Highly competitive space, freemium model challenges, commoditization risk
- *Resource Fit*: Good - builds on existing technical capabilities
- *Risk Level*: Moderate - market size uncertainty but capability alignment

**Option C (Vertical): Lower Risk, Focused Reward**
- *Pros*: Premium pricing, defensible positioning, builds on current users
- *Cons*: Limited market size, requires domain expertise acquisition
- *Resource Fit*: Moderate - leverages current product, requires new expertise
- *Risk Level*: Lower - extension of current model with proven user base

**Strategic Recommendation: Hybrid Approach - "Vertical First, Consumer Next"**

**Phase 1 (Months 1-6): Vertical Specialization Focus**
1. **Target Selection**: Researchers/academics (highest current user concentration)
2. **Feature Development**: Research-specific templates, citation management, collaboration tools
3. **Pricing Strategy**: $49/month premium tier for professional features
4. **Go-to-Market**: Partner with academic institutions, research communities

**Phase 2 (Months 7-12): Consumer AI Expansion** 
1. **Leverage Learnings**: Use vertical insights to inform consumer features
2. **Technical Investment**: Voice interface, mobile app using refined AI engine
3. **Market Entry**: Freemium model with enterprise-proven AI features as premium

**Resource Allocation Strategy:**
- **60% Engineering**: Core vertical features + AI optimization
- **20% Product/Design**: User experience for target vertical
- **20% Growth/Partnerships**: Academic/professional community building

**Risk Mitigation:**
1. **Revenue Timeline**: Vertical focus should generate revenue within 4-6 months
2. **Market Hedge**: Positions for consumer expansion if vertical succeeds
3. **Competitive Defense**: Deep vertical expertise harder to replicate
4. **Team Leverage**: Maximizes existing AI capabilities while building new competencies

**Success Metrics & Milestones:**
- **Month 3**: 50+ researcher beta users, $5K MRR from premium features
- **Month 6**: $25K MRR, partnership with 2 academic institutions
- **Month 9**: Consumer MVP launch, $40K MRR total
- **Month 12**: 100K+ consumer users, $75K MRR, Series A positioning

**Strategic Rationale:**
This approach minimizes execution risk while maximizing learning opportunities. Vertical success provides revenue stability and market proof points for eventual consumer expansion. It plays to our strengths while building capabilities needed for broader market success.
            """,
            reference_reasoning="Expert strategic analysis requiring multi-dimensional option evaluation, risk assessment, resource allocation, and implementation planning",
            evaluation_criteria={
                "strategic_thinking": "Must demonstrate systems-level strategic reasoning",
                "option_evaluation": "Should systematically assess all three options",
                "resource_realism": "Must account for team capabilities and constraints",
                "implementation_detail": "Should provide specific, actionable roadmap"
            }
        ))

        # Sentiment Analysis Test
        samples.append(TestSample(
            test_id="SA001",
            category=TestCategory.SENTIMENT_ANALYSIS,
            difficulty_level="moderate",
            input_content="""
**Personal Reflection - Career Transition Thoughts (August 2025)**

I've been thinking a lot about the job offer from TechCorp. On paper, it's everything I thought I wanted - 40% salary increase, prestigious company, challenging technical problems. But something feels off.

The interview process was impressive but also exhausting. Five rounds, multiple technical challenges, panel interviews. Everyone seemed competent but also... intense? Stressed? It's hard to put my finger on it. The office tour showed beautiful facilities but people looked kind of burned out.

Part of me is excited about the technical challenges they described. Working on AI systems at scale, access to incredible datasets, smart colleagues to learn from. This could be the career acceleration I've been looking for.

But I keep thinking about my current job. Sure, the pay isn't amazing and sometimes I feel underutilized, but I have genuine friendships with my coworkers. I leave work at 5 PM and don't think about it until the next morning. My manager actually cares about my development and personal life.

My partner thinks I should take the TechCorp offer - "it's time to level up," she says. My parents are excited about the prestige. But my gut keeps saying something isn't right. Maybe I'm just scared of change? Or maybe I'm finally learning to value work-life balance over prestige and money.

I have until Friday to decide. The logical part of my brain says take it. The emotional part says stay. I wish I could fast-forward five years and see which choice leads to greater satisfaction.
            """,
            task_description="Analyze the emotional complexity and psychological patterns in this career decision reflection",
            expected_output="""
**Sentiment Analysis: Complex Career Decision Processing**

**Overall Emotional Tone: Conflicted Ambivalence with Underlying Anxiety**

**Primary Emotional Themes:**

**1. Opportunity Excitement vs. Intuitive Resistance**
- *Positive Sentiment*: "excited about technical challenges," "career acceleration," "smart colleagues"
- *Negative Counters*: "something feels off," "gut keeps saying something isn't right"
- *Pattern*: Logical attraction competing with emotional/intuitive warnings

**2. External Validation vs. Internal Values Conflict**
- *External Pressure*: Partner encourages "level up," parents excited about prestige
- *Internal Priorities*: Values work-life balance, genuine friendships, caring management
- *Tension*: Social expectations pulling against personal well-being preferences

**3. Fear of Change vs. Growth Stagnation**
- *Change Anxiety*: "Maybe I'm just scared of change?"
- *Stagnation Concerns*: "underutilized," need for "career acceleration"
- *Self-Doubt*: Questioning whether resistance is fear-based or wisdom-based

**Psychological Processing Patterns:**

**Decision-Making Style:**
- **Analytical Processing**: Systematic pro/con evaluation (salary, prestige, challenges vs. balance, relationships)
- **Somatic Intelligence**: Strong reliance on "gut feelings" and intuitive responses
- **Social Calibration**: Heavily influenced by partner and family opinions
- **Values Clarification**: Using decision process to understand personal priorities

**Stress Indicators:**
- **Time Pressure**: Friday deadline creating urgency stress
- **Identity Questions**: "Who am I?" - prestige-seeker vs. balance-prioritizer
- **Regret Anticipation**: "fast-forward five years" wish indicates fear of wrong choice

**Emotional Sophistication Markers:**
- **Meta-Cognitive Awareness**: Recognizing multiple decision factors and internal conflicts
- **Values Evolution**: "finally learning to value work-life balance" suggests maturity development
- **Intuitive Validation**: Trusting non-rational decision inputs alongside logical analysis

**Underlying Concerns:**
- **Burnout Risk**: Observing "stressed," "burned out" culture signals
- **Authenticity**: Current job allows genuine self-expression and relationships
- **Life Integration**: Conflict between career ambition and holistic well-being

**Emotional Trajectory Prediction:**
- **Short-term**: Decision anxiety will peak before Friday deadline
- **Resolution Style**: Likely to make choice based on values clarification rather than pure logic
- **Post-Decision**: Whichever choice made, will need to actively manage regret/doubt thoughts

**Key Insight**: This reflection reveals someone in active values evolution - transitioning from external success metrics to internal satisfaction measures. The decision difficulty stems from this values transition rather than the specific job characteristics.
            """,
            reference_reasoning="Moderate sentiment analysis requiring emotional nuance recognition, psychological pattern identification, and values conflict analysis",
            evaluation_criteria={
                "emotional_nuance": "Must identify complex, competing emotions",
                "psychological_insight": "Should recognize decision-making patterns and values evolution",
                "predictive_elements": "Must provide insights about likely emotional trajectory"
            }
        ))

        return samples

    def _initialize_scoring_functions(self) -> Dict[ScoringApproach, Callable]:
        """Initialize different scoring approaches."""
        return {
            ScoringApproach.RULE_BASED: self._rule_based_scoring,
            ScoringApproach.AI_JUDGE: self._ai_judge_scoring,
            ScoringApproach.HUMAN_REFERENCE: self._human_reference_scoring,
            ScoringApproach.HYBRID: self._hybrid_scoring,
            ScoringApproach.SELF_CONSISTENCY: self._self_consistency_scoring
        }

    def run_comprehensive_evaluation(self,
                                   strategies: List[PromptingStrategy],
                                   test_categories: List[TestCategory] = None,
                                   scoring_approach: ScoringApproach = ScoringApproach.AI_JUDGE,
                                   custom_samples: List[TestSample] = None) -> EvaluationReport:
        """
        Run comprehensive evaluation of prompting strategies.

        Args:
            strategies: List of prompting strategies to test
            test_categories: Categories of tests to run (all by default)
            scoring_approach: How to score the results
            custom_samples: Additional test samples to include

        Returns:
            EvaluationReport with detailed results and analysis
        """
        # Filter test samples by category
        test_samples = custom_samples or self.test_samples
        if test_categories:
            test_samples = [s for s in test_samples if s.category in test_categories]

        # Initialize report
        report = EvaluationReport(
            test_suite_name=f"VaultMind Evaluation - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            strategies_tested=strategies,
            total_tests=len(test_samples) * len(strategies)
        )

        # Run tests for each strategy
        for strategy in strategies:
            strategy_results = []

            for sample in test_samples:
                print(f"Testing {strategy.value} on {sample.test_id}...")

                # Simulate strategy execution (in real implementation, this would call actual prompting systems)
                actual_output = self._simulate_strategy_execution(strategy, sample)

                # Score the result
                result = self._score_result(sample, strategy, actual_output, scoring_approach)
                strategy_results.append(result)

            report.results_by_strategy[strategy] = strategy_results

        # Calculate summary statistics
        report.summary_statistics = self._calculate_summary_statistics(report)

        return report

    def _simulate_strategy_execution(self, strategy: PromptingStrategy, sample: TestSample) -> str:
        """
        Simulate executing different prompting strategies.
        In real implementation, this would call actual prompting systems.
        """
        # This is a simulation - in real implementation, you'd call:
        # - zero_shot_analyzer.analyze()
        # - one_shot_analyzer.analyze()
        # - multi_shot_analyzer.analyze()
        # - dynamic_builder.build_prompt()
        # - chain_of_thought_analyzer.analyze()

        # For demo purposes, return strategy-specific mock outputs
        strategy_outputs = {
            PromptingStrategy.ZERO_SHOT: f"Zero-shot analysis of {sample.test_id}: Basic analysis with direct reasoning.",
            PromptingStrategy.ONE_SHOT: f"One-shot analysis of {sample.test_id}: Analysis guided by single example pattern.",
            PromptingStrategy.MULTI_SHOT: f"Multi-shot analysis of {sample.test_id}: Analysis informed by multiple example patterns showing variation.",
            PromptingStrategy.DYNAMIC: f"Dynamic analysis of {sample.test_id}: Personalized analysis adapted to user patterns and context.",
            PromptingStrategy.CHAIN_OF_THOUGHT: f"Chain-of-thought analysis of {sample.test_id}: Step-by-step reasoning through: Observe → Hypothesize → Evidence → Patterns → Compare → Infer → Validate → Synthesize → Implications → Recommend."
        }

        return strategy_outputs.get(strategy, f"Unknown strategy analysis of {sample.test_id}")

    def _score_result(self,
                     sample: TestSample,
                     strategy: PromptingStrategy,
                     actual_output: str,
                     scoring_approach: ScoringApproach) -> TestResult:
        """Score a test result using the specified approach."""
        result = TestResult(
            test_id=sample.test_id,
            strategy=strategy,
            actual_output=actual_output,
            processing_time=0.5,  # Mock timing
            token_count=len(actual_output.split()) * 1.3,  # Mock token count
        )

        # Apply scoring function
        scoring_function = self.scoring_functions[scoring_approach]
        scores = scoring_function(sample, actual_output)
        result.scores = scores

        return result

    def _rule_based_scoring(self, sample: TestSample, actual_output: str) -> Dict[EvaluationMetric, float]:
        """Rule-based scoring using heuristics and keyword matching."""
        scores = {}

        # Simple heuristics (in real implementation, these would be more sophisticated)
        output_length = len(actual_output.split())
        expected_length = len(sample.expected_output.split())

        # Accuracy based on keyword overlap
        actual_keywords = set(re.findall(r'\b\w{4,}\b', actual_output.lower()))
        expected_keywords = set(re.findall(r'\b\w{4,}\b', sample.expected_output.lower()))
        keyword_overlap = len(actual_keywords & expected_keywords) / max(len(expected_keywords), 1)
        scores[EvaluationMetric.ACCURACY] = min(keyword_overlap * 10, 10)

        # Completeness based on length ratio
        length_ratio = min(output_length / max(expected_length, 1), 2.0)
        scores[EvaluationMetric.COMPLETENESS] = min(length_ratio * 5, 10)

        # Clarity based on sentence structure
        sentences = len(re.split(r'[.!?]+', actual_output))
        avg_sentence_length = output_length / max(sentences, 1)
        clarity_score = 10 if 10 <= avg_sentence_length <= 25 else max(5, 10 - abs(avg_sentence_length - 17.5) / 2)
        scores[EvaluationMetric.CLARITY] = clarity_score

        # Default scores for other metrics
        scores[EvaluationMetric.RELEVANCE] = 7.0
        scores[EvaluationMetric.INSIGHT_DEPTH] = 6.5
        scores[EvaluationMetric.ACTIONABILITY] = 6.0

        return scores

    def _ai_judge_scoring(self, sample: TestSample, actual_output: str) -> Dict[EvaluationMetric, float]:
        """AI judge scoring using AI-generated evaluation prompts."""
        scores = {}

        # In real implementation, these would be actual AI calls
        # For demo, return realistic mock scores with some variation
        base_scores = {
            EvaluationMetric.ACCURACY: 7.5,
            EvaluationMetric.COMPLETENESS: 8.0,
            EvaluationMetric.RELEVANCE: 7.8,
            EvaluationMetric.CLARITY: 8.2,
            EvaluationMetric.INSIGHT_DEPTH: 6.8,
            EvaluationMetric.ACTIONABILITY: 7.2
        }

        # Add some realistic variation based on content characteristics
        complexity_factor = sample.get_complexity_score()
        output_quality_factor = len(actual_output) / 200  # Rough quality indicator

        for metric, base_score in base_scores.items():
            # Adjust based on complexity and output quality
            adjusted_score = base_score
            if complexity_factor > 3.0:  # Complex samples are harder
                adjusted_score -= 0.5
            if output_quality_factor > 1.5:  # Longer outputs might be better
                adjusted_score += 0.3

            scores[metric] = max(0, min(10, adjusted_score))

        return scores

    def _human_reference_scoring(self, sample: TestSample, actual_output: str) -> Dict[EvaluationMetric, float]:
        """Human reference scoring using pre-scored reference outputs."""
        # Mock implementation - in real system, would compare against human-scored references
        return {
            EvaluationMetric.ACCURACY: 8.0,
            EvaluationMetric.COMPLETENESS: 7.5,
            EvaluationMetric.RELEVANCE: 8.2,
            EvaluationMetric.CLARITY: 7.8,
            EvaluationMetric.INSIGHT_DEPTH: 7.0,
            EvaluationMetric.ACTIONABILITY: 7.5
        }

    def _hybrid_scoring(self, sample: TestSample, actual_output: str) -> Dict[EvaluationMetric, float]:
        """Hybrid scoring combining multiple approaches."""
        rule_scores = self._rule_based_scoring(sample, actual_output)
        ai_scores = self._ai_judge_scoring(sample, actual_output)

        # Weight different approaches
        hybrid_scores = {}
        for metric in EvaluationMetric:
            if metric in rule_scores and metric in ai_scores:
                # 30% rule-based, 70% AI judge
                hybrid_scores[metric] = 0.3 * rule_scores[metric] + 0.7 * ai_scores[metric]
            elif metric in ai_scores:
                hybrid_scores[metric] = ai_scores[metric]
            elif metric in rule_scores:
                hybrid_scores[metric] = rule_scores[metric]

        return hybrid_scores

    def _self_consistency_scoring(self, sample: TestSample, actual_output: str) -> Dict[EvaluationMetric, float]:
        """Self-consistency scoring by running multiple times and measuring agreement."""
        # Mock implementation - would run same prompt multiple times and measure consistency
        return {
            EvaluationMetric.CONSISTENCY: 8.5,
            EvaluationMetric.ACCURACY: 7.8,
            EvaluationMetric.COMPLETENESS: 7.5,
            EvaluationMetric.RELEVANCE: 8.0,
            EvaluationMetric.CLARITY: 7.9,
            EvaluationMetric.INSIGHT_DEPTH: 7.2
        }

    def _calculate_summary_statistics(self, report: EvaluationReport) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics for the evaluation."""
        stats = {}

        # Overall performance by strategy
        strategy_performance = {}
        for strategy, results in report.results_by_strategy.items():
            valid_results = [r for r in results if not r.error_occurred]
            if valid_results:
                overall_scores = [r.get_overall_score() for r in valid_results]
                strategy_performance[strategy.value] = {
                    "mean_score": statistics.mean(overall_scores),
                    "std_dev": statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    "min_score": min(overall_scores),
                    "max_score": max(overall_scores),
                    "success_rate": len(valid_results) / len(results)
                }

        stats["strategy_performance"] = strategy_performance

        # Metric analysis
        metric_analysis = {}
        for metric in EvaluationMetric:
            metric_scores = []
            for results in report.results_by_strategy.values():
                for result in results:
                    if metric in result.scores:
                        metric_scores.append(result.scores[metric])

            if metric_scores:
                metric_analysis[metric.value] = {
                    "mean": statistics.mean(metric_scores),
                    "std_dev": statistics.stdev(metric_scores) if len(metric_scores) > 1 else 0,
                    "distribution": Counter([round(score) for score in metric_scores])
                }

        stats["metric_analysis"] = metric_analysis

        # Test category performance
        category_performance = defaultdict(list)
        for strategy, results in report.results_by_strategy.items():
            for result in results:
                # Find the test sample to get category
                sample = next((s for s in self.test_samples if s.test_id == result.test_id), None)
                if sample:
                    category_performance[sample.category.value].append(result.get_overall_score())

        category_stats = {}
        for category, scores in category_performance.items():
            if scores:
                category_stats[category] = {
                    "mean_score": statistics.mean(scores),
                    "difficulty_correlation": "mock_correlation_data"  # Would be actual correlation in real implementation
                }

        stats["category_performance"] = category_stats

        # Rankings
        stats["strategy_rankings"] = [(s.value, score) for s, score in report.calculate_strategy_rankings()]

        return stats

    def generate_detailed_report(self, report: EvaluationReport) -> str:
        """Generate a detailed text report of the evaluation results."""
        lines = [
            f"# VaultMind Prompting Strategy Evaluation Report",
            f"Generated: {report.generated_at}",
            f"Test Suite: {report.test_suite_name}",
            f"Total Tests: {report.total_tests}",
            "",
            "## Executive Summary",
            ""
        ]

        # Strategy rankings
        rankings = report.calculate_strategy_rankings()
        lines.extend([
            "### Strategy Performance Rankings:",
            ""
        ])

        for i, (strategy, score) in enumerate(rankings, 1):
            lines.append(f"{i}. **{strategy.value}**: {score:.2f}/10")

        lines.extend(["", "## Detailed Results", ""])

        # Detailed results by strategy
        for strategy, results in report.results_by_strategy.items():
            lines.extend([
                f"### {strategy.value} Results:",
                ""
            ])

            valid_results = [r for r in results if not r.error_occurred]
            if valid_results:
                avg_score = statistics.mean([r.get_overall_score() for r in valid_results])
                lines.append(f"**Average Score**: {avg_score:.2f}/10")
                lines.append(f"**Tests Completed**: {len(valid_results)}/{len(results)}")
                lines.append("")

                # Top performing tests
                top_results = sorted(valid_results, key=lambda x: x.get_overall_score(), reverse=True)[:3]
                lines.append("**Top Performing Tests**:")
                for result in top_results:
                    lines.append(f"- {result.test_id}: {result.get_overall_score():.2f}/10")
                lines.append("")

        # Summary statistics
        if report.summary_statistics:
            lines.extend([
                "## Statistical Analysis",
                ""
            ])

            if "metric_analysis" in report.summary_statistics:
                lines.append("### Metric Performance:")
                lines.append("")
                for metric, stats in report.summary_statistics["metric_analysis"].items():
                    lines.append(f"**{metric}**: {stats['mean']:.2f} ± {stats['std_dev']:.2f}")
                lines.append("")

        return "\n".join(lines)

    def export_results(self, report: EvaluationReport, format: str = "json") -> str:
        """Export evaluation results in specified format."""
        if format == "json":
            # Convert enum objects to strings for JSON serialization
            serializable_report = {
                "test_suite_name": report.test_suite_name,
                "strategies_tested": [s.value for s in report.strategies_tested],
                "total_tests": report.total_tests,
                "generated_at": report.generated_at,
                "results_by_strategy": {
                    strategy.value: [
                        {
                            "test_id": result.test_id,
                            "strategy": result.strategy.value,
                            "overall_score": result.get_overall_score(),
                            "scores": {metric.value: score for metric, score in result.scores.items()},
                            "processing_time": result.processing_time,
                            "token_count": result.token_count,
                            "error_occurred": result.error_occurred
                        }
                        for result in results
                    ]
                    for strategy, results in report.results_by_strategy.items()
                },
                "summary_statistics": report.summary_statistics
            }
            return json.dumps(serializable_report, indent=2)

        elif format == "csv":
            # Generate CSV format (simplified)
            csv_lines = ["test_id,strategy,overall_score,accuracy,completeness,clarity,processing_time"]

            for strategy, results in report.results_by_strategy.items():
                for result in results:
                    scores = result.scores
                    csv_lines.append(
                        f"{result.test_id},{result.strategy.value},{result.get_overall_score():.2f},"
                        f"{scores.get(EvaluationMetric.ACCURACY, 0):.2f},"
                        f"{scores.get(EvaluationMetric.COMPLETENESS, 0):.2f},"
                        f"{scores.get(EvaluationMetric.CLARITY, 0):.2f},"
                        f"{result.processing_time:.3f}"
                    )

            return "\n".join(csv_lines)

        else:
            return self.generate_detailed_report(report)

    def create_custom_test_sample(self,
                                 category: TestCategory,
                                 difficulty: str,
                                 input_content: str,
                                 task_description: str,
                                 expected_output: str,
                                 evaluation_criteria: Dict[str, str] = None) -> TestSample:
        """Create a custom test sample for specific evaluation needs."""
        return TestSample(
            test_id=f"CUSTOM_{uuid.uuid4().hex[:8].upper()}",
            category=category,
            difficulty_level=difficulty,
            input_content=input_content,
            task_description=task_description,
            expected_output=expected_output,
            evaluation_criteria=evaluation_criteria or {}
        )


# Convenience functions for common evaluation scenarios
def quick_strategy_comparison(strategies: List[PromptingStrategy] = None,
                            test_categories: List[TestCategory] = None) -> EvaluationReport:
    """Quick function to compare prompting strategies on standard test suite."""
    framework = TestFramework()

    strategies = strategies or [
        PromptingStrategy.ZERO_SHOT,
        PromptingStrategy.ONE_SHOT,
        PromptingStrategy.MULTI_SHOT,
        PromptingStrategy.CHAIN_OF_THOUGHT
    ]

    return framework.run_comprehensive_evaluation(
        strategies=strategies,
        test_categories=test_categories,
        scoring_approach=ScoringApproach.AI_JUDGE
    )


def evaluate_single_strategy(strategy: PromptingStrategy,
                           custom_samples: List[TestSample] = None) -> EvaluationReport:
    """Evaluate a single prompting strategy in detail."""
    framework = TestFramework()
    return framework.run_comprehensive_evaluation(
        strategies=[strategy],
        custom_samples=custom_samples,
        scoring_approach=ScoringApproach.HYBRID
    )


def create_evaluation_benchmark(name: str,
                              samples: List[TestSample]) -> str:
    """Create a reusable evaluation benchmark from custom samples."""
    framework = TestFramework()

    # Save benchmark (in real implementation, would persist to file/database)
    benchmark_data = {
        "name": name,
        "created_at": datetime.now().isoformat(),
        "samples": [
            {
                "test_id": s.test_id,
                "category": s.category.value,
                "difficulty": s.difficulty_level,
                "input_content": s.input_content,
                "task_description": s.task_description,
                "expected_output": s.expected_output
            }
            for s in samples
        ]
    }

    return json.dumps(benchmark_data, indent=2)


# Example usage and demonstration
if __name__ == "__main__":
    print("=== VaultMind Evaluation Framework Demo ===\n")

    # Initialize framework
    framework = TestFramework()

    print("1. Test Sample Overview:")
    print(f"   Total test samples: {len(framework.test_samples)}")

    category_counts = Counter(s.category.value for s in framework.test_samples)
    difficulty_counts = Counter(s.difficulty_level for s in framework.test_samples)

    print(f"   Categories: {dict(category_counts)}")
    print(f"   Difficulty levels: {dict(difficulty_counts)}")
    print()

    print("2. Sample Test Case Detail (Strategic Reasoning):")
    strategic_test = next(s for s in framework.test_samples if s.test_id == "SR001")
    print(f"   ID: {strategic_test.test_id}")
    print(f"   Category: {strategic_test.category.value}")
    print(f"   Difficulty: {strategic_test.difficulty_level}")
    print(f"   Task: {strategic_test.task_description[:100]}...")
    print(f"   Input length: {len(strategic_test.input_content.split())} words")
    print(f"   Expected output length: {len(strategic_test.expected_output.split())} words")
    print()

    print("3. Quick Strategy Comparison:")
    strategies_to_test = [
        PromptingStrategy.ZERO_SHOT,
        PromptingStrategy.ONE_SHOT,
        PromptingStrategy.MULTI_SHOT,
        PromptingStrategy.CHAIN_OF_THOUGHT
    ]

    report = quick_strategy_comparison(strategies_to_test)

    print(f"   Evaluation completed: {report.total_tests} total tests")
    print(f"   Strategies tested: {[s.value for s in report.strategies_tested]}")
    print()

    print("4. Strategy Performance Rankings:")
    rankings = report.calculate_strategy_rankings()
    for i, (strategy, score) in enumerate(rankings, 1):
        print(f"   {i}. {strategy.value}: {score:.2f}/10")
    print()

    print("5. Detailed Performance Analysis:")
    for strategy, results in list(report.results_by_strategy.items())[:2]:  # Show first 2
        valid_results = [r for r in results if not r.error_occurred]
        if valid_results:
            avg_score = statistics.mean([r.get_overall_score() for r in valid_results])
            print(f"   {strategy.value}:")
            print(f"     Average score: {avg_score:.2f}/10")
            print(f"     Success rate: {len(valid_results)}/{len(results)} tests")

            # Show metric breakdown
            metric_scores = defaultdict(list)
            for result in valid_results:
                for metric, score in result.scores.items():
                    metric_scores[metric].append(score)

            print(f"     Top metrics:")
            top_metrics = sorted(metric_scores.items(),
                               key=lambda x: statistics.mean(x[1]), reverse=True)[:3]
            for metric, scores in top_metrics:
                print(f"       {metric.value}: {statistics.mean(scores):.1f}/10")
    print()

    print("6. Judge Prompt Example:")
    judge_builder = JudgePromptBuilder()
    sample_test = framework.test_samples[0]  # Simple extraction test

    accuracy_prompt = judge_builder.build_judge_prompt(
        EvaluationMetric.ACCURACY,
        sample_test,
        "Mock AI output for demonstration"
    )

    print(f"   Judge prompt for {EvaluationMetric.ACCURACY.value}:")
    print(f"   Length: {len(accuracy_prompt)} characters")
    print(f"   Preview: {accuracy_prompt[:200]}...")
    print()

    print("7. Custom Test Sample Creation:")
    custom_sample = framework.create_custom_test_sample(
        category=TestCategory.PATTERN_RECOGNITION,
        difficulty="moderate",
        input_content="Sample content for pattern recognition testing",
        task_description="Identify recurring patterns in the content",
        expected_output="Expected pattern analysis results",
        evaluation_criteria={"pattern_identification": "Must identify at least 3 patterns"}
    )

    print(f"   Created custom sample: {custom_sample.test_id}")
    print(f"   Category: {custom_sample.category.value}")
    print(f"   Complexity score: {custom_sample.get_complexity_score()}")
    print()

    print("8. Export Capabilities:")
    json_export = framework.export_results(report, "json")
    print(f"   JSON export length: {len(json_export):,} characters")

    text_report = framework.generate_detailed_report(report)
    print(f"   Text report length: {len(text_report):,} characters")
    print(f"   Report preview: {text_report[:300]}...")

    print("\n=== Evaluation Framework Ready ===")
    print("Key capabilities:")
    print("• Comprehensive test suite with 6 diverse samples")
    print("• Multiple scoring approaches (rule-based, AI judge, hybrid)")
    print("• Statistical analysis and strategy rankings")
    print("• Custom test sample creation")
    print("• Export to JSON, CSV, and detailed text reports")
    print("• Judge prompt generation for AI-assisted evaluation")
