# Multi-Shot Prompting: When Examples Rule: 3-Minute Video Script

**Duration**: ~3 minutes  
**Target Audience**: AI practitioners and prompt engineers  
**Goal**: Explain multi-shot prompting advantages, example curation strategies, and when to use it over zero-shot/one-shot

---

## Opening Hook (0:00 - 0:20)

**[Screen: Progression visual showing 0 examples → 1 example → 3-5 examples with increasingly sophisticated outputs]**

**Narrator**: "Zero-shot gives you speed. One-shot gives you consistency. But what if your task is so complex that even one example isn't enough? That's where multi-shot prompting shines - and today I'll show you when more examples make all the difference."

**[Animation showing: Complex Task + Multiple Examples → Nuanced, High-Quality Analysis]**

**Narrator**: "In VaultMind, we've discovered that some analysis tasks need to see pattern variations, edge cases, and complexity progressions to truly excel."

---

## The Multi-Shot Advantage (0:20 - 1:05)

**[Screen: Side-by-side comparison of outputs from zero-shot, one-shot, and multi-shot approaches]**

**Narrator**: "Here's the key insight: some tasks have too much variation for a single example to cover. Take sentiment analysis of complex personal journals."

**[Show examples on screen:]**
- Simple: "Had a good day" 
- Complex: "Mixed emotions about career transition with family implications"
- Expert: "Grief-integrated growth with generational wisdom processing"

**Narrator**: "One example can't show the AI how to handle simple satisfaction, complex emotional contradictions, AND expert-level psychological processing. But 4-5 examples can."

**[Code demonstration showing MultiShotAnalyzer with progressive complexity examples]**

**Narrator**: "VaultMind's MultiShotAnalyzer uses what we call 'progressive complexity curation' - examples that build from simple to expert levels, showing the AI how to scale its analysis sophistication."

---

## Example Curation Strategies (1:05 - 1:50)

**[Screen: Code showing different selection strategies]**

**Narrator**: "The secret sauce isn't just having multiple examples - it's choosing the RIGHT examples. We've implemented five curation strategies:"

**[Visual list with animations:]**

1. **Progressive**: "Simple → Complex, building sophistication"
2. **Diverse**: "Maximum variation in approaches and styles"  
3. **Contextual**: "Examples most similar to your target content"
4. **Comprehensive**: "Cover all major edge cases and variations"
5. **Balanced**: "Optimal mix of complexity and diversity"

**[Live demo showing strategy comparison]**

```python
# Progressive: 3 examples showing increasing complexity
# Diverse: 4 examples from different note types
# Contextual: 4 examples matching target content style
```

**Narrator**: "Notice how each strategy produces different example sets for the same task. Progressive builds understanding step-by-step. Diverse shows multiple valid approaches. Contextual ensures relevance."

---

## When Multi-Shot Outperforms (1:50 - 2:30)

**[Screen: Decision matrix showing use cases]**

**Narrator**: "When should you choose multi-shot over simpler approaches? Three key scenarios:"

**[Visual hierarchy appearing:]**

**Use Multi-Shot When:**
- **Task complexity is high** (nuanced analysis, creative interpretation)
- **Input variation is extreme** (handling simple AND complex content)  
- **Quality consistency is critical** (professional reports, research analysis)
- **Pattern learning is needed** (AI must understand variations)

**[Real example demonstration]**

**Narrator**: "Here's a real example: analyzing a 6-month personal development journal entry. Zero-shot misses emotional nuances. One-shot handles moderate complexity but struggles with psychological sophistication. Multi-shot nails the complexity progression, emotional integration patterns, AND growth trajectory analysis."

**[Show actual output comparison on screen]**

**Narrator**: "The multi-shot version identifies 'grief-integrated growth' and 'generational wisdom processing' - insights that simpler approaches completely miss."

---

## Curation Best Practices (2:30 - 2:50)

**[Screen: Example repository structure in VaultMind]**

**Narrator**: "Effective multi-shot prompting requires strategic example curation. Our three key principles:"

**[Animated principles appearing:]**

1. **Complexity Progression**: "Always include examples building from simple to target complexity level"

2. **Edge Case Coverage**: "Show boundary conditions - what does 'good analysis' look like at different extremes?"

3. **Pattern Variation**: "Multiple valid approaches for the same task - teach flexibility, not rigid templates"

**[Code snippet showing example repository structure]**

**Narrator**: "In VaultMind, we maintain 3-5 curated examples per task-type combination. Each example includes not just input-output, but explanation of WHY it's effective."

---

## Trade-offs & Closing (2:50 - 3:00)

**[Screen: Cost-benefit comparison chart]**

**Narrator**: "Multi-shot prompting isn't free. Longer prompts mean higher token costs and slower processing. But when task complexity demands it, the quality improvement is dramatic."

**[Summary slide with key takeaways]**

**Narrator**: "Choose multi-shot when you need the AI to understand pattern variations and handle complexity gracefully. Your examples become the AI's training data - curate them thoughtfully."

**[End screen: "VaultMind: Zero-Shot Speed, One-Shot Consistency, Multi-Shot Sophistication"]**

---

## Technical Notes for Implementation

### Key Demonstrations to Show:

1. **Progressive Complexity Examples**: Show the same task (sentiment analysis) with simple → complex → expert examples
2. **Strategy Comparison**: Live demo of different selection strategies producing different example sets
3. **Quality Difference**: Side-by-side outputs showing multi-shot handling nuanced content better
4. **Repository Structure**: How examples are organized and curated in the codebase

### Visual Elements:

- **Complexity progression charts** showing example difficulty scaling
- **Strategy decision flowchart** for choosing selection approaches
- **Live code execution** showing the analyzer selecting different example sets
- **Quality comparison tables** showing output sophistication differences

### Script Timing Breakdown:
- **Hook**: 20 seconds (establish the progression concept)
- **Advantage explanation**: 45 seconds (why multi-shot matters)
- **Curation strategies**: 45 seconds (how to choose examples)
- **Usage scenarios**: 40 seconds (when to use multi-shot)
- **Best practices**: 20 seconds (curation principles)
- **Trade-offs**: 10 seconds (honest assessment)

### Key Messages:
1. Multi-shot excels when tasks require understanding pattern variations
2. Example curation strategy determines effectiveness
3. Progressive complexity helps AI scale analysis sophistication  
4. Higher token cost justified by quality improvement for complex tasks

---

## Extended Content Ideas

### Deep-Dive Version (5-7 minutes):
- **Example Quality Metrics**: How to evaluate whether examples are effective
- **A/B Testing Framework**: Comparing example sets for effectiveness
- **Custom Example Creation**: Workshop on writing effective examples
- **Token Optimization**: Keeping multi-shot prompts efficient while comprehensive

### Technical Implementation Focus:
- **Repository Architecture**: Database design for example management
- **Dynamic Selection Algorithms**: How the strategy algorithms work internally
- **Performance Benchmarking**: Measuring multi-shot vs simpler approaches
- **Example Versioning**: Managing and updating example quality over time

### User Journey Focus:
- **Real-World Scenarios**: Actual user cases where multi-shot made the difference
- **ROI Analysis**: When the extra cost/complexity pays off
- **Integration Patterns**: How multi-shot fits into larger AI workflows
- **User Feedback**: How to collect and integrate user insights on example quality

---

## Production Notes

### Code Demonstrations:
```python
# Show this progression in the video
analyzer = MultiShotAnalyzer()

# Simple case - might work with zero-shot
simple_note = "Had a productive day today."

# Complex case - needs multi-shot
complex_note = "Six months into this growth journey, seeing unexpected patterns..."

# Show how multi-shot handles complexity better
prompt = analyzer.analyze_note(complex_note, AnalysisTask.ANALYZE_SENTIMENT, 
                              selection_strategy="progressive", max_examples=4)
```

### Visual Storytelling:
- Use consistent color coding (zero-shot: blue, one-shot: green, multi-shot: purple)
- Show example selection happening in real-time
- Use progressive disclosure for complex concepts
- Include actual output comparisons, not just descriptions

### Audience Considerations:
- **Technical practitioners**: Include implementation details and code
- **Business stakeholders**: Focus on ROI and quality improvements  
- **Researchers**: Emphasize the pattern learning and sophistication aspects
- **Educators**: Show the pedagogical aspects of teaching AI through examples
