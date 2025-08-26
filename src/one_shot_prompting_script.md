# One-Shot Prompting vs Zero-Shot: 3-Minute Video Script

**Duration**: ~3 minutes  
**Target Audience**: Developers and AI practitioners interested in prompt engineering  
**Goal**: Explain one-shot prompting, compare with zero-shot, and demonstrate VaultMind's implementation

---

## Opening Hook (0:00 - 0:25)

**[Screen: Side-by-side comparison showing zero-shot vs one-shot prompts]**

**Narrator**: "What's the difference between telling an AI what to do versus showing it how to do it? That's the key distinction between zero-shot and one-shot prompting. Today, I'll show you when examples make all the difference in AI analysis quality."

**[Animation showing: Task Description Only → AI → Result vs Task + Example → AI → Better Result]**

**Narrator**: "In VaultMind, we've implemented both approaches. Let me show you why sometimes one example is worth a thousand instructions."

---

## Zero-Shot vs One-Shot Comparison (0:25 - 1:10)

**[Screen: Code comparison showing both approaches]**

**Narrator**: "Zero-shot prompting relies purely on task descriptions. We tell the AI: 'Extract action items from this meeting.' The AI uses its training to understand what that means."

**[Show zero-shot prompt example]**

**Narrator**: "One-shot prompting adds a concrete example. We show the AI exactly what good action item extraction looks like, then ask it to follow the same pattern."

**[Show one-shot prompt with example template]**

```python
# Zero-shot approach
"Extract action items from this meeting content"

# One-shot approach  
"Extract action items like this example:
Input: 'Sarah will email the client by Friday...'
Output: '• Sarah: Email client (Due: Friday, High Priority)'
Now extract from: [your content]"
```

**Narrator**: "The example provides structure, format, and quality expectations that pure instruction can't match."

---

## VaultMind's One-Shot Implementation (1:10 - 2:00)

**[Screen: Code walkthrough of OneShotAnalyzer class]**

**Narrator**: "VaultMind's OneShotAnalyzer uses curated example templates for different combinations of note types, analysis tasks, and output formats."

**[Show example template structure]**

**Narrator**: "Each template contains four key elements:"

**[Animation highlighting each element:]**

1. **Example Input**: "Real note content similar to what users will analyze"
2. **Example Output**: "High-quality analysis result showing the desired pattern"  
3. **Explanation**: "Why this analysis is effective"
4. **Metadata**: "Note type, task, and format specifications"

**[Screen: Live demo showing template matching]**

**Narrator**: "When you analyze a meeting note, the system automatically selects a meeting-specific example. Analyzing a journal entry? It picks a journal example. This contextual matching ensures relevance."

---

## Live Demonstration (2:00 - 2:40)

**[Screen: Running the one_shot_prompting.py demo]**

**Narrator**: "Let's see this in action with a team retrospective note."

**[Code execution showing:]**

```python
analyzer = OneShotAnalyzer()
retrospective_note = NoteContent(
    title="Sprint 8 Retrospective",
    content="What went well... Action items for next sprint...",
    note_type=NoteType.MEETING
)

prompt = analyzer.analyze_note(retrospective_note, AnalysisTask.FIND_ACTION_ITEMS)
```

**[Screen: Generated prompt with example]**

**Narrator**: "Notice how the system includes a complete meeting example showing exactly how to format action items - with priorities, owners, and deadlines clearly structured."

**[Quick comparison demo]**

**Narrator**: "When we compare prompt lengths: our zero-shot version is 800 characters, while one-shot is 2,400 characters. That extra 1,600 characters is pure guidance - showing rather than just telling."

---

## When to Use Each Approach (2:40 - 2:55)

**[Screen: Decision matrix showing use cases]**

**Narrator**: "When should you choose one-shot over zero-shot?"

**[Visual list appearing:]**

**Use One-Shot When:**
- **Format consistency is critical** (reports, structured data)
- **Quality standards are high** (professional outputs)
- **Complex tasks** require clear examples
- **Token budget allows** for longer prompts

**Use Zero-Shot When:**
- **Speed is essential** (quick analysis)
- **Token efficiency matters** (cost optimization)
- **Tasks are simple** and self-explanatory
- **Flexibility is key** (avoiding example bias)

**Narrator**: "VaultMind supports both because different scenarios call for different approaches."

---

## Closing & Key Takeaways (2:55 - 3:00)

**[Screen: Summary slide with key points]**

**Narrator**: "One-shot prompting bridges the gap between zero-shot flexibility and few-shot complexity. One good example can dramatically improve AI output consistency and quality."

**[Call-to-action display]**

**Narrator**: "Try both approaches in VaultMind and see the difference examples make in your note analysis."

**[End screen: "VaultMind: Zero-Shot Speed, One-Shot Precision"]**

---

## Technical Notes for Implementation

### Key Code Demonstrations:

1. **ExampleTemplate Structure**: Show the four-component template design
2. **Automatic Template Matching**: Demonstrate intelligent example selection
3. **Prompt Length Comparison**: Visualize the size difference between approaches
4. **Quality Difference**: Show side-by-side output comparisons

### Visual Elements:

- **Split-screen comparisons** showing zero-shot vs one-shot prompts
- **Template structure diagrams** illustrating the example components
- **Live code execution** showing the analyzer in action
- **Decision flowchart** for choosing between approaches

### Script Timing Breakdown:
- **Hook**: 25 seconds (establish the comparison)
- **Concept comparison**: 45 seconds (explain the difference)
- **Implementation**: 50 seconds (how VaultMind does it)
- **Live demo**: 40 seconds (seeing it work)
- **Usage guidance**: 15 seconds (when to use which)
- **Closing**: 5 seconds (key takeaway)

### Key Messages:
1. One-shot prompting adds concrete examples to guide AI behavior
2. Examples provide structure and quality standards that instructions alone can't match
3. Template matching ensures contextually relevant examples
4. Choose based on your priorities: speed vs consistency vs quality

---

## Extended Demo Ideas

If you want to expand the video or create additional content:

### Advanced Topics (For 5-minute version):
- **Custom Example Creation**: Show how to add domain-specific templates
- **Example Quality**: Demonstrate what makes a good vs poor example
- **Batch Processing**: Show analyzing multiple notes with consistent examples
- **Performance Metrics**: Compare accuracy between zero-shot and one-shot

### Technical Deep-Dive (For Developer-Focused Version):
- **Template Storage and Retrieval**: Database design for example management
- **Example Versioning**: How to update and improve templates over time
- **A/B Testing**: Framework for comparing example effectiveness
- **Token Optimization**: Strategies for keeping examples concise yet effective

### User Story Focus (For Product Demo):
- **Real User Scenarios**: Show actual use cases where examples matter
- **Before/After Results**: Demonstrate quality improvements with examples
- **User Feedback**: Include testimonials about consistency improvements
- **ROI Analysis**: Show time savings from better initial results

---

## Production Notes

### Filming Considerations:
- Use split-screen effectively to show comparisons
- Highlight code sections clearly as you discuss them
- Keep example content readable on screen
- Use consistent color coding (zero-shot in blue, one-shot in green)

### Post-Production:
- Add smooth transitions between concepts
- Include text overlays for key takeaways
- Use consistent animation style for diagrams
- Ensure code examples are clearly visible at all video resolutions
