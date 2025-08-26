# RTFC Framework in VaultMind: A 3-Minute Video Script

## Video Title: "RTFC Framework: Structured AI Prompting for VaultMind CLI"

---

## INTRODUCTION (0:00 - 0:30)

**[Screen: VaultMind logo and title]**

**Narrator**: "Welcome to VaultMind! Today we're diving into the RTFC framework - a powerful approach to AI prompt engineering that makes our CLI tool smarter and more reliable. RTFC stands for Role, Task, Format, and Context, and it's the backbone of how VaultMind communicates with AI to analyze your Obsidian vault."

**[Screen: RTFC acronym appears with brief definitions]**

---

## WHAT IS RTFC? (0:30 - 1:15)

**[Screen: Split view showing messy prompt vs structured RTFC prompt]**

**Narrator**: "Traditional AI prompts can be messy and unpredictable. But RTFC gives us structure. Let's break it down:"

**[Screen: Each component highlights as mentioned]**

- **Role**: "First, we define WHO the AI is. In VaultMind, this could be a Knowledge Analyst, Vault Explorer, or Personal Assistant. Each role has specific expertise."

- **Task**: "Next, we specify WHAT the AI should do - analyze, summarize, extract insights, or find connections between your notes."

- **Format**: "Then we determine HOW the response should look - JSON for data, Markdown for readability, or conversational for chat."

- **Context**: "Finally, we provide WHAT the AI knows - your vault info, relevant notes, and user preferences."

---

## CODE IMPLEMENTATION (1:15 - 2:15)

**[Screen: Code editor showing system_user_prompts.py]**

**Narrator**: "Let's see RTFC in action. In our `system_user_prompts.py` file, we've created classes that implement each component:"

**[Screen: Highlight RTFCRole enum]**
"The RTFCRole enum defines five specialist roles - from Knowledge Analyst to Note Synthesizer."

**[Screen: Highlight SystemPrompt class]**
"Our SystemPrompt class builds the AI's personality using the RTFC structure. Notice how each section has its own method - this keeps our code clean and modular."

**[Screen: Show _build_role_section method]**
"For example, the role section gives detailed expertise definitions. A Knowledge Analyst specializes in pattern recognition, while a Personal Assistant focuses on natural conversation."

**[Screen: Show RTFCPromptBuilder class]**
"The RTFCPromptBuilder makes it easy to create common prompt types. Need to analyze a vault? One method call gives you both system and user prompts, perfectly structured."

---

## PRACTICAL EXAMPLE (2:15 - 2:45)

**[Screen: Terminal showing VaultMind command]**

**Narrator**: "Here's the magic in action. When you run:"

**[Screen: Command line]**
```bash
vaultmind analyze ~/MyVault
```

**[Screen: Split showing the generated prompts]**

"VaultMind uses RTFC to create precise prompts. The system prompt establishes a Knowledge Analyst role, defines the analysis task, specifies Markdown format, and includes your vault context. The user prompt then makes the specific request."

**[Screen: Show AI response]**

"The result? Consistent, high-quality analysis that understands your vault structure and provides actionable insights."

---

## BENEFITS & WRAP-UP (2:45 - 3:00)

**[Screen: Benefits list with checkmarks]**

**Narrator**: "RTFC brings three key benefits to VaultMind:"

- "**Consistency**: Every interaction follows the same structured approach"
- "**Flexibility**: Easy to customize roles, tasks, and formats for different needs"  
- "**Maintainability**: Clean, modular code that's easy to extend and debug"

**[Screen: VaultMind logo with call-to-action]**

"Ready to experience structured AI prompting? Try VaultMind today and see how RTFC makes your Obsidian vault more intelligent and accessible. Thanks for watching!"

---

## TECHNICAL NOTES FOR VIDEO PRODUCTION

### Visual Elements:
- **Code highlighting**: Use syntax highlighting for Python code segments
- **Split screens**: Show prompt structure alongside generated output
- **Animations**: Smooth transitions between RTFC components
- **Screenshots**: Real terminal interactions and vault analysis results

### Timing Breakdown:
- **Introduction**: 30 seconds - Hook and overview
- **Framework explanation**: 45 seconds - Deep dive into RTFC
- **Code walkthrough**: 60 seconds - Implementation details
- **Live example**: 30 seconds - Practical demonstration
- **Conclusion**: 15 seconds - Benefits and call-to-action

### Key Code Snippets to Show:

1. **RTFC Enum Definitions**:
```python
class RTFCRole(Enum):
    KNOWLEDGE_ANALYST = "knowledge_analyst"
    VAULT_EXPLORER = "vault_explorer"
    # ...
```

2. **System Prompt Building**:
```python
def _build_role_section(self) -> str:
    role_definitions = {
        RTFCRole.KNOWLEDGE_ANALYST: """You are VaultMind's Knowledge Analyst..."""
    }
```

3. **Quick Builder Usage**:
```python
system, user = RTFCPromptBuilder.create_analysis_prompt(vault_info)
```

### Sample Terminal Commands:
```bash
# Show these commands in the demo
vaultmind analyze ~/Vault --format json
vaultmind chat "What are my main research themes?" 
vaultmind summarize --notes "project-*.md"
```

### Voice-Over Tips:
- **Pace**: Moderate speed with clear enunciation
- **Tone**: Professional but approachable, like a technical tutorial
- **Emphasis**: Stress key terms (RTFC, VaultMind, structured prompting)
- **Pauses**: Allow time for viewers to absorb code snippets

---

## FOLLOW-UP CONTENT IDEAS

This 3-minute video could be part of a series:

1. **"VaultMind Basics: Setting Up Your First Analysis"**
2. **"Advanced RTFC: Custom Roles and Tasks"** 
3. **"Integration Guide: VaultMind + Obsidian Workflows"**
4. **"Extending VaultMind: Writing Custom Prompt Templates"**

Each video would build on the RTFC foundation established here, creating a comprehensive learning path for VaultMind users.
