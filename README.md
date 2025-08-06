# VaultMind 🧠

**An AI-powered CLI assistant for your Obsidian vault**

VaultMind is a lightweight, terminal-based AI tool to interact with your Obsidian vault using Gemini 1.5 Flash. Summarize, analyze, and chat with your notes — all from the command line.

---

## ✨ Features

* **Vault Analysis** – Get insights on structure, themes, and content
* **Smart Summarization** – Extract key points, themes, and actions
* **Intelligent Chat** – Ask questions using RAG (Retrieval-Augmented Generation)
* **Context-Aware** – Finds and cites relevant notes
* **Fast & Efficient** – Powered by Gemini 1.5 Flash

---

## 🎯 Goal

Build a CLI tool that enables:

* **System + User Prompts** – Define assistant behavior and queries
* **Tuning Parameters** – Control creativity and output length
* **Structured Output** – JSON/YAML/Markdown formatting
* **Function Calling** – Trigger Python functions from AI
* **RAG** – Retrieve context from your markdown notes

---

## 🛠️ Tech Stack

* **Language**: Python 3.8+
* **LLM**: Gemini 1.5 Flash
* **CLI Framework**: Typer
* **RAG Engine**: LangChain or LlamaIndex
* **Vector Store**: FAISS or ChromaDB
* **Markdown Parsing**: frontmatter, markdown2

---

## 💡 Examples

### Analyzing

```bash
$ python main.py analyze ~/Vault
{ "total_notes": 120, "main_themes": ["research", "ideas"] }
```

### Summarizing

```bash
$ python main.py summarize ~/Vault "project-plan.md"
- Deadline: Oct 15
- Tech: React + Firebase
- Action: Set up CI/CD
```

### Chatting

```bash
$ python main.py chat ~/Vault "Thoughts on learning Rust?"
From "rust-notes.md": You find it challenging but rewarding...
```


## 🌟 License

MIT - see [LICENSE](LICENSE)


**Made with ❤️ for the Obsidian community**

*Transform your notes into conversations, insights into actions.*

