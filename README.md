# 🧠 AI LLM Testing Suite (RAG + DeepEval + Promptfoo)

A practical AI QA framework for testing LLM-based applications using RAG (Retrieval-Augmented Generation), LLM-as-Judge evaluation, and hybrid testing (rule-based + semantic scoring).

This project demonstrates how a QA Engineer can transition into **AI Testing / LLM Evaluation Engineering** — testing a real AI career guidance chatbot end-to-end.

---

## 🤔 What Is This Project? (Plain English)

Imagine you've built a chatbot that helps people figure out their career path in tech. Someone types "I like design but also coding idk help" — and the chatbot needs to give a useful, structured response.

Now the question is: **how do you know if the chatbot is actually doing a good job?**

You can't just check if the word "Python" appears in the output. You need to ask: *Was the response helpful? Was it relevant? Did it stay on topic? Did it ask clarifying questions when the input was vague?*

That's exactly what this project does. It builds a **testing and evaluation system** for an AI chatbot — measuring quality in a way that goes beyond simple pass/fail checks.

---

## 🚀 Project Overview

### The Problem with Testing AI

Traditional software testing is straightforward:

```
Input → Function → Expected Output → Assert Equal ✅ or ❌
```

**AI breaks this entirely** because:

- The same question can produce 10 different but equally valid answers
- "Correctness" is about meaning, not exact wording
- A response can be factually off but grammatically perfect

### How This Project Solves It

We combine three tools that each attack the problem differently:

| Approach | What It Does | Think of It As |
|----------|--------------|----------------|
| **Promptfoo** | Rule-based pass/fail checks (regex, keywords) | The strict gatekeeper |
| **DeepEval** | Semantic scoring using another LLM as the judge | The thoughtful reviewer |
| **RAG** | Grounds the chatbot's answers in real documents | The fact-checker |

Together, they give you a **complete picture** of how your AI is performing.

---

## 🤖 What the Chatbot Does

The chatbot being tested is an **AI career assistant** built for people exploring tech careers. It:

- Answers questions about career paths (frontend, backend, data science, etc.)
- Asks clarifying questions if your input is vague or unclear
- Handles multi-part questions by answering all parts
- Ignores irrelevant noise (weather jokes, off-topic rambling)
- Gives structured, practical advice — not generic filler

It runs on **Groq's API** using the **Llama 3.1 8B Instant** model (fast and free-tier friendly).

---

## 🏗️ How It All Fits Together

Here's the real flow — this is a **batch testing pipeline**, not a live chatbot interface:

```
Test Datasets (YAML files)
         ↓
   Test Runner loads each case
         ↓
   RAG Retriever finds relevant docs (FAISS)
         ↓
   Context injected into prompt
         ↓
   Groq / Llama generates a response
         ↓
      Evaluation Layer
      ├── Promptfoo  → Did it match the rules? (Pass/Fail)
      └── DeepEval   → Is it actually good? (Score 0–1)
         ↓
   Report generated with insights
```

Think of it like running a test suite with `pytest` — except instead of checking return values, you're checking whether an AI response is *sensible and grounded*.

---

## 📁 Project Structure

```
LLM-QA-Evaluation-Framework/
│
├── chatbot/
│   └── app.py                  # Career chatbot — calls Groq API with RAG context
│
├── rag/
│   ├── docs.py                 # The knowledge base (career-related documents)
│   └── retriever.py            # FAISS index + embeddings — finds relevant docs per query
│
├── datasets/
│   ├── normal.yaml             # Clear, well-formed questions
│   ├── vague.yaml              # Ambiguous / unclear inputs
│   ├── multi_step.yaml         # Complex multi-part queries
│   └── blabber.yaml            # Noisy, irrelevant, or rambling inputs
│
├── evaluation/
│   ├── test_evaluation.py      # Main DeepEval runner (semantic scoring)
│   ├── combined_eval.py        # Hybrid runner — Promptfoo rules + DeepEval scores together
│   ├── judge_prompt.py         # The rubric given to the LLM judge (what makes a good answer)
│   └── llm_judge.py            # Sends the rubric + response to the judge LLM, gets a score
│
├── reports/
│   └── generate_report.py      # Compiles all results into a summary report
│
├── .deepeval/                  # DeepEval config and cached evaluation state
├── promptfooconfig.yaml        # Promptfoo config: model, system prompt, and test dataset links
├── explanation.txt             # Deep-dive explanation of every file and concept in the project
├── .gitignore
└── README.md
```

---

## 🧠 Core Concepts Explained Simply

### 1. 🗂️ RAG — Teaching the Chatbot What It's Allowed to Know

**RAG = Retrieval-Augmented Generation**

Without RAG, an LLM just *guesses* based on everything it was trained on. It might hallucinate facts, give outdated info, or go completely off-topic.

With RAG, you give the model a **controlled knowledge base** — a set of documents it must draw from when answering.

```
❌ Pure LLM  →  "I think Python is great for everything!" (might hallucinate)
✅ RAG        →  "Based on the provided context, Python suits data roles..."
```

**How it works in this project:**

1. Career-related documents are stored in `rag/docs.py`
2. `rag/retriever.py` converts those docs into vectors (mathematical representations of meaning) and stores them in a **FAISS index**
3. When a question comes in, FAISS finds the most *semantically similar* documents — not keyword matching, but meaning matching
4. Those documents are injected into the prompt as context

```python
context = retrieve_context(user_input)

prompt = f"""
Context:
{context}

User:
{user_input}
"""
```

**Why this matters for testing:** RAG gives us a "source of truth." We can now measure whether the chatbot *stuck to what it was told* — that's the Faithfulness metric.

---

### 2. ⚖️ LLM-as-Judge — Using AI to Grade AI

Instead of writing code like `assert "Python" in output`, DeepEval sends the response to another LLM with a grading rubric, asking: *"Is this actually a good answer?"*

```
❌ Old way:  "Must contain the word 'Python'"       (brittle, misses meaning)
✅ New way:  "Is this response helpful and relevant?" (understands intent)
```

The grading criteria in this project (`judge_prompt.py`) tells the judge LLM things like:

- Did the chatbot help the user figure out a career direction?
- Did it ask clarifying questions when the input was vague?
- Did it address all parts of a multi-part question?

This approach catches failures that regex never could.

---

### 3. 📏 The Three Metrics Used

```python
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    ToxicityMetric(threshold=0.0)
]
```

| Metric | What It Asks | Why It Matters |
|--------|-------------|----------------|
| **Answer Relevancy** | Does the response actually address what was asked? | Catches off-topic or evasive answers |
| **Faithfulness** | Did the chatbot stick to the retrieved context, or make stuff up? | Directly measures hallucination |
| **Toxicity** | Is the output safe and appropriate? | Safety gate — must score 0 |

> ⚠️ **Important:** Faithfulness *requires* `retrieval_context` to be passed in. Without RAG context, DeepEval has no source of truth to compare against and the evaluation will fail.

---

### 4. 🧪 Promptfoo — The Fast Rule-Based Gate

Promptfoo is a **command-line tool** (not a Python library) that runs your test datasets against the chatbot and checks the outputs against rules you define in YAML.

It answers questions like:
- Did the response mention at least one career option?
- Did it include actionable advice keywords?
- Did it follow the expected response format?

```yaml
# Example from datasets/vague.yaml
- vars:
    input: "I like design but also coding idk help"
  assert:
    - type: regex
      value: "(design|coding|frontend|backend|interest|choose)"
```

Promptfoo is fast and deterministic — great for catching obvious failures before spending money on LLM-judge evaluation.

---

### 5. 🔀 Hybrid Evaluation — Best of Both Worlds

`combined_eval.py` runs **both** tools together, giving you two layers of confidence:

| Layer | Tool | Type | What It Catches |
|-------|------|------|-----------------|
| Layer 1 | Promptfoo | Rule-based | Format violations, missing keywords |
| Layer 2 | DeepEval | Semantic | Vague answers, hallucination, poor reasoning |

A response can pass Promptfoo (contains the right keywords) and still fail DeepEval (the overall answer was unhelpful). Both signals together give you the full picture.

---

## 📂 Test Dataset Design

We intentionally test with **messy, real-world inputs** — not clean, perfectly-worded questions. Because real users are messy.

| Dataset | What It Simulates | Example Input |
|---------|------------------|---------------|
| `normal.yaml` | Clear, direct questions | "What skills do I need for frontend development?" |
| `vague.yaml` | Ambiguous, unclear intent | "I like design but also coding idk help" |
| `multi_step.yaml` | Complex multi-part questions | "Compare frontend vs backend and tell me which pays more" |
| `blabber.yaml` | Noisy or irrelevant input | "lol what even is coding, also what's the weather today" |

This mirrors how real users actually behave — and exposes weaknesses that clean test cases never would.

---

## ⚙️ Setup & Running the Project

### Prerequisites

- Python 3.9+
- Node.js (for Promptfoo)
- A **Groq API key** (free at [console.groq.com](https://console.groq.com))
- An **OpenAI API key** (DeepEval uses it for the judge LLM — or configure an alternative)

### Step 1 — Clone the repo

```bash
git clone https://github.com/Yahya123-hub/LLM-QA-Evaluation-Framework-with-Promptfoo-and-Deepeval.git
cd LLM-QA-Evaluation-Framework-with-Promptfoo-and-Deepeval
```

### Step 2 — Set up Python environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install faiss-cpu sentence-transformers deepeval groq
```

### Step 3 — Install Promptfoo (CLI tool via npm)

```bash
npm install -g promptfoo
```

### Step 4 — Set your API keys

```bash
# On Mac/Linux
export GROQ_API_KEY=your_groq_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here

# On Windows (Command Prompt)
set GROQ_API_KEY=your_groq_api_key_here
set OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5 — Run the evaluations

**Run DeepEval only (semantic scoring):**
```bash
python -m evaluation.test_evaluation
```

**Run Promptfoo only (rule-based checks):**
```bash
promptfoo eval
```

**Run the hybrid evaluation (both together — recommended):**
```bash
python -m evaluation.combined_eval
```

**Generate the summary report:**
```bash
python reports/generate_report.py
```

---

## 📊 Results & Findings

These are the actual results from running the framework before and after prompt improvements:

```
Pass rate:              75%
Hallucination rate:     18%
Multi-step failure:     45%
Clarification failure:  60%   ← biggest weakness found
Recovery success:       70%
```

### 📉 Key Observations

- The model struggled most with **vague queries** — it would give a generic answer instead of asking a clarifying question
- **Multi-step queries** were often only partially answered (the model would address the first part and ignore the rest)
- **Hallucination** occurred mostly when the RAG retriever returned weak or irrelevant context
- Toxicity was never an issue (0% failures)

### 🔧 Improvements Applied After Testing

Based on these findings, the following changes were made and re-tested:

- Added explicit clarification instructions to the system prompt
- Refined the prompt to require answering *all parts* of multi-step questions
- Expanded the RAG knowledge base to cover more career topics
- Added more vague-input test cases to the dataset for better coverage

### ✅ After Improvements

- Clarification failure rate dropped significantly
- Relevancy scores improved across all dataset types
- Hallucination rate reduced thanks to stronger RAG grounding

---

## ⚠️ Common Issues

**Rate limit errors from DeepEval:**
```
RateLimitError: insufficient_quota
```
DeepEval calls an LLM to judge each test case, which costs API credits. If you hit limits:
- Reduce the number of test cases per run
- Run datasets one at a time instead of all together
- Add a small `time.sleep()` between test cases

**Promptfoo not recognized as a command:**
Make sure Node.js is installed and `npm install -g promptfoo` completed without errors. Run `promptfoo --version` to verify.

**FAISS index errors on first run:**
The FAISS index is built on first use from `rag/docs.py`. If it fails, check that `faiss-cpu` and `sentence-transformers` installed correctly.

---

## 🧭 What This Project Demonstrates

- ✅ How to transition from traditional QA thinking to AI QA thinking
- ✅ Building a RAG pipeline from scratch (FAISS + sentence-transformers)
- ✅ Using LLM-as-judge for semantic evaluation (DeepEval)
- ✅ Rule-based LLM testing with Promptfoo
- ✅ Hybrid evaluation combining both approaches
- ✅ Designing test datasets that simulate real-world messy users
- ✅ Measuring hallucination, relevancy, and safety systematically

---

## 🔥 Why This Matters

Companies are shipping AI features faster than they can test them. Failures in AI aren't like traditional bugs — they're subtle, semantic, and hard to catch with conventional tools.

This project shows a complete workflow for:

- **Testing AI like an engineer** — not just vibes-checking outputs manually
- **Measuring quality beyond pass/fail** — understanding *why* a model fails
- **Iterating on prompts with data** — not guessing, but measuring

---

## 📖 Want to Understand the Code Deeper?

Check out [`explanation.txt`](./explanation.txt) — it's a detailed, plain-English walkthrough of every file in this project, every concept used, and how they all connect. It was written specifically to help you understand the *reasoning* behind each decision, not just the code itself.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Groq API** | LLM inference (Llama 3.1 8B Instant) |
| **FAISS** | Vector similarity search for RAG |
| **sentence-transformers** | Converting text to embeddings |
| **DeepEval** | Semantic evaluation + LLM-as-judge |
| **Promptfoo** | Rule-based CLI evaluation framework |
| **Python** | Core language for all evaluation logic |
