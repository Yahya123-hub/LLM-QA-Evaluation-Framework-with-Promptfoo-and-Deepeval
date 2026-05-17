# 🧠 AI LLM Testing Suite (RAG + DeepEval + Promptfoo)

A practical AI QA framework for testing LLM-based applications using RAG (Retrieval-Augmented Generation), LLM-as-Judge evaluation, and hybrid testing (rule-based + semantic scoring).

This project demonstrates how a QA Engineer can transition into AI Testing / LLM Evaluation Engineering.

---

## 🚀 Project Overview

### Traditional QA focuses on:

* deterministic outputs
* exact assertions

### LLMs break this model because:

* outputs are non-deterministic
* correctness is semantic, not exact

👉 This project solves that by combining:

| Approach  | Purpose                                      |
| --------- | -------------------------------------------- |
| Promptfoo | Pass/fail validation (regex, rules)          |
| DeepEval  | Semantic scoring (LLM-as-Judge)              |
| RAG       | Grounding responses to prevent hallucination |

---

## 🏗️ Architecture

```
User Input
    ↓
Retriever (FAISS + embeddings)
    ↓
Relevant Context (RAG)
    ↓
LLM (Groq / Llama)
    ↓
Generated Response
    ↓
Evaluation Layer
   ├── Promptfoo (rule-based)
   └── DeepEval (semantic scoring)
```

---

## 📁 Project Structure

```
ai-llm-testing-suite/
│
├── chatbot/
│   ├── app.py              # LLM interface (Groq)
│
├── rag/
│   ├── docs.py             # Knowledge base
│   ├── retriever.py        # FAISS + embeddings
│
├── datasets/
│   ├── normal.yaml
│   ├── vague.yaml
│   ├── multi_step.yaml
│   ├── blabber.yaml
│
├── evaluation/
│   ├── test_evaluation.py  # DeepEval runner
    |-combined_eval.py # Hybrid: Promptfoo + DeepEval
    |-judge_prompt.py # Evaluation rubric
    |-llm_judge.py # Custom LLM judge
│
├── reports/
│   └── generate_report.py
│
└── README.md
```

---

## 🧠 Core Concepts

### 1. RAG (Retrieval-Augmented Generation)

Instead of letting the model guess:

❌ Pure LLM → hallucination risk
✅ RAG → grounded responses

#### How it works:

* Convert documents → embeddings
* Store in FAISS index
* Retrieve top-k relevant docs per query
* Inject into prompt as context

```python
context = retrieve_context(user_input)

prompt = f"""
Context:
{context}

User:
{user_input}
"""
```

---

### 2. LLM-as-Judge (DeepEval)

Instead of keyword checks:

❌ "must contain 'Python'"
✅ "Is the answer relevant and helpful?"

#### Evaluation intent:

```python
expected_output = """
The response should guide the user toward a career path
and ask clarifying questions if the input is vague.
"""
```

---

### 3. Metrics Used

```python
metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.7),
    ToxicityMetric(threshold=0.0)
]
```

| Metric       | Purpose                                  |
| ------------ | ---------------------------------------- |
| Relevancy    | Does answer match the question?          |
| Faithfulness | Is answer grounded in retrieved context? |
| Toxicity     | Safety check                             |

---

### ⚠️ Important (RAG Requirement)

Faithfulness requires:

```python
retrieval_context = retrieve_context(input)
```

Without this → ❌ evaluation fails

---

## 🧪 Test Design Strategy

We simulate real-world messy users.

### 📂 Dataset Types

| Dataset         | Purpose                  |
| --------------- | ------------------------ |
| normal.yaml     | Clear questions          |
| vague.yaml      | Ambiguous inputs         |
| multi_step.yaml | Complex queries          |
| blabber.yaml    | Noisy / irrelevant input |

---

### Example Test Case

```yaml
- vars:
    input: "I like design but also coding idk help"
  assert:
    - type: regex
      value: "(design|coding|frontend|backend|interest|choose)"
```

---

## 🔗 Hybrid Testing Approach

We combine:

### ✅ Promptfoo (Rule-based)

* Regex validation
* Pass/fail signals

### ✅ DeepEval (Semantic)

* Scoring (0–1)
* LLM reasoning

---

## 📊 Final Report Example

```
Pass rate: 75%
Hallucination rate: 18%
Multi-step failure: 45%
Clarification failure: 60%
Recovery success: 70%
```

---

## 📉 Key Observations

* Model struggles with vague queries
* Often fails to ask clarification questions
* Multi-step queries not handled properly
* Some hallucination when context is weak

---

## 🔧 Improvements Applied

* Prompt refinement
* Added clarification instructions
* Introduced RAG grounding
* Improved dataset coverage

---

## 🧠 Key Insights (Most Important)

The model failed ~60% of vague queries due to lack of clarification prompting.

### After improvements:

* clarification failures reduced significantly
* relevancy scores improved
* hallucination reduced via RAG

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install faiss-cpu sentence-transformers deepeval
```

---

### 2. Run evaluation

```bash
python -m evaluation.test_evaluation
```

---

## ⚠️ Note on API Usage

DeepEval uses LLMs for scoring.

If you see:

```
RateLimitError: insufficient_quota
```

👉 Add billing or reduce usage:

* limit test cases
* run in batches

---

## 🧭 What This Project Demonstrates

* Transition from QA → AI QA
* Understanding of LLM limitations
* RAG pipeline implementation
* Semantic evaluation (LLM-as-judge)
* Real-world test design for AI systems

---

## 🔥 Why This Matters

Modern systems ship AI features fast — but:

* they are not deeply tested
* failures are subtle and semantic

👉 This project shows how to:

* test AI like a QA engineer
* evaluate beyond pass/fail
* measure real model quality


