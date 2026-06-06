import json
import subprocess
import time

from evaluation.test_evaluation import load_dataset
from chatbot.app import get_response
from rag.retriever import retrieve_context
from deepeval.test_case import LLMTestCase
from evaluation.deepeval_runner import run_deepeval


# -----------------------------
# PROMPTFOO RUNNER
# -----------------------------
def run_promptfoo(input_text, output_text):

    payload = {
        "input": input_text,
        "output": output_text
    }

    with open("temp_promptfoo.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)

    result = subprocess.run(
        ["npx", "promptfoo", "eval", "--config", "promptfooconfig.yaml"],
        capture_output=True,
        text=True,
        shell=True,
        encoding="utf-8",
        errors="ignore"
    )

    return result.stdout


# -----------------------------
# NORMALIZE YAML ASSERTIONS
# -----------------------------
def normalize_expected_output(assert_val):

    if assert_val is None:
        return "General response"

    # regex-style list
    if isinstance(assert_val, list):
        try:
            if len(assert_val) > 0 and isinstance(assert_val[0], dict):
                return assert_val[0].get("value", str(assert_val))
        except:
            return str(assert_val)

    # dict-style assert
    if isinstance(assert_val, dict):
        return str(assert_val.get("value", assert_val))

    return str(assert_val)


# -----------------------------
# CORE PIPELINE
# -----------------------------
def run_combined_eval(test_cases):

    results = []
    deepeval_cases = []

    for idx, tc in enumerate(test_cases):

        print(f"\n🔥 Processing test case {idx+1}/{len(test_cases)}")

        input_text = tc["input"]

        # ---- RAG CONTEXT ----
        context = retrieve_context(input_text)

        # ---- LLM CALL ----
        output = get_response(input_text)

        # 🔥 sleep to avoid Groq rate limit spikes
        time.sleep(2)

        # ---- PROMPTFOO ----
        promptfoo_result = run_promptfoo(input_text, output)

        # 🔥 extra safety sleep (Node + LLM combo protection)
        time.sleep(1)

        # ---- NORMALIZE ASSERT ----
        expected_output = normalize_expected_output(tc.get("expected_output"))

        # ---- DEEPEVAL CASE ----
        deepeval_case = LLMTestCase(
            input=input_text,
            actual_output=output,
            expected_output=expected_output,
            retrieval_context=context
        )

        deepeval_cases.append(deepeval_case)

        results.append({
            "input": input_text,
            "output": output,
            "promptfoo": promptfoo_result
        })

    # ---- FINAL DEEPEVAL RUN ----
    deepeval_results = run_deepeval(deepeval_cases)

    return results, deepeval_results


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    dataset_files = [
        "datasets/normal.yaml",
        "datasets/vague.yaml",
        "datasets/multi_step.yaml",
        "datasets/blabber.yaml"
    ]

    all_test_cases = []

    # 🔥 CAP TEST CASES TO 5 TOTAL
    MAX_CASES = 3
    count = 0

    for file in dataset_files:

        print(f"\n📂 Loading dataset: {file}")
        dataset = load_dataset(file)

        for item in dataset:

            if count >= MAX_CASES:
                break

            all_test_cases.append({
                "input": item["vars"]["input"],
                "expected_output": item.get("assert")
            })

            count += 1

        if count >= MAX_CASES:
            break

    print(f"\n✅ Total test cases loaded: {len(all_test_cases)}")

    # ---- RUN PIPELINE ----
    results, deepeval_results = run_combined_eval(all_test_cases)

    # ---- OUTPUT ----
    print("\n=== PROMPTFOO RESULTS ===")
    print(results)

    print("\n=== DEEPEVAL RESULTS ===")
    print(deepeval_results)