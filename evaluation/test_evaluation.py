import os
import time
import yaml
from dotenv import load_dotenv
from rag.retriever import retrieve_context
from reports.generate_report import generate_report
from evaluation.groq_judge import GroqJudge
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric)
from chatbot.app import get_response

load_dotenv()


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_test_cases(dataset, model_name):
    test_cases = []

    for item in dataset:
        input_text = item["vars"]["input"]

        response = get_response(input_text, model_name)
        time.sleep(1)

        expected = "Relevant career guidance"

        if "assert" in item:
            expected = str(item["assert"])

        test_cases.append(
            LLMTestCase(
                input=input_text,
                actual_output=response,
                expected_output=expected,
                retrieval_context=retrieve_context(input_text),
                additional_metadata={
                    "model": model_name
                }
            )
        )

    return test_cases


def run_evaluation():

    dataset_files = [
        "datasets/normal.yaml",
        "datasets/vague.yaml",
        "datasets/multi_step.yaml",
        "datasets/blabber.yaml"
    ]

    models = [
        "llama-3.1-8b-instant",       
        "llama-3.3-70b-versatile"  
    ]

    all_test_cases = []

    for model_name in models:
        print(f"\nRunning evaluation for model: {model_name}")

        for file in dataset_files:
            print(f"\nLoading dataset: {file}")
            dataset = load_dataset(file)

            test_cases = build_test_cases(dataset, model_name)
            all_test_cases.extend(test_cases)

    all_test_cases = all_test_cases[:5]

    print(f"\nRunning evaluation on {len(all_test_cases)} test cases...\n")

    judge_model = GroqJudge(model="llama-3.3-70b-versatile")

    metrics = [
        AnswerRelevancyMetric(
            threshold=0.7,
            model=judge_model
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=judge_model
        ),
        ToxicityMetric(
            threshold=0.0,
            model=judge_model
        )
    ]

    results = evaluate(
        test_cases=all_test_cases,
        metrics=metrics
    )

    generate_report(results)

    print("\nRESULTS:")
    print(results)


if __name__ == "__main__":
    run_evaluation()