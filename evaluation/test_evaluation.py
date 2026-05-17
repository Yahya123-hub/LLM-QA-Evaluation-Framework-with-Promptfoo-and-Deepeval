import os
import time
import yaml
from dotenv import load_dotenv
from rag.retriever import retrieve_context
from reports.generate_report import generate_report
from evaluation.groq_judge import GroqJudge
load_dotenv()
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric
)
from chatbot.app import get_response


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_test_cases(dataset):
    test_cases = []

    for item in dataset:
        input_text = item["vars"]["input"]    

        response = get_response(input_text)
        time.sleep(1)

        expected = "Relevant career guidance"

        if "assert" in item:
            expected = str(item["assert"])

        test_cases.append(
            LLMTestCase(
                input=input_text,
                actual_output=response,
                expected_output=expected,
                retrieval_context=retrieve_context(input_text)  

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

    all_test_cases = []

    for file in dataset_files:
        print(f"\nLoading dataset: {file}")
        dataset = load_dataset(file)

        test_cases = build_test_cases(dataset)
        all_test_cases.extend(test_cases)

    all_test_cases = all_test_cases[:5] #capped for testing purposes, can be removed for full eval
    print(f"\nRunning evaluation on {len(all_test_cases)} test cases...\n")

    judge_model = GroqJudge()

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