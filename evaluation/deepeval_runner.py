from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric
)
from evaluation.groq_judge import GroqJudge


def run_deepeval(test_cases):
    judge_model = GroqJudge(model="llama-3.3-70b-versatile")

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge_model),
        FaithfulnessMetric(threshold=0.7, model=judge_model),
        ToxicityMetric(threshold=0.0, model=judge_model)
    ]

    deepeval_results = evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

    return deepeval_results