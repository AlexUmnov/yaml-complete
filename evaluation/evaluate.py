from pathlib import Path
import json
import time
from typing import Optional

import fire
import pandas
from tqdm.auto import tqdm

from predictors import predictor_registry


def longest_common_substring(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    max_len = 0
    end_index = 0

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i

    longest_substring = str1[end_index - max_len:end_index]
    return longest_substring

def completion_exact_match(expected_values, predictions):
    assert len(expected_values) == len(predictions)
    return sum(
        expected.strip() == pred.strip() for expected, pred in zip(expected_values, predictions)
    ) / len(expected_values)

def completion_iou(expected_values, predictions):
    assert len(expected_values) == len(predictions)
    results = []
    for expected, pred in zip(expected_values, predictions):
        intersection = len(longest_common_substring(expected, pred))
        union = len(expected) + len(pred) - intersection
        if union:
            results.append(intersection / union)
        else:
            if not expected and not pred:
                results.append(1)
            else:
                results.append(0)
    return sum(results) / len(results)

def evaluate_predictor(test_cases, predictor, predictor_name):
    start = time.time()
    predictions = [
        predictor.predict(text_before=case['code_before'], text_after=case['code_after'])
        for case in tqdm(test_cases, desc=predictor_name)
    ]
    elapsed = time.time() - start
    correct = [
        case['expected_code']  for case in test_cases
    ]
    return {
        "exact_match": completion_exact_match(correct, predictions),
        "iou": completion_iou(correct, predictions),
        "average_latency (s)": elapsed / len(test_cases),
        "average_cost ($)": predictor.total_cost / len(test_cases)
    }

def evaluate_all_predictors(test_file: Path, output_file: Path, limit_cases: Optional[int] = None):
    if isinstance(test_file, str):
        test_file = Path(test_file)
    if isinstance(output_file, str):
        output_file = Path(output_file)
    
    test_cases = json.load(open(test_file))
    if limit_cases:
        test_cases = test_cases[:limit_cases]
    print(f"Total test cases: {len(test_cases)}")

    results = []
    for predictor_key, predictor_object in tqdm(predictor_registry.items()):
        result = {
            "name": predictor_key
        }
        result.update(evaluate_predictor(test_cases, predictor_object, predictor_key))
        results.append(result)

    pandas.DataFrame(results).to_csv(output_file)
    
if __name__ == "__main__":
    fire.Fire(evaluate_all_predictors)