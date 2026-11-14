def binary_metrics(truth, predicted):
    if len(truth) != len(predicted):
        raise ValueError("Truth and prediction lengths must match.")
    if any(value not in (0, 1) for value in [*truth, *predicted]):
        raise ValueError("Binary labels must contain only 0 or 1.")

    true_positive = sum(t == 1 and p == 1 for t, p in zip(truth, predicted))
    false_positive = sum(t == 0 and p == 1 for t, p in zip(truth, predicted))
    false_negative = sum(t == 1 and p == 0 for t, p in zip(truth, predicted))
    true_negative = sum(t == 0 and p == 0 for t, p in zip(truth, predicted))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else None
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive else None
    )
    accuracy = (
        (true_positive + true_negative) / len(truth)
        if truth else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    return {
        "support": len(truth),
        "positive_support": sum(truth),
        "negative_support": len(truth) - sum(truth),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
    }


def cohen_kappa(first_rater, second_rater):
    if len(first_rater) != len(second_rater):
        raise ValueError("Rater label lengths must match.")
    if not first_rater:
        raise ValueError("At least one paired rating is required.")
    if any(value not in (0, 1) for value in [*first_rater, *second_rater]):
        raise ValueError("Rater labels must contain only 0 or 1.")

    count = len(first_rater)
    observed_agreement = sum(
        first == second
        for first, second in zip(first_rater, second_rater)
    ) / count
    first_positive = sum(first_rater) / count
    second_positive = sum(second_rater) / count
    expected_agreement = (
        first_positive * second_positive
        + (1 - first_positive) * (1 - second_positive)
    )

    if expected_agreement == 1:
        return None
    return (
        observed_agreement - expected_agreement
    ) / (1 - expected_agreement)
