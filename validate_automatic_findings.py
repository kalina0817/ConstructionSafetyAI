
import argparse
import json
import re
import sys
from pathlib import Path


FORBIDDEN_OBSERVATION_PATTERNS = [
    re.compile(r"\bhuman[\s-]+confirmed\b", re.IGNORECASE),
    re.compile(
        r"\bconfirmed\s+(?:missing|absence|lack of)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bconfirmed\s+PPE\s+(?:violation|finding)\b",
        re.IGNORECASE,
    ),
]


def validate_report(report_path):
    """Check automatic safety finding observations for misleading wording."""
    try:
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{report_path}: cannot read valid JSON: {exc}"]

    assessment = report.get("safety_assessment", {})
    findings = assessment.get("findings", [])

    if not isinstance(findings, list):
        return [f"{report_path}: safety_assessment.findings is not a list"]

    issues = []

    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            issues.append(
                f"{report_path}: finding index {index} is not an object"
            )
            continue

        observation = finding.get("observation", "")
        if not isinstance(observation, str):
            continue

        for pattern in FORBIDDEN_OBSERVATION_PATTERNS:
            match = pattern.search(observation)
            if match:
                finding_id = finding.get(
                    "finding_id", f"index {index}"
                )
                issues.append(
                    f"{report_path}: {finding_id}: "
                    f"misleading observation wording: {observation}"
                )
                break

    return issues


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate automatic safety findings for wording that "
            "incorrectly implies human confirmation."
        )
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="Directory containing generated detections.json reports.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    if not reports_dir.is_dir():
        print(f"ERROR: Directory not found: {reports_dir}")
        return 2

    report_files = sorted(reports_dir.rglob("detections.json"))

    if not report_files:
        print(f"ERROR: No detections.json files found in {reports_dir}")
        return 2

    all_issues = []

    for report_path in report_files:
        all_issues.extend(validate_report(report_path))

    print(f"Reports checked: {len(report_files)}")
    print(f"Wording issues found: {len(all_issues)}")

    if all_issues:
        print("\nVALIDATION FAILED:")
        for issue in all_issues:
            print(f"- {issue}")
        return 1

    print("\nVALIDATION PASSED.")
    print("No misleading human-confirmation wording was found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
