import json
import random
from pathlib import Path
from collections import Counter


INPUT_PATH = Path("kgain_questions_with_news0_news2.json")
OUTPUT_PATH = Path("kgain_questions_with_news0_news2_shuffled_mc_only.json")
SEED = 20260501

TRUE_TEXT = "True"
FALSE_TEXT = "False"
DONT_KNOW_TEXT = "I do not know the answer."


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text):
    return " ".join(str(text).strip().split()).lower()


def is_dont_know(option):
    return normalize_text(option) == normalize_text(DONT_KNOW_TEXT)


def is_true_false_question(options):
    substantive_options = [opt for opt in options if not is_dont_know(opt)]
    normalized = {normalize_text(opt) for opt in substantive_options}

    return normalized == {
        normalize_text(TRUE_TEXT),
        normalize_text(FALSE_TEXT),
    }


def fix_true_false_question(question):
    correct_answer = question["correct_answer"]

    if normalize_text(correct_answer) == normalize_text(TRUE_TEXT):
        normalized_correct_answer = TRUE_TEXT
        new_correct_option = 1
    elif normalize_text(correct_answer) == normalize_text(FALSE_TEXT):
        normalized_correct_answer = FALSE_TEXT
        new_correct_option = 2
    else:
        raise ValueError(
            f"True/False question has non-True/False correct answer.\n"
            f"Question: {question.get('question-text')}\n"
            f"Correct answer: {correct_answer}"
        )

    question["question_type"] = "true_false"
    question["options"] = [
        TRUE_TEXT,
        FALSE_TEXT,
        DONT_KNOW_TEXT,
    ]
    question["correct_answer"] = normalized_correct_answer
    question["correct_option"] = new_correct_option

    return question


def shuffle_multiple_choice_question(question, rng):
    options = question["options"]
    correct_answer = question["correct_answer"]

    substantive_options = [opt for opt in options if not is_dont_know(opt)]
    dont_know_options = [opt for opt in options if is_dont_know(opt)]

    if len(dont_know_options) > 1:
        raise ValueError(f"Multiple 'I do not know' options found: {options}")

    if correct_answer not in substantive_options:
        raise ValueError(
            f"Correct answer not found among substantive options.\n"
            f"Question: {question.get('question-text')}\n"
            f"Correct answer: {correct_answer}\n"
            f"Options: {options}"
        )

    rng.shuffle(substantive_options)

    new_options = substantive_options + dont_know_options
    new_correct_option = new_options.index(correct_answer) + 1

    question["question_type"] = "multiple_choice"
    question["options"] = new_options
    question["correct_option"] = new_correct_option

    return question


def main():
    data = load_json(INPUT_PATH)

    overall_distribution = Counter()
    tf_distribution = Counter()
    mc_distribution = Counter()
    question_type_counts = Counter()

    for item in data:
        article_id = item["article_id"]

        for question in item.get("qa_annotations", []):
            options = question.get("options", [])

            if is_true_false_question(options):
                fix_true_false_question(question)
                tf_distribution[int(question["correct_option"])] += 1
                question_type_counts["true_false"] += 1

            else:
                q_seed = f"{SEED}_{article_id}_{question.get('question_in_set')}"
                q_rng = random.Random(q_seed)

                shuffle_multiple_choice_question(question, q_rng)
                mc_distribution[int(question["correct_option"])] += 1
                question_type_counts["multiple_choice"] += 1

            overall_distribution[int(question["correct_option"])] += 1

    save_json(data, OUTPUT_PATH)

    print(f"Saved shuffled dataset to {OUTPUT_PATH}")

    print("\nQuestion type counts:")
    for q_type, count in sorted(question_type_counts.items()):
        print(f"{q_type}: {count}")

    print("\nTrue/False correct option distribution:")
    tf_total = sum(tf_distribution.values())
    for option, count in sorted(tf_distribution.items()):
        pct = round((count / tf_total) * 100, 2) if tf_total else 0
        print(f"Option {option}: {count} questions ({pct}%)")

    print("\nMultiple-choice correct option distribution:")
    mc_total = sum(mc_distribution.values())
    for option, count in sorted(mc_distribution.items()):
        pct = round((count / mc_total) * 100, 2) if mc_total else 0
        print(f"Option {option}: {count} questions ({pct}%)")

    print("\nOverall correct option distribution:")
    total = sum(overall_distribution.values())
    for option, count in sorted(overall_distribution.items()):
        pct = round((count / total) * 100, 2) if total else 0
        print(f"Option {option}: {count} questions ({pct}%)")


if __name__ == "__main__":
    main()