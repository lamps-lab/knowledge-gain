#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


N_STUDENTS = int(os.getenv("N_STUDENTS", "30"))
RESUME_RUN_DIR = os.getenv("RESUME_RUN_DIR", "runs/kgain_run_20260512_051123").strip()

# article = one memory trace per student/article/version
# question = one memory trace per student/article/version/question
MEMORY_SCOPE = os.getenv("MEMORY_SCOPE", "article").strip().lower()

# For dry runs. 0 means no limit.
LIMIT_STUDENTS = int(os.getenv("LIMIT_STUDENTS", "0"))
LIMIT_ITEMS = int(os.getenv("LIMIT_ITEMS", "0"))
LIMIT_QUESTIONS = int(os.getenv("LIMIT_QUESTIONS", "0"))


G3A_ROWS = [
    (1, 29733, "B", "Health", "news_2b"),
    (2, 29734, "B", "Space", "news_2b"),
    (3, 29739, "A", "Health", "news_0a"),
    (4, 29753, "B", "Health", "news_2b"),
    (5, 29775, "A", "Tech", "news_0a"),
    (6, 29802, "A", "Humans", "news_0a"),
    (7, 29846, "B", "Nature", "news_2b"),
    (8, 29866, "A", "Nature", "news_0a"),
    (9, 29873, "A", "Humans", "news_0a"),
    (10, 29902, "A", "Nature", "news_0a"),
    (11, 29929, "A", "Physics", "news_0a"),
    (12, 29943, "B", "Tech", "news_2b"),
    (13, 29958, "B", "Physics", "news_2b"),
    (14, 29988, "B", "Humans", "news_2b"),
    (15, 30052, "A", "Physics", "news_0a"),
    (16, 30061, "B", "Physics", "news_2b"),
    (17, 30086, "A", "Space", "news_0a"),
    (18, 30184, "A", "Tech", "news_0a"),
    (19, 30214, "B", "Space", "news_2b"),
    (20, 30708, "B", "Tech", "news_2b"),
]

G3B_ROWS = [
    (1, 29733, "B", "Health", "news_0b"),
    (2, 29734, "B", "Space", "news_0b"),
    (3, 29739, "A", "Health", "news_2a"),
    (4, 29753, "B", "Health", "news_0b"),
    (5, 29775, "A", "Tech", "news_2a"),
    (6, 29802, "A", "Humans", "news_2a"),
    (7, 29846, "B", "Nature", "news_0b"),
    (8, 29866, "A", "Nature", "news_2a"),
    (9, 29873, "A", "Humans", "news_2a"),
    (10, 29902, "A", "Nature", "news_2a"),
    (11, 29929, "A", "Physics", "news_2a"),
    (12, 29943, "B", "Tech", "news_0b"),
    (13, 29958, "B", "Physics", "news_0b"),
    (14, 29988, "B", "Humans", "news_0b"),
    (15, 30052, "A", "Physics", "news_2a"),
    (16, 30061, "B", "Physics", "news_0b"),
    (17, 30086, "A", "Space", "news_2a"),
    (18, 30184, "A", "Tech", "news_2a"),
    (19, 30214, "B", "Space", "news_0b"),
    (20, 30708, "B", "Tech", "news_0b"),
]


@dataclass(frozen=True)
class StudyItem:
    order: int
    article_id: int
    set_id: str
    category: str
    article_version_label: str
    article_key: str


@dataclass(frozen=True)
class Student:
    student_id: int
    group: str
    cluster_id: str
    source_annotator_id: Optional[int]


def article_key_from_version_label(label: str) -> str:
    if label.startswith("news_0"):
        return "news_0"
    if label.startswith("news_2"):
        return "news_2"
    raise ValueError(f"Unexpected article version label: {label}")


def build_items(rows: Iterable[Tuple[int, int, str, str, str]]) -> List[StudyItem]:
    items = []
    for order, article_id, set_id, category, version_label in rows:
        items.append(
            StudyItem(
                order=int(order),
                article_id=int(article_id),
                set_id=str(set_id),
                category=str(category),
                article_version_label=str(version_label),
                article_key=article_key_from_version_label(str(version_label)),
            )
        )
    return items


def load_llmsim(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find llmsim.py at: {path}")

    llmsim_dir = os.path.dirname(path)
    if llmsim_dir not in sys.path:
        sys.path.insert(0, llmsim_dir)

    spec = importlib.util.spec_from_file_location("base_llmsim", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import llmsim.py from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction_key(
    student_id: int,
    group: str,
    article_id: int,
    article_version_label: str,
    question_index: int,
) -> Tuple[int, str, int, str, int]:
    return (
        int(student_id),
        str(group),
        int(article_id),
        str(article_version_label),
        int(question_index),
    )


def memory_key(
    student_id: int,
    group: str,
    article_id: int,
    article_version_label: str,
    question_index: Optional[int],
) -> Tuple[int, str, int, str, int]:
    q = -1 if question_index is None else int(question_index)
    return (
        int(student_id),
        str(group),
        int(article_id),
        str(article_version_label),
        q,
    )


def load_existing_predictions(path: str) -> Dict[Tuple[int, str, int, str, int], Dict[str, Any]]:
    existing = {}

    if not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                key = prediction_key(
                    rec["student_id"],
                    rec["group"],
                    rec["article_id"],
                    rec["article_version_label"],
                    rec["question_index"],
                )
                existing[key] = rec
            except Exception:
                print(f"Warning: skipping malformed prediction line {line_no} in {path}")

    return existing


def load_existing_memory(path: str) -> Dict[Tuple[int, str, int, str, int], Dict[str, Any]]:
    existing = {}

    if not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
                key = memory_key(
                    rec["student_id"],
                    rec["group"],
                    rec["article_id"],
                    rec["article_version_label"],
                    rec.get("question_index"),
                )
                existing[key] = rec
            except Exception:
                print(f"Warning: skipping malformed memory line {line_no} in {path}")

    return existing


def build_synthetic_students(
    annotator_to_cluster: Dict[int, str],
    cluster_prompts: Dict[str, str],
    n_students: int,
) -> List[Student]:
    if n_students <= 0:
        raise ValueError("N_STUDENTS must be positive.")

    annotator_ids = sorted(int(k) for k in annotator_to_cluster.keys())

    if annotator_ids:
        base = [(aid, annotator_to_cluster[aid]) for aid in annotator_ids]
    else:
        base = [(None, cid) for cid in sorted(cluster_prompts.keys())]

    if not base:
        raise RuntimeError("No persona clusters found.")

    split = n_students // 2
    students = []

    for i in range(n_students):
        source_annotator_id, cluster_id = base[i % len(base)]

        if cluster_id not in cluster_prompts:
            cluster_id = sorted(cluster_prompts.keys())[i % len(cluster_prompts)]

        students.append(
            Student(
                student_id=i + 1,
                group="G3a" if i < split else "G3b",
                cluster_id=cluster_id,
                source_annotator_id=source_annotator_id,
            )
        )

    return students


def init_counts() -> Dict[str, Any]:
    return {
        "pre": [0, 0, 0],
        "post": [0, 0, 0],
        "post_by_article_key": {
            "news_0": [0, 0, 0],
            "news_2": [0, 0, 0],
        },
        "post_by_article_version_label": defaultdict(lambda: [0, 0, 0]),
        "post_by_group": {
            "G3a": [0, 0, 0],
            "G3b": [0, 0, 0],
        },
        "post_by_set": {
            "A": [0, 0, 0],
            "B": [0, 0, 0],
        },
        "post_by_category": defaultdict(lambda: [0, 0, 0]),
        "cluster_usage": Counter(),
    }


def label_to_bucket(label: str) -> int:
    return {
        "correct": 0,
        "incorrect": 1,
        "dk": 2,
    }[label]


def rebuild_counts(existing_predictions: Dict[Tuple[int, str, int, str, int], Dict[str, Any]]) -> Dict[str, Any]:
    counts = init_counts()

    for rec in existing_predictions.values():
        b_pre = label_to_bucket(rec["classification_pre"])
        b_post = label_to_bucket(rec["classification_post"])

        counts["pre"][b_pre] += 1
        counts["post"][b_post] += 1
        counts["post_by_article_key"][rec["article_key"]][b_post] += 1
        counts["post_by_article_version_label"][rec["article_version_label"]][b_post] += 1
        counts["post_by_group"][rec["group"]][b_post] += 1
        counts["post_by_set"][rec["set"]][b_post] += 1
        counts["post_by_category"][rec["category"]][b_post] += 1
        counts["cluster_usage"][rec.get("cluster_id", "UNKNOWN")] += 1

    return counts


def distribution(vec: List[int]) -> Dict[str, Any]:
    total = sum(vec)

    if total == 0:
        return {
            "n": 0,
            "correct": 0.0,
            "incorrect": 0.0,
            "dk": 0.0,
        }

    return {
        "n": total,
        "correct": vec[0] / total,
        "incorrect": vec[1] / total,
        "dk": vec[2] / total,
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def print_distribution(name: str, vec: List[int]) -> None:
    total = sum(vec)
    if total == 0:
        return

    d = distribution(vec)
    print(f"\nTASK: {name}")
    print(f"  N: {total}")
    print(f"  Correct:   {pct(d['correct'])}")
    print(f"  Incorrect: {pct(d['incorrect'])}")
    print(f"  IDK:       {pct(d['dk'])}")


def serialize_counts(counts: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pre": distribution(counts["pre"]),
        "post": distribution(counts["post"]),
        "post_by_article_key": {
            k: distribution(v)
            for k, v in counts["post_by_article_key"].items()
        },
        "post_by_article_version_label": {
            k: distribution(v)
            for k, v in dict(counts["post_by_article_version_label"]).items()
        },
        "post_by_group": {
            k: distribution(v)
            for k, v in counts["post_by_group"].items()
        },
        "post_by_set": {
            k: distribution(v)
            for k, v in counts["post_by_set"].items()
        },
        "post_by_category": {
            k: distribution(v)
            for k, v in dict(counts["post_by_category"]).items()
        },
        "cluster_usage": dict(counts["cluster_usage"]),
    }


def run_pre_answer(
    sim,
    client,
    persona_sys: str,
    q_text: str,
    options_str: str,
    n_options: int,
    idk: int,
) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]]]:
    gate_obj = sim.responses_create_json(
        client,
        persona_sys + "\n\n" + sim.PRE_GATE_PROMPT,
        f"QUESTION:\n{q_text}\n\nReturn JSON only.",
        "pre_gate",
        sim.PRE_GATE_SCHEMA,
        temperature=sim.TEMP,
        max_output_tokens=80,
    )

    familiarity = gate_obj["familiarity"]
    gate_conf = float(gate_obj["confidence"])

    pre_dist_obj = None

    if familiarity == "technical_or_unknown":
        a_pre = idk
    else:
        pre_dist_obj = sim.responses_create_json(
            client,
            persona_sys + "\n\n" + sim.PRE_ANSWER_PROMPT,
            (
                f"FAMILIARITY: {familiarity}\n"
                f"CONFIDENCE: {gate_conf:.2f}\n\n"
                f"QUESTION:\n{q_text}\n\n"
                f"OPTIONS:\n{options_str}\n\n"
                f"Return JSON only."
            ),
            "pre_answer_dist",
            sim.ANSWER_DIST_SCHEMA,
            temperature=sim.TEMP,
            max_output_tokens=120,
        )

        pre_cands = [sim.safe_int(x) for x in pre_dist_obj["candidates"]]
        pre_probs = [float(x) for x in pre_dist_obj["probs"]]
        a_pre = sim.sample_from_candidates(pre_cands, pre_probs)

    if a_pre < 1 or a_pre > n_options:
        a_pre = idk

    return int(a_pre), gate_obj, pre_dist_obj


def create_or_get_memory(
    sim,
    client,
    persona_sys: str,
    content: str,
    student: Student,
    item: StudyItem,
    question_index_for_memory: Optional[int],
    memory_cache: Dict[Tuple[int, str, int, str, int], Dict[str, Any]],
    memory_writer,
) -> Dict[str, Any]:
    key = memory_key(
        student.student_id,
        student.group,
        item.article_id,
        item.article_version_label,
        question_index_for_memory,
    )

    if key in memory_cache:
        return memory_cache[key]

    mem_obj = sim.responses_create_json(
        client,
        persona_sys + "\n\n" + sim.NEWS_MEMORY_DUAL_PROMPT,
        f"ARTICLE:\n{content}\n\nReturn JSON only.",
        "news_memory_dual",
        sim.NEWS_MEMORY_DUAL_SCHEMA,
        temperature=sim.TEMP,
        max_output_tokens=200,
    )

    traces = mem_obj["traces"]

    trace_probs = sim.clamp_probs(
        [
            float(mem_obj["trace_probs"][0]),
            float(mem_obj["trace_probs"][1]),
        ],
        lo=0.10,
        hi=0.90,
    )

    chosen_idx = 0 if sim.rng.random() < trace_probs[0] else 1
    chosen_trace = traces[chosen_idx]

    rec = {
        "student_id": student.student_id,
        "group": student.group,
        "cluster_id": student.cluster_id,
        "source_annotator_id": student.source_annotator_id,
        "article_id": item.article_id,
        "set": item.set_id,
        "category": item.category,
        "article_key": item.article_key,
        "article_version_label": item.article_version_label,
        "question_index": question_index_for_memory,
        "memory_scope": MEMORY_SCOPE,
        "topic": mem_obj.get("topic"),
        "traces": traces,
        "trace_probs": trace_probs,
        "chosen_trace_index": chosen_idx,
        "chosen_trace_type": chosen_trace.get("type"),
        "memory": chosen_trace.get("memory"),
        "likely_confusion": mem_obj.get("likely_confusion"),
        "memory_confidence": float(mem_obj.get("confidence", 0.0)),
    }

    memory_cache[key] = rec
    memory_writer.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return rec


def run_post_answer_from_memory(
    sim,
    client,
    persona_sys: str,
    memory_rec: Dict[str, Any],
    q_text: str,
    options_str: str,
    n_options: int,
    idk: int,
) -> Tuple[int, Dict[str, Any]]:
    try:
        ans_dist_obj = sim.responses_create_json(
            client,
            persona_sys + "\n\n" + sim.NEWS_ANSWER_DIST_PROMPT,
            (
                f"MEMORY TRACE:\n{memory_rec['memory']}\n"
                f"MEMORY_CONFIDENCE: {float(memory_rec['memory_confidence']):.2f}\n"
                f"DETAIL_QUESTION: {sim.is_detail_question(q_text)}\n\n"
                f"QUESTION:\n{q_text}\n\n"
                f"OPTIONS:\n{options_str}\n\n"
                f"Return JSON only."
            ),
            "news_answer_dist",
            sim.ANSWER_DIST_SCHEMA,
            temperature=sim.TEMP,
            max_output_tokens=140,
        )

        cands = [sim.safe_int(x) for x in ans_dist_obj["candidates"]]
        probs = [float(x) for x in ans_dist_obj["probs"]]
        a_post = sim.sample_from_candidates(cands, probs)

    except Exception as e:
        ans_dist_obj = {"error": str(e)}
        a_post = -1

    if a_post < 1 or a_post > n_options:
        a_post = idk

    return int(a_post), ans_dist_obj


def validate_inputs(
    questions_by_id: Dict[int, Dict[str, Any]],
    eval_by_id: Dict[int, Dict[str, Any]],
    design_by_group: Dict[str, List[StudyItem]],
) -> None:
    missing_questions = []
    missing_eval_rows = []
    missing_article_fields = []

    for group, items in design_by_group.items():
        for item in items:
            if item.article_id not in questions_by_id:
                missing_questions.append((group, item.article_id))

            if item.article_id not in eval_by_id:
                missing_eval_rows.append((group, item.article_id))
            else:
                row = eval_by_id[item.article_id]
                if item.article_key not in row or not row.get(item.article_key):
                    missing_article_fields.append((group, item.article_id, item.article_key))

    if missing_questions or missing_eval_rows or missing_article_fields:
        raise RuntimeError(
            "Input validation failed:\n"
            f"  Missing generated questions: {missing_questions}\n"
            f"  Missing eval rows: {missing_eval_rows}\n"
            f"  Missing article text fields: {missing_article_fields}\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--llmsim",
        default=os.getenv("LLMSIM_PATH", "./scripts/llmsim.py"),
        help="Path to the original llmsim.py.",
    )

    parser.add_argument(
        "--questions",
        default=os.getenv("QUESTIONS_PATH", "../evaluation/generated_questions.json"),
        help="Path to generated_questions.json.",
    )

    parser.add_argument(
        "--eval",
        default=os.getenv("EVAL_DATASET_PATH", "../evaluation/eval_dataset.json"),
        help="Path to eval_dataset.json.",
    )

    parser.add_argument(
        "--personas",
        default=os.getenv("PERSONA_PATH", "./scripts/persona_cards.json"),
        help="Path to persona_cards.json.",
    )

    args = parser.parse_args()

    if MEMORY_SCOPE not in {"article", "question"}:
        raise ValueError("MEMORY_SCOPE must be either 'article' or 'question'.")

    sim = load_llmsim(args.llmsim)

    client = sim.OpenAI()

    cluster_prompts, annotator_to_cluster = sim.load_personas(args.personas)

    students = build_synthetic_students(
        annotator_to_cluster=annotator_to_cluster,
        cluster_prompts=cluster_prompts,
        n_students=N_STUDENTS,
    )

    if LIMIT_STUDENTS > 0:
        students = students[:LIMIT_STUDENTS]

    questions_raw = load_json(args.questions)
    eval_raw = load_json(args.eval)

    questions_by_id = {
        int(row["article_id"]): row
        for row in questions_raw
    }

    eval_by_id = {
        int(row["id"]): row
        for row in eval_raw
    }

    design_by_group = {
        "G3a": build_items(G3A_ROWS),
        "G3b": build_items(G3B_ROWS),
    }

    validate_inputs(
        questions_by_id=questions_by_id,
        eval_by_id=eval_by_id,
        design_by_group=design_by_group,
    )

    if RESUME_RUN_DIR:
        out_dir = RESUME_RUN_DIR
        os.makedirs(out_dir, exist_ok=True)
        write_mode = "a"
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
        out_dir = f"runs/kgain_run_{run_id}"
        os.makedirs(out_dir, exist_ok=True)
        write_mode = "w"

    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    memory_path = os.path.join(out_dir, "memory_cache.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    config_path = os.path.join(out_dir, "config.json")

    existing_predictions = load_existing_predictions(predictions_path)
    processed_keys = set(existing_predictions.keys())

    memory_cache = load_existing_memory(memory_path)
    counts = rebuild_counts(existing_predictions)

    config = {
        "llmsim_path": os.path.abspath(args.llmsim),
        "questions_path": os.path.abspath(args.questions),
        "eval_dataset_path": os.path.abspath(args.eval),
        "persona_path": os.path.abspath(args.personas),
        "openai_model": getattr(sim, "MODEL", None),
        "temp": getattr(sim, "TEMP", None),
        "seed": getattr(sim, "SEED", None),
        "n_students_requested": N_STUDENTS,
        "n_students_running": len(students),
        "memory_scope": MEMORY_SCOPE,
        "limit_students": LIMIT_STUDENTS,
        "limit_items": LIMIT_ITEMS,
        "limit_questions": LIMIT_QUESTIONS,
        "resume_run_dir": RESUME_RUN_DIR or None,
        "output_dir": out_dir,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Output dir: {out_dir}")
    print(f"Using llmsim: {os.path.abspath(args.llmsim)}")
    print(f"Loaded {len(existing_predictions)} existing predictions.")
    print(f"Loaded {len(memory_cache)} existing memory records.")
    print(f"Running {len(students)} synthetic students.")
    print(f"MEMORY_SCOPE={MEMORY_SCOPE}")

    with open(predictions_path, write_mode, encoding="utf-8", buffering=1) as pred_w, open(
        memory_path, write_mode, encoding="utf-8", buffering=1
    ) as mem_w:

        for student in students:
            items = design_by_group[student.group]

            if LIMIT_ITEMS > 0:
                items = items[:LIMIT_ITEMS]

            persona_sys = cluster_prompts[student.cluster_id]

            print(
                f"\nStudent {student.student_id}/{len(students)} "
                f"| group={student.group} "
                f"| cluster={student.cluster_id} "
                f"| source_annotator={student.source_annotator_id}"
            )

            for item in items:
                eval_row = eval_by_id[item.article_id]
                questions_row = questions_by_id[item.article_id]

                content = eval_row[item.article_key]
                qa_list = questions_row["qa_annotations"]

                if LIMIT_QUESTIONS > 0:
                    qa_list = qa_list[:LIMIT_QUESTIONS]

                print(
                    f"  Order {item.order:02d} "
                    f"| article={item.article_id} "
                    f"| set={item.set_id} "
                    f"| category={item.category} "
                    f"| version={item.article_version_label} "
                    f"| field={item.article_key} "
                    f"| questions={len(qa_list)}"
                )

                article_memory_rec = None

                if MEMORY_SCOPE == "article":
                    article_memory_rec = create_or_get_memory(
                        sim=sim,
                        client=client,
                        persona_sys=persona_sys,
                        content=content,
                        student=student,
                        item=item,
                        question_index_for_memory=None,
                        memory_cache=memory_cache,
                        memory_writer=mem_w,
                    )

                for q_idx, q_ref in enumerate(qa_list):
                    key = prediction_key(
                        student_id=student.student_id,
                        group=student.group,
                        article_id=item.article_id,
                        article_version_label=item.article_version_label,
                        question_index=q_idx,
                    )

                    if key in processed_keys:
                        continue

                    q_text = q_ref["question-text"]
                    options_list = q_ref["options"]
                    options_str = sim.options_to_text(options_list)
                    n_options = len(options_list)

                    correct = q_ref.get("correct_option")
                    correct = int(correct) if correct is not None else None

                    idk = sim.find_idk_index(options_list)

                    a_pre, pre_gate_obj, pre_dist_obj = run_pre_answer(
                        sim=sim,
                        client=client,
                        persona_sys=persona_sys,
                        q_text=q_text,
                        options_str=options_str,
                        n_options=n_options,
                        idk=idk,
                    )

                    b_pre = sim.bucket(a_pre, correct, idk)
                    counts["pre"][b_pre] += 1

                    if MEMORY_SCOPE == "question":
                        memory_rec = create_or_get_memory(
                            sim=sim,
                            client=client,
                            persona_sys=persona_sys,
                            content=content,
                            student=student,
                            item=item,
                            question_index_for_memory=q_idx,
                            memory_cache=memory_cache,
                            memory_writer=mem_w,
                        )
                    else:
                        assert article_memory_rec is not None
                        memory_rec = article_memory_rec

                    a_post, post_dist_obj = run_post_answer_from_memory(
                        sim=sim,
                        client=client,
                        persona_sys=persona_sys,
                        memory_rec=memory_rec,
                        q_text=q_text,
                        options_str=options_str,
                        n_options=n_options,
                        idk=idk,
                    )

                    b_post = sim.bucket(a_post, correct, idk)

                    counts["post"][b_post] += 1
                    counts["post_by_article_key"][item.article_key][b_post] += 1
                    counts["post_by_article_version_label"][item.article_version_label][b_post] += 1
                    counts["post_by_group"][student.group][b_post] += 1
                    counts["post_by_set"][item.set_id][b_post] += 1
                    counts["post_by_category"][item.category][b_post] += 1
                    counts["cluster_usage"][student.cluster_id] += 1

                    rec = {
                        "student_id": student.student_id,
                        "group": student.group,
                        "cluster_id": student.cluster_id,
                        "source_annotator_id": student.source_annotator_id,

                        "order": item.order,
                        "article_id": item.article_id,
                        "set": item.set_id,
                        "category": item.category,
                        "article_key": item.article_key,
                        "article_version_label": item.article_version_label,

                        "question_index": q_idx,
                        "question_in_set": q_ref.get("question_in_set", q_idx + 1),
                        "question": q_text,
                        "options": options_list,
                        "correct_option": correct,
                        "correct_answer": q_ref.get("correct_answer"),
                        "idk_option": idk,

                        "model_pre_answer": int(a_pre),
                        "model_post_answer": int(a_post),
                        "classification_pre": sim.bucket_label(b_pre),
                        "classification_post": sim.bucket_label(b_post),

                        "memory_scope": MEMORY_SCOPE,
                        "memory_key_question_index": memory_rec.get("question_index"),
                        "memory_trace_type": memory_rec.get("chosen_trace_type"),
                        "memory_trace": memory_rec.get("memory"),
                        "memory_confidence": memory_rec.get("memory_confidence"),
                        "likely_confusion": memory_rec.get("likely_confusion"),

                        "pre_gate": pre_gate_obj,
                        "pre_answer_dist": pre_dist_obj,
                        "post_answer_dist": post_dist_obj,
                    }

                    pred_w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    processed_keys.add(key)

                    if len(processed_keys) % 50 == 0:
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(serialize_counts(counts), f, indent=2, ensure_ascii=False)

    print("\nCluster usage:")
    for cid, cnt in counts["cluster_usage"].most_common():
        print(f"  {cid}: {cnt}")

    print_distribution("PRE overall", counts["pre"])
    print_distribution("POST overall", counts["post"])

    print_distribution("POST news_0", counts["post_by_article_key"]["news_0"])
    print_distribution("POST news_2", counts["post_by_article_key"]["news_2"])

    print_distribution("POST G3a", counts["post_by_group"]["G3a"])
    print_distribution("POST G3b", counts["post_by_group"]["G3b"])

    print_distribution("POST Set A", counts["post_by_set"]["A"])
    print_distribution("POST Set B", counts["post_by_set"]["B"])

    for version_label, vec in sorted(dict(counts["post_by_article_version_label"]).items()):
        print_distribution(f"POST {version_label}", vec)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serialize_counts(counts), f, indent=2, ensure_ascii=False)

    print(f"\nSaved predictions to: {out_dir}/")
    print(f"  predictions: {predictions_path}")
    print(f"  memory:      {memory_path}")
    print(f"  summary:     {summary_path}")
    print(f"  config:      {config_path}")


if __name__ == "__main__":
    main()
