#!/usr/bin/env python3
import json
import os
import random
import hashlib
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from llmsim import (
    load_personas,
    responses_create_json,
    options_to_text,
    find_idk_index,
    safe_int,
    sample_from_candidates,
    clamp_probs,
    is_detail_question,
    PRE_GATE_PROMPT,
    PRE_ANSWER_PROMPT,
    NEWS_MEMORY_DUAL_PROMPT,
    NEWS_ANSWER_DIST_PROMPT,
    PRE_GATE_SCHEMA,
    ANSWER_DIST_SCHEMA,
    NEWS_MEMORY_DUAL_SCHEMA,
)

IDK_TEXT = "I do not know the answer."
PERSONA_PATH = os.getenv("PERSONA_PATH", "scripts/persona_cards.json")
TEMP = float(os.getenv("TEMP", "1.7"))


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class JsonlCache:
    def __init__(self, path: str):
        self.path = path
        self.data = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        self.data[rec["key"]] = rec["value"]
                    except Exception:
                        continue

        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: Any):
        if key in self.data:
            return
        self.data[key] = value
        self.f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
        self.f.flush()


class LLMSimKGRewardCached:
    def __init__(
        self,
        persona_path: str = PERSONA_PATH,
        n_simulated_readers: int = 2,
        seed: int = 0,
        pre_cache_path: str = "cache/pre_cache.jsonl",
    ):
        self.client = OpenAI()
        self.cluster_prompts, self.annotator_to_cluster = load_personas(persona_path)
        self.cluster_ids = list(self.cluster_prompts.keys())
        self.n_simulated_readers = n_simulated_readers
        self.rng = random.Random(seed)
        self.pre_cache = JsonlCache(pre_cache_path)

    def _sample_persona(self) -> Tuple[str, str]:
        cid = self.rng.choice(self.cluster_ids)
        return cid, self.cluster_prompts[cid]

    def _simulate_pre_answer_cached(
        self,
        cluster_id: str,
        persona_sys: str,
        question_text: str,
        options: List[str],
    ) -> int:
        idk = find_idk_index(options)

        cache_key = stable_hash({
            "type": "pre",
            "cluster_id": cluster_id,
            "question_text": question_text,
            "options": options,
        })

        cached = self.pre_cache.get(cache_key)
        if cached is not None:
            return int(cached["answer"])

        options_str = options_to_text(options)

        gate_obj = responses_create_json(
            self.client,
            persona_sys + "\n\n" + PRE_GATE_PROMPT,
            f"QUESTION:\n{question_text}\n\nReturn JSON only.",
            "pre_gate",
            PRE_GATE_SCHEMA,
            temperature=TEMP,
            max_output_tokens=80,
        )

        familiarity = gate_obj["familiarity"]
        gate_conf = float(gate_obj["confidence"])

        if familiarity == "technical_or_unknown":
            answer = idk
        else:
            pre_dist_obj = responses_create_json(
                self.client,
                persona_sys + "\n\n" + PRE_ANSWER_PROMPT,
                (
                    f"FAMILIARITY: {familiarity}\n"
                    f"CONFIDENCE: {gate_conf:.2f}\n\n"
                    f"QUESTION:\n{question_text}\n\n"
                    f"OPTIONS:\n{options_str}\n\n"
                    f"Return JSON only."
                ),
                "pre_answer_dist",
                ANSWER_DIST_SCHEMA,
                temperature=TEMP,
                max_output_tokens=120,
            )

            cands = [safe_int(x) for x in pre_dist_obj["candidates"]]
            probs = [float(x) for x in pre_dist_obj["probs"]]
            answer = sample_from_candidates(cands, probs)

            if answer < 1 or answer > len(options):
                answer = idk

        self.pre_cache.set(cache_key, {"answer": int(answer)})
        return int(answer)

    def _make_news_memory(self, persona_sys: str, article: str) -> Dict[str, Any]:
        mem_obj = responses_create_json(
            self.client,
            persona_sys + "\n\n" + NEWS_MEMORY_DUAL_PROMPT,
            f"ARTICLE:\n{article}\n\nReturn JSON only.",
            "news_memory_dual",
            NEWS_MEMORY_DUAL_SCHEMA,
            temperature=TEMP,
            max_output_tokens=200,
        )

        traces = mem_obj["traces"]
        probs = mem_obj["trace_probs"]
        probs = clamp_probs([float(probs[0]), float(probs[1])], lo=0.10, hi=0.90)

        chosen_idx = 0 if self.rng.random() < probs[0] else 1

        return {
            "memory": traces[chosen_idx]["memory"],
            "memory_confidence": float(mem_obj["confidence"]),
            "trace_type": traces[chosen_idx]["type"],
        }

    def _simulate_post_news_answer(
        self,
        persona_sys: str,
        memory: str,
        memory_confidence: float,
        question_text: str,
        options: List[str],
    ) -> int:
        idk = find_idk_index(options)
        options_str = options_to_text(options)

        ans_dist_obj = responses_create_json(
            self.client,
            persona_sys + "\n\n" + NEWS_ANSWER_DIST_PROMPT,
            (
                f"MEMORY TRACE:\n{memory}\n"
                f"MEMORY_CONFIDENCE: {memory_confidence:.2f}\n"
                f"DETAIL_QUESTION: {is_detail_question(question_text)}\n\n"
                f"QUESTION:\n{question_text}\n\n"
                f"OPTIONS:\n{options_str}\n\n"
                f"Return JSON only."
            ),
            "news_answer_dist",
            ANSWER_DIST_SCHEMA,
            temperature=TEMP,
            max_output_tokens=140,
        )

        cands = [safe_int(x) for x in ans_dist_obj["candidates"]]
        probs = [float(x) for x in ans_dist_obj["probs"]]
        answer = sample_from_candidates(cands, probs)

        if answer < 1 or answer > len(options):
            answer = idk

        return int(answer)

    def score_article(
        self,
        abstract: str,
        article: str,
        qas: List[Dict[str, Any]],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        total_pre_correct = 0
        total_post_correct = 0
        total = 0
        details = []

        for reader_idx in range(self.n_simulated_readers):
            cluster_id, persona_sys = self._sample_persona()

            # One memory trace per simulated reader per article.
            memory_obj = self._make_news_memory(persona_sys, article)
            memory = memory_obj["memory"]
            mem_conf = memory_obj["memory_confidence"]

            for qa in qas:
                q_text = qa.get("question_text") or qa.get("question-text")
                options = qa["options"]
                correct_raw = qa.get("correct_option")
                if isinstance(correct_raw, list):
                    if len(correct_raw) == 0:
                        continue
                    correct = int(correct_raw[0])
                else:
                    correct = int(correct_raw)

                pre_ans = self._simulate_pre_answer_cached(
                    cluster_id=cluster_id,
                    persona_sys=persona_sys,
                    question_text=q_text,
                    options=options,
                )

                post_ans = self._simulate_post_news_answer(
                    persona_sys=persona_sys,
                    memory=memory,
                    memory_confidence=mem_conf,
                    question_text=q_text,
                    options=options,
                )

                pre_correct = int(pre_ans == correct)
                post_correct = int(post_ans == correct)

                total_pre_correct += pre_correct
                total_post_correct += post_correct
                total += 1

                if return_details:
                    details.append({
                        "reader_idx": reader_idx,
                        "cluster_id": cluster_id,
                        "question": q_text,
                        "correct_option": correct,
                        "pre_answer": pre_ans,
                        "post_answer": post_ans,
                        "pre_correct": pre_correct,
                        "post_correct": post_correct,
                        "memory": memory,
                        "memory_confidence": mem_conf,
                    })

        pre_acc = total_pre_correct / max(1, total)
        post_acc = total_post_correct / max(1, total)
        kg = post_acc - pre_acc

        out = {
            "kg": kg,
            "reward": kg,
            "pre_acc": pre_acc,
            "post_acc": post_acc,
            "n_simulated_readers": self.n_simulated_readers,
            "n_questions": len(qas),
        }

        if return_details:
            out["details"] = details

        return out