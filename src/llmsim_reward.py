#!/usr/bin/env python3
"""
LLMSim KnowledgeGain reward scorer.

This file assumes you keep the prompt/schema/helper functions from llmsim.py:
- responses_create_json
- load_personas
- options_to_text
- find_idk_index
- safe_int
- bucket
- sample_from_candidates
- clamp_probs
- is_detail_question
- PRE_GATE_PROMPT
- PRE_ANSWER_PROMPT
- NEWS_MEMORY_DUAL_PROMPT
- NEWS_ANSWER_DIST_PROMPT
- PRE_GATE_SCHEMA
- ANSWER_DIST_SCHEMA
- NEWS_MEMORY_DUAL_SCHEMA
"""

import os
import json
import random
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Import from your current llmsim.py
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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMP = float(os.getenv("TEMP", "1.7"))
PERSONA_PATH = os.getenv("PERSONA_PATH", "./persona_cards.json")


class LLMSimKGReward:
    def __init__(
        self,
        persona_path: str = PERSONA_PATH,
        n_simulated_readers: int = 10,
        seed: int = 0,
        temperature: float = TEMP,
    ):
        self.client = OpenAI()
        self.cluster_prompts, self.annotator_to_cluster = load_personas(persona_path)
        self.cluster_ids = list(self.cluster_prompts.keys())
        self.n_simulated_readers = n_simulated_readers
        self.temperature = temperature
        self.rng = random.Random(seed)

    def _sample_persona(self) -> str:
        """
        Sample a persona cluster.

        If you have empirical cluster weights from human annotators,
        replace this uniform sampling with those weights.
        """
        cid = self.rng.choice(self.cluster_ids)
        return cid, self.cluster_prompts[cid]

    def _simulate_pre_answer(
        self,
        persona_sys: str,
        question_text: str,
        options: List[str],
    ) -> int:
        """
        Simulate pre-reading answer using the calibrated PRE procedure.
        """
        idk = find_idk_index(options)
        options_str = options_to_text(options)

        gate_obj = responses_create_json(
            self.client,
            persona_sys + "\n\n" + PRE_GATE_PROMPT,
            f"QUESTION:\n{question_text}\n\nReturn JSON only.",
            "pre_gate",
            PRE_GATE_SCHEMA,
            temperature=self.temperature,
            max_output_tokens=80,
        )

        familiarity = gate_obj["familiarity"]
        gate_conf = float(gate_obj["confidence"])

        if familiarity == "technical_or_unknown":
            return idk

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
            temperature=self.temperature,
            max_output_tokens=120,
        )

        cands = [safe_int(x) for x in pre_dist_obj["candidates"]]
        probs = [float(x) for x in pre_dist_obj["probs"]]
        ans = sample_from_candidates(cands, probs)

        if ans < 1 or ans > len(options):
            ans = idk

        return ans

    def _make_news_memory(
        self,
        persona_sys: str,
        article: str,
    ) -> Dict[str, Any]:
        """
        Simulate one skimmed memory trace from a generated news article.
        """
        mem_obj = responses_create_json(
            self.client,
            persona_sys + "\n\n" + NEWS_MEMORY_DUAL_PROMPT,
            f"ARTICLE:\n{article}\n\nReturn JSON only.",
            "news_memory_dual",
            NEWS_MEMORY_DUAL_SCHEMA,
            temperature=self.temperature,
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
        """
        Simulate post-reading answer using only the memory trace.
        """
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
            temperature=self.temperature,
            max_output_tokens=140,
        )

        cands = [safe_int(x) for x in ans_dist_obj["candidates"]]
        probs = [float(x) for x in ans_dist_obj["probs"]]
        ans = sample_from_candidates(cands, probs)

        if ans < 1 or ans > len(options):
            ans = idk

        return ans

    def score_article(
        self,
        abstract: str,
        article: str,
        qas: List[Dict[str, Any]],
        return_details: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute LLMSim KnowledgeGain reward for one generated article.

        Expected QA schema:
        {
            "question_in_set": int,
            "question_text" or "question-text": str,
            "options": [str],
            "correct_option": int,  # 1-indexed
            "correct_answer": str
        }
        """

        total_pre_correct = 0
        total_post_correct = 0
        total_questions = 0

        details = []

        for s in range(self.n_simulated_readers):
            cluster_id, persona_sys = self._sample_persona()

            # Important: create one memory trace per simulated reader,
            # then use it for all questions, matching "read once then answer".
            memory_obj = self._make_news_memory(persona_sys, article)
            memory = memory_obj["memory"]
            mem_conf = memory_obj["memory_confidence"]

            for qa in qas:
                question_text = qa.get("question_text") or qa.get("question-text")
                options = qa["options"]
                correct = int(qa["correct_option"])

                pre_ans = self._simulate_pre_answer(
                    persona_sys=persona_sys,
                    question_text=question_text,
                    options=options,
                )

                post_ans = self._simulate_post_news_answer(
                    persona_sys=persona_sys,
                    memory=memory,
                    memory_confidence=mem_conf,
                    question_text=question_text,
                    options=options,
                )

                pre_correct = int(pre_ans == correct)
                post_correct = int(post_ans == correct)

                total_pre_correct += pre_correct
                total_post_correct += post_correct
                total_questions += 1

                if return_details:
                    details.append({
                        "reader": s,
                        "cluster_id": cluster_id,
                        "question": question_text,
                        "correct_option": correct,
                        "pre_answer": pre_ans,
                        "post_answer": post_ans,
                        "pre_correct": pre_correct,
                        "post_correct": post_correct,
                        "memory": memory,
                        "memory_confidence": mem_conf,
                    })

        pre_acc = total_pre_correct / max(1, total_questions)
        post_acc = total_post_correct / max(1, total_questions)
        kg = post_acc - pre_acc

        # Optional shaping: prevent negative rewards from dominating.
        # For RL, usually keep the raw value but you may normalize later.
        result = {
            "reward": kg,
            "kg": kg,
            "pre_acc": pre_acc,
            "post_acc": post_acc,
            "n_simulated_readers": self.n_simulated_readers,
            "n_questions": len(qas),
        }

        if return_details:
            result["details"] = details

        return result