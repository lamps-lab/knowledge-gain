#!/usr/bin/env python3
import json
import os
import math
import time
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DATASET_PATH = os.getenv("DATASET_PATH", "../data/kgain_annotated_dataset.json")
PERSONA_PATH = os.getenv("PERSONA_PATH", "./persona_cards.json")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TEMP = float(os.getenv("TEMP", "1.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "160"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
SEED = int(os.getenv("SEED", "0"))

# --- TWEET-ONLY thresholds (deterministic) ---
# Relaxed defaults to reduce over-IDK on tweets
TWEET_GATE_CONF_TH = float(os.getenv("TWEET_GATE_CONF_TH", "0.30"))
TWEET_MEM_CONF_TH = float(os.getenv("TWEET_MEM_CONF_TH", "0.20"))
TWEET_EXPL_DETAIL_MEMCONF_TH = float(os.getenv("TWEET_EXPL_DETAIL_MEMCONF_TH", "0.40"))

IDK_TEXT = "I do not know the answer."
EPS = 1e-9

rng = random.Random(SEED)

def pct(x: float) -> str:
    return f"{100*x:.1f}%"

def kl(p: List[float], q: List[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        pi = max(EPS, pi); qi = max(EPS, qi)
        s += pi * math.log(pi / qi)
    return s

def print_report(task_name: str, kl_val: float, pH: List[float], pM: List[float]) -> None:
    print(f"\nTASK: {task_name}")
    print(f"  KL Divergence: {kl_val:.4f}")
    print("  Distribution (Human vs LLM):")
    print(f"    Correct:   {pct(pH[0])} vs {pct(pM[0])}")
    print(f"    Incorrect: {pct(pH[1])} vs {pct(pM[1])}")
    print(f"    IDK:       {pct(pH[2])} vs {pct(pM[2])}")

def options_to_text(options_list: List[str]) -> str:
    return "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options_list)])

def find_idk_index(options_list: List[str]) -> int:
    try:
        return options_list.index(IDK_TEXT) + 1
    except ValueError:
        return len(options_list)

def bucket(ans: int, correct: Optional[int], idk: int) -> int:
    # 0=correct, 1=wrong, 2=idk
    if ans == idk:
        return 2
    if correct is not None and ans == correct:
        return 0
    return 1

def bucket_label(b: int) -> str:
    return {0: "correct", 1: "incorrect", 2: "dk"}[b]

def safe_int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except Exception:
        return default

def clamp_probs(ps: List[float], lo: float = 0.05, hi: float = 0.65) -> List[float]:
    # clamp then renormalize
    ps2 = [min(hi, max(lo, float(p))) for p in ps]
    s = sum(ps2)
    if s <= 0:
        return [1/len(ps2)] * len(ps2)
    return [p/s for p in ps2]

def sample_from_candidates(cands: List[int], probs: List[float]) -> int:
    probs = clamp_probs(probs)
    r = rng.random()
    acc = 0.0
    for c, p in zip(cands, probs):
        acc += p
        if r <= acc:
            return int(c)
    return int(cands[-1])

# NEW (tweet-only helper): sample conditional on NOT choosing IDK
def sample_non_idk(cands: List[int], probs: List[float], idk: int) -> int:
    keep_c = []
    keep_p = []
    for c, p in zip(cands, probs):
        ci = safe_int(c)
        if ci == int(idk):
            continue
        keep_c.append(ci)
        try:
            keep_p.append(float(p))
        except Exception:
            keep_p.append(0.0)
    if not keep_c:
        return int(idk)
    keep_p = clamp_probs(keep_p)
    r = rng.random()
    acc = 0.0
    for c, p in zip(keep_c, keep_p):
        acc += p
        if r <= acc:
            return int(c)
    return int(keep_c[-1])

def is_detail_question(q: str) -> bool:
    ql = q.lower()
    triggers = ["how many", "what year", "which year", "date", "percent", "%", "according to",
                "name of", "who was", "exact", "statistically", "p-value", "sample size", "randomized",
                "inclusion", "exclusion", "criteria", "confidence interval"]
    return any(t in ql for t in triggers)

def responses_create_json(
    client: OpenAI,
    system_text: str,
    user_text: str,
    schema_name: str,
    schema: Dict[str, Any],
    temperature: float = TEMP,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> Dict[str, Any]:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True
                    }
                },
            )
            out = (resp.output_text or "").strip()
            if not out:
                raise RuntimeError("Empty output_text.")
            return json.loads(out)
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")

# Schemas
PRE_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "familiarity": {"type": "string", "enum": ["obvious", "maybe_heard", "technical_or_unknown"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["familiarity", "confidence"],
    "additionalProperties": False
}

TWEET_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "explicitness": {"type": "string", "enum": ["explicit", "implied", "unclear"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["explicitness", "confidence"],
    "additionalProperties": False
}

NEWS_MEMORY_DUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "traces": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["gist", "schema"]},
                    "memory": {"type": "string"}
                },
                "required": ["type", "memory"],
                "additionalProperties": False
            }
        },
        "trace_probs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "likely_confusion": {"type": "string", "enum": ["direction", "actor", "causality", "scope", "none"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["topic", "traces", "trace_probs", "likely_confusion", "confidence"],
    "additionalProperties": False
}

ABSTRACT_MEMORY_DUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "traces": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["gist", "schema"]},
                    "memory": {"type": "string"}
                },
                "required": ["type", "memory"],
                "additionalProperties": False
            }
        },
        "trace_probs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "likely_confusion": {"type": "string", "enum": ["direction", "group", "causality", "endpoint", "none"]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["topic", "traces", "trace_probs", "likely_confusion", "confidence"],
    "additionalProperties": False
}

TWEET_MEMORY_DUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "traces": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["literal", "vibe"]},
                    "memory": {"type": "string"}
                },
                "required": ["type", "memory"],
                "additionalProperties": False
            }
        },
        "trace_probs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["topic", "traces", "trace_probs", "confidence"],
    "additionalProperties": False
}

ANSWER_DIST_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3
        },
        "probs": {
            "type": "array",
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "minItems": 3,
            "maxItems": 3
        }
    },
    "required": ["candidates", "probs"],
    "additionalProperties": False
}

def load_personas(path: str) -> Tuple[Dict[str, str], Dict[int, str]]:
    obj = json.load(open(path, "r", encoding="utf-8"))
    cluster_prompts = {c["cluster_id"]: c["system_prompt"] for c in obj["clusters"]}
    mapping = {int(k): v for k, v in obj["annotator_to_cluster"].items()}
    return cluster_prompts, mapping

# Prompts

PRE_GATE_PROMPT = """TASK: PRE GATE (before seeing options)

You are answering BEFORE any supporting text.

Decide if you could answer from immediate memory in under 1 second, with zero reasoning.
If you would need to think, infer, eliminate options, or “work it out”, then you do NOT know it.

Use:
- obvious: only if you are sure you could say the answer before seeing options.
- maybe_heard: feels familiar but not instantly recallable.
- technical_or_unknown: anything else.

Be conservative. Most questions are NOT obvious.

Return JSON: {"familiarity":"obvious|maybe_heard|technical_or_unknown","confidence":0..1}
"""

PRE_ANSWER_PROMPT = """TASK: PRE ANSWER (options now shown)

Rules:
- You are answering BEFORE reading any supporting text.
- Do NOT do elimination or multi-step reasoning.
- If you do not truly know the answer, IDK should be a very plausible choice.

Verbalized Sampling:
- Provide exactly 3 candidate option numbers and probabilities (not all mass on one).
- Include the IDK option as a candidate unless the answer is truly obvious.

Return JSON: {"candidates":[n1,n2,n3],"probs":[p1,p2,p3]}
"""

NEWS_MEMORY_DUAL_PROMPT = """TASK: NEWS MEMORY TRACE (dual-trace)

You skimmed quickly and cannot re-check the article.

Produce TWO recollections:
1) gist: short, vague, main takeaway (no numbers/dates/proper names).
2) schema: a typical/assumed version people might remember for this topic (can swap one detail: direction/actor/causality/scope).

Constraints:
- Each memory: 8–14 words.
- No numbers, no dates, no proper names.
- Provide probabilities for recalling each trace (sum ~ 1, not extreme).
- Provide likely_confusion and confidence (0..1).

Return JSON with:
{"topic":"...","traces":[{"type":"gist","memory":"..."},{"type":"schema","memory":"..."}],
 "trace_probs":[p_gist,p_schema],"likely_confusion":"direction|actor|causality|scope|none","confidence":0..1}
"""

NEWS_ANSWER_DIST_PROMPT = """TASK: NEWS ANSWER (from one recalled trace ONLY)

You must answer using ONLY the provided memory trace. You do NOT have the article.
Do NOT verify options against the original text.

Rules:
- If the question seems to require exact numbers/dates/names, IDK becomes plausible.
- Otherwise answer from gist; slips can happen (negation, direction, actor/target, causality).

Verbalized Sampling:
- Give 3 candidates + probabilities (sum ~ 1, not extreme; max prob <= 0.65).
- Include IDK as a candidate when memory confidence is low OR the question is detail-like.

Return JSON: {"candidates":[n1,n2,n3],"probs":[p1,p2,p3]}
"""

ABSTRACT_MEMORY_DUAL_PROMPT = """TASK: ABSTRACT MEMORY TRACE (dual-trace, non-expert)

You read the abstract once and cannot re-check it.

Produce TWO recollections:
1) gist: the simplest story takeaway (ignore hedging).
2) schema: a typical overgeneralized recollection people might form (can swap one detail: direction/group/causality/endpoint).

Constraints:
- Each memory: 8–14 words.
- No statistics, no exact measures, no sample sizes, no p-values.
- Provide probabilities for recalling each trace (sum ~ 1, not extreme).
- Provide likely_confusion and confidence (0..1).

Return JSON with:
{"topic":"...","traces":[{"type":"gist","memory":"..."},{"type":"schema","memory":"..."}],
 "trace_probs":[p_gist,p_schema],"likely_confusion":"direction|group|causality|endpoint|none","confidence":0..1}
"""

ABSTRACT_ANSWER_DIST_PROMPT = """TASK: ABSTRACT ANSWER (from one recalled trace ONLY)

You must answer using ONLY the provided memory trace. You do NOT have the abstract.

Rules:
- If the question is about methods/statistics/inclusion criteria/confounds/exact measures -> IDK is plausible.
- Otherwise answer from a simple story; overgeneralization and causality confusion are common.

Verbalized Sampling:
- Give 3 candidates + probabilities (sum ~ 1, not extreme; max prob <= 0.65).
- Include IDK as a candidate when memory confidence is low OR the question is methods/detail-like.

Return JSON: {"candidates":[n1,n2,n3],"probs":[p1,p2,p3]}
"""

# ---------------- TWEET ----------------

TWEET_GATE_PROMPT = """TASK: TWEET EVIDENCE GATE

Given the tweet and the question, decide if the tweet supports answering.

- explicit: tweet directly states the needed claim.
- implied: tweet is on-topic and suggests the claim but does not clearly state it.
- unclear: tweet is off-topic OR provides essentially no usable signal for the question.

Be realistic: many tweets are "implied", but some are truly "unclear".

Return JSON: {"explicitness":"explicit|implied|unclear","confidence":0..1}
"""

TWEET_MEMORY_DUAL_PROMPT = """TASK: TWEET INTERPRETATION (dual-trace)

You saw a short tweet once.

Produce TWO interpretations:
1) literal: what the tweet literally says (short, incomplete).
2) vibe: what it seems to imply from tone/stance (can be wrong).

Constraints:
- Each: 6–12 words.
- Provide probabilities for recalling each (sum ~ 1, not extreme).
- Provide confidence (0..1).

Return JSON:
{"topic":"...","traces":[{"type":"literal","memory":"..."},{"type":"vibe","memory":"..."}],
 "trace_probs":[p_lit,p_vibe],"confidence":0..1}
"""

# Slightly more “guess-friendly” for implied to reduce over-IDK.
TWEET_ANSWER_DIST_PROMPT = """TASK: TWEET ANSWER (from one interpretation ONLY)

You must answer using ONLY the provided tweet interpretation. You do NOT have the tweet.
No outside context. No careful inference.

Rules:
- explicit: answer if clearly supported; otherwise IDK.
- implied: people often guess from keywords/vibe. IDK happens, but should NOT dominate.
- unclear: usually IDK.

Verbalized Sampling:
- Give 3 candidates + probabilities (not extreme; max prob <= 0.65).
- If explicitness is implied, include IDK only when the interpretation provides little/no usable signal.
- If explicitness is unclear, include IDK.

Return JSON: {"candidates":[n1,n2,n3],"probs":[p1,p2,p3]}
"""

# ---------------- Main ----------------
if __name__ == "__main__":
    client = OpenAI()
    cluster_prompts, annotator_to_cluster = load_personas(PERSONA_PATH)

    raw = json.load(open(DATASET_PATH, "r", encoding="utf-8"))
    num_docs = len(raw)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = f"runs/run_{run_id}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")

    human_pre = [0, 0, 0]
    model_pre = [0, 0, 0]
    human_post = {"news": [0, 0, 0], "abstract": [0, 0, 0], "tweet": [0, 0, 0]}
    model_post = {"news": [0, 0, 0], "abstract": [0, 0, 0], "tweet": [0, 0, 0]}

    cluster_usage = Counter()

    with open(out_path, "w", encoding="utf-8") as w:
        for doc_id, doc in enumerate(raw):
            if (doc_id % 10 == 0) or (doc_id == num_docs - 1):
                print(f"Processed {doc_id}/{num_docs} documents...")

            media = doc["content-type"]
            content = doc.get("content", "")
            anns = doc.get("human_annotations") or []
            if not anns:
                continue

            qs = anns[0]["qa_annotations"]
            for q_idx, q_ref in enumerate(qs):
                q_text = q_ref["question-text"]
                options_list = q_ref["options"]
                options_str = options_to_text(options_list)
                n = len(options_list)

                correct = q_ref.get("correct_option")
                correct = int(correct) if correct is not None else None
                idk = find_idk_index(options_list)

                for ann in anns:
                    aid = ann["annotator_id"]
                    qa = ann["qa_annotations"][q_idx]

                    ha_pre = safe_int(qa["human-answer-pre"])
                    human_pre[bucket(ha_pre, correct, idk)] += 1

                    ha_post = qa.get("human-answer-post")
                    if ha_post is not None and media in human_post:
                        human_post[media][bucket(safe_int(ha_post), correct, idk)] += 1

                    cluster_id = annotator_to_cluster.get(aid, "C_average_gist")
                    cluster_usage[cluster_id] += 1
                    persona_sys = cluster_prompts[cluster_id]

                    # ---------------- PRE ----------------
                    gate_obj = responses_create_json(
                        client,
                        persona_sys + "\n\n" + PRE_GATE_PROMPT,
                        f"QUESTION:\n{q_text}\n\nReturn JSON only.",
                        "pre_gate",
                        PRE_GATE_SCHEMA,
                        temperature=TEMP,
                        max_output_tokens=80
                    )
                    familiarity = gate_obj["familiarity"]
                    gate_conf = float(gate_obj["confidence"])

                    if familiarity == "technical_or_unknown":
                        a_pre = idk
                    else:
                        pre_dist_obj = responses_create_json(
                            client,
                            persona_sys + "\n\n" + PRE_ANSWER_PROMPT,
                            f"FAMILIARITY: {familiarity}\nCONFIDENCE: {gate_conf:.2f}\n\nQUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only.",
                            "pre_answer_dist",
                            ANSWER_DIST_SCHEMA,
                            temperature=TEMP,
                            max_output_tokens=120
                        )
                        pre_cands = [safe_int(x) for x in pre_dist_obj["candidates"]]
                        pre_probs = [float(x) for x in pre_dist_obj["probs"]]
                        a_pre = sample_from_candidates(pre_cands, pre_probs)

                    if a_pre < 1 or a_pre > n:
                        a_pre = idk
                    b_pre = bucket(a_pre, correct, idk)
                    model_pre[b_pre] += 1

                    # ---------------- POST ----------------
                    a_post = idk
                    b_post = 2

                    if ha_post is not None and media in ("news", "abstract", "tweet"):
                        try:
                            if media == "news":
                                mem_obj = responses_create_json(
                                    client,
                                    persona_sys + "\n\n" + NEWS_MEMORY_DUAL_PROMPT,
                                    f"ARTICLE:\n{content}\n\nReturn JSON only.",
                                    "news_memory_dual",
                                    NEWS_MEMORY_DUAL_SCHEMA,
                                    temperature=TEMP,
                                    max_output_tokens=200
                                )
                                traces = mem_obj["traces"]
                                probs = mem_obj["trace_probs"]
                                probs = clamp_probs([float(probs[0]), float(probs[1])], lo=0.10, hi=0.90)
                                chosen_idx = 0 if rng.random() < probs[0] else 1
                                memory = traces[chosen_idx]["memory"]
                                mem_conf = float(mem_obj["confidence"])

                                ans_dist_obj = responses_create_json(
                                    client,
                                    persona_sys + "\n\n" + NEWS_ANSWER_DIST_PROMPT,
                                    f"MEMORY TRACE:\n{memory}\nMEMORY_CONFIDENCE: {mem_conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q_text)}\n\nQUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only.",
                                    "news_answer_dist",
                                    ANSWER_DIST_SCHEMA,
                                    temperature=TEMP,
                                    max_output_tokens=140
                                )
                                cands = [safe_int(x) for x in ans_dist_obj["candidates"]]
                                probs2 = [float(x) for x in ans_dist_obj["probs"]]
                                a_post = sample_from_candidates(cands, probs2)

                            elif media == "abstract":
                                mem_obj = responses_create_json(
                                    client,
                                    persona_sys + "\n\n" + ABSTRACT_MEMORY_DUAL_PROMPT,
                                    f"ABSTRACT:\n{content}\n\nReturn JSON only.",
                                    "abstract_memory_dual",
                                    ABSTRACT_MEMORY_DUAL_SCHEMA,
                                    temperature=TEMP,
                                    max_output_tokens=200
                                )
                                traces = mem_obj["traces"]
                                probs = mem_obj["trace_probs"]
                                probs = clamp_probs([float(probs[0]), float(probs[1])], lo=0.10, hi=0.90)
                                chosen_idx = 0 if rng.random() < probs[0] else 1
                                memory = traces[chosen_idx]["memory"]
                                mem_conf = float(mem_obj["confidence"])

                                ans_dist_obj = responses_create_json(
                                    client,
                                    persona_sys + "\n\n" + ABSTRACT_ANSWER_DIST_PROMPT,
                                    f"MEMORY TRACE:\n{memory}\nMEMORY_CONFIDENCE: {mem_conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q_text)}\n\nQUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only.",
                                    "abstract_answer_dist",
                                    ANSWER_DIST_SCHEMA,
                                    temperature=TEMP,
                                    max_output_tokens=140
                                )
                                cands = [safe_int(x) for x in ans_dist_obj["candidates"]]
                                probs2 = [float(x) for x in ans_dist_obj["probs"]]
                                a_post = sample_from_candidates(cands, probs2)

                            else:  # ---------------- TWEET ----------------
                                tg = responses_create_json(
                                    client,
                                    persona_sys + "\n\n" + TWEET_GATE_PROMPT,
                                    f"TWEET:\n{content}\n\nQUESTION:\n{q_text}\n\nReturn JSON only.",
                                    "tweet_gate",
                                    TWEET_GATE_SCHEMA,
                                    temperature=TEMP,
                                    max_output_tokens=80
                                )
                                explicitness = tg["explicitness"]
                                tgate_conf = float(tg["confidence"])

                                # Keep: unclear => IDK
                                if explicitness == "unclear":
                                    a_post = idk
                                else:
                                    mem_obj = responses_create_json(
                                        client,
                                        persona_sys + "\n\n" + TWEET_MEMORY_DUAL_PROMPT,
                                        f"EXPLICITNESS: {explicitness}\nQUESTION:\n{q_text}\n\nTWEET:\n{content}\n\nReturn JSON only.",
                                        "tweet_memory_dual",
                                        TWEET_MEMORY_DUAL_SCHEMA,
                                        temperature=TEMP,
                                        max_output_tokens=160
                                    )
                                    traces = mem_obj["traces"]
                                    tprobs = mem_obj["trace_probs"]
                                    tprobs = clamp_probs([float(tprobs[0]), float(tprobs[1])], lo=0.10, hi=0.90)
                                    mem_conf = float(mem_obj["confidence"])

                                    # Implied tweets: favor "vibe" when confidence is not strong / detail-like.
                                    if explicitness == "implied" and ((tgate_conf < 0.60) or (mem_conf < 0.55) or is_detail_question(q_text)):
                                        chosen_idx = 1
                                    else:
                                        chosen_idx = 0 if rng.random() < tprobs[0] else 1

                                    memory = traces[chosen_idx]["memory"]

                                    # Implied: abstain only if BOTH signals are weak (AND).
                                    if explicitness == "implied" and (tgate_conf < TWEET_GATE_CONF_TH) and (mem_conf < TWEET_MEM_CONF_TH):
                                        a_post = idk
                                    else:
                                        # Explicit: still allow IDK on detail questions when memory is weak.
                                        if explicitness == "explicit" and is_detail_question(q_text) and (mem_conf < TWEET_EXPL_DETAIL_MEMCONF_TH):
                                            a_post = idk
                                        else:
                                            ans_dist_obj = responses_create_json(
                                                client,
                                                persona_sys + "\n\n" + TWEET_ANSWER_DIST_PROMPT,
                                                f"EXPLICITNESS: {explicitness}\nGATE_CONFIDENCE: {tgate_conf:.2f}\nINTERPRETATION:\n{memory}\nCONFIDENCE: {mem_conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q_text)}\n\nQUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only.",
                                                "tweet_answer_dist",
                                                ANSWER_DIST_SCHEMA,
                                                temperature=TEMP,
                                                max_output_tokens=120
                                            )
                                            cands = [safe_int(x) for x in ans_dist_obj["candidates"]]
                                            probs2 = [float(x) for x in ans_dist_obj["probs"]]
                                            a_post = sample_from_candidates(cands, probs2)

                                            # If we sampled IDK on an IMPLIED tweet but the signals aren't both weak,
                                            # resample conditional on non-IDK to mimic “guessing from vibes”.
                                            if explicitness == "implied" and a_post == idk:
                                                if not ((tgate_conf < TWEET_GATE_CONF_TH) and (mem_conf < TWEET_MEM_CONF_TH)):
                                                    a_post = sample_non_idk(cands, probs2, idk)

                        except Exception:
                            a_post = -1

                        if a_post < 1 or a_post > n:
                            a_post = idk
                        b_post = bucket(a_post, correct, idk)
                        model_post[media][b_post] += 1

                    record = {
                        "doc_id": doc_id,
                        "question_index": q_idx,
                        "media": media,
                        "annotator_id": aid,
                        "cluster_id": cluster_id,
                        "question": q_text,
                        "correct_option": correct,
                        "model_pre_answer": int(a_pre),
                        "model_post_answer": int(a_post),
                        "classification_pre": bucket_label(b_pre),
                        "classification_post": bucket_label(b_post),
                    }
                    w.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\nCluster usage (LLM calls), derived from human annotator appearances:")
    for cid, cnt in cluster_usage.most_common():
        print(f"  {cid}: {cnt}")

    Hp = sum(human_pre); Mp = sum(model_pre)
    pH_pre = [x / Hp for x in human_pre]
    pM_pre = [x / Mp for x in model_pre]
    print_report("PRE", kl(pH_pre, pM_pre), pH_pre, pM_pre)

    for m in ("news", "abstract", "tweet"):
        Hm = sum(human_post[m]); Mm = sum(model_post[m])
        if Hm == 0 or Mm == 0:
            continue
        pH = [x / Hm for x in human_post[m]]
        pM = [x / Mm for x in model_post[m]]
        print_report(m.upper(), kl(pH, pM), pH, pM)

    print(f"\nSaved predictions to: {out_dir}/")