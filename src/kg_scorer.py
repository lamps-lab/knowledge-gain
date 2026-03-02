#!/usr/bin/env python3
"""
KG scorer using LLMSim distributions.

score(abstract, candidate_news) = mean_q [ p_post(correct) - p_pre(correct) ]
Averages over a persona mixture and over memory-trace uncertainty.

This is the expensive "ground-truth scorer" used to train a Reward Model (RM).
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from collections import defaultdict

from openai import OpenAI
import llmsim 

IDK_TEXT = llmsim.IDK_TEXT


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def format_options(options: List[str]) -> str:
    return "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])


def idk_index(options: List[str]) -> int:
    for i, opt in enumerate(options, start=1):
        if opt.strip() == IDK_TEXT:
            return i
    return len(options)


def dist_to_bucket_probs(
    candidates: List[int],
    probs: List[float],
    correct_opt: int,
    idk_opt: int,
    clamp: bool = True,
) -> Tuple[float, float, float]:
    cands = [int(x) for x in candidates]
    ps = [float(x) for x in probs]
    if clamp:
        ps = llmsim.clamp_probs(ps)  # reuse clamp+renorm

    pc = sum(p for c, p in zip(cands, ps) if c == int(correct_opt))
    pd = sum(p for c, p in zip(cands, ps) if c == int(idk_opt))
    pw = max(0.0, 1.0 - pc - pd)
    return pc, pw, pd


@dataclass
class Theta:
    temp: float = llmsim.TEMP
    trace_lo: float = 0.10
    trace_hi: float = 0.90
    trace_mode: str = "expectation"  # "expectation" or "sample"


def hard_correct_from_dist(
    cands: List[int],
    probs: List[float],
    correct_opt: int,
    mode: str,
    rng: random.Random,
) -> int:
    """
    Returns 1 if the chosen option equals correct_opt, else 0.
    mode:
      - "argmax": choose candidate with highest prob
      - "sample": sample according to probs
    """
    ps = llmsim.clamp_probs([float(p) for p in probs])
    if not cands:
        return 0

    if mode == "argmax":
        j = max(range(len(ps)), key=lambda i: ps[i])
        choice = int(cands[j])
        return 1 if choice == int(correct_opt) else 0

    # sample
    r = rng.random()
    cum = 0.0
    for c, p in zip(cands, ps):
        cum += p
        if r <= cum:
            return 1 if int(c) == int(correct_opt) else 0
    return 1 if int(cands[-1]) == int(correct_opt) else 0

class KGScorer:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        persona_path: str = "",
        persona_weights: Optional[Dict[str, float]] = None,
        seed: int = 0,
        theta_pre: Optional[Theta] = None,
        theta_post: Optional[Theta] = None,
        cache_dir: str = ".cache/kg_scorer",
    ):
        self.client = client or OpenAI()
        self.rng = random.Random(seed)

        self.theta_pre = theta_pre or Theta()
        self.theta_post = theta_post or Theta()

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cluster_prompts: Dict[str, str] = {}
        self.cluster_ids: List[str] = []
        if persona_path and os.path.exists(persona_path):
            cluster_prompts, _annot_map = llmsim.load_personas(persona_path)
            self.cluster_prompts = cluster_prompts
            self.cluster_ids = sorted(cluster_prompts.keys())
        else:
            self.cluster_prompts = {"C_default": "You are a careful student."}
            self.cluster_ids = ["C_default"]

        if persona_weights:
            w = {k: float(v) for k, v in persona_weights.items() if k in self.cluster_prompts}
            s = sum(w.values()) or 1.0
            self.persona_weights = {k: v / s for k, v in w.items()}
        else:
            self.persona_weights = {cid: 1.0 / len(self.cluster_ids) for cid in self.cluster_ids}

        self._persona_cdf: List[Tuple[str, float]] = []
        cum = 0.0
        for cid in self.cluster_ids:
            cum += self.persona_weights.get(cid, 0.0)
            self._persona_cdf.append((cid, cum))
        self._persona_cdf[-1] = (self._persona_cdf[-1][0], 1.0)

    def sample_persona_cluster(self) -> str:
        r = self.rng.random()
        for cid, c in self._persona_cdf:
            if r <= c:
                return cid
        return self.cluster_ids[-1]

    # caching
    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return None

    def _cache_put(self, key: str, obj: Dict[str, Any]) -> None:
        path = self._cache_path(key)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)

    # PRE
    def pre_stats(
    self,
    persona_sys: str,
    q_text: str,
    options: List[str],
    correct_option: int,
    hard_mode: str = "argmax",
    ) -> Tuple[float, int]:
        idk_opt = idk_index(options)
        options_str = format_options(options)

        gate_key = _sha1("pre_gate|" + persona_sys + "|" + q_text)
        gate_obj = self._cache_get(gate_key)
        if gate_obj is None:
            gate_obj = llmsim.responses_create_json(
                self.client,
                persona_sys + "\n\n" + llmsim.PRE_GATE_PROMPT,
                f"QUESTION:\n{q_text}\n\nReturn JSON only.",
                "pre_gate_v3",
                llmsim.PRE_GATE_SCHEMA,
                temperature=self.theta_pre.temp,
                max_output_tokens=80,
            )
            self._cache_put(gate_key, gate_obj)

        familiarity = gate_obj["familiarity"]
        gate_conf = float(gate_obj["confidence"])
        if familiarity == "technical_or_unknown":
            return 0.0, 0  # p_correct=0, hard_correct=0

        ans_key = _sha1("pre_ans|" + persona_sys + "|" + q_text + "|" + options_str + f"|{familiarity}|{gate_conf:.3f}")
        dist_obj = self._cache_get(ans_key)
        if dist_obj is None:
            dist_obj = llmsim.responses_create_json(
                self.client,
                persona_sys + "\n\n" + llmsim.PRE_ANSWER_PROMPT,
                (
                    f"FAMILIARITY: {familiarity}\nCONFIDENCE: {gate_conf:.2f}\n\n"
                    f"QUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only."
                ),
                "pre_answer_dist_v3",
                llmsim.ANSWER_DIST_SCHEMA,
                temperature=self.theta_pre.temp,
                max_output_tokens=120,
            )
            self._cache_put(ans_key, dist_obj)

        cands = [llmsim.safe_int(x) for x in dist_obj["candidates"]]
        probs = [float(x) for x in dist_obj["probs"]]

        pc, _, _ = dist_to_bucket_probs(cands, probs, correct_option, idk_opt, clamp=True)
        hard = hard_correct_from_dist(cands, probs, correct_option, hard_mode, self.rng)
        return pc, hard
    
    # POST(news)
    def news_memory_dual(self, persona_sys: str, news_text: str) -> Dict[str, Any]:
        mem_key = _sha1("news_mem|" + persona_sys + "|" + _sha1(news_text))
        mem_obj = self._cache_get(mem_key)
        if mem_obj is None:
            mem_obj = llmsim.responses_create_json(
                self.client,
                persona_sys + "\n\n" + llmsim.NEWS_MEMORY_DUAL_PROMPT,
                f"ARTICLE:\n{news_text}\n\nReturn JSON only.",
                "news_memory_dual_v3",
                llmsim.NEWS_MEMORY_DUAL_SCHEMA,
                temperature=self.theta_post.temp,
                max_output_tokens=220,
            )
            self._cache_put(mem_key, mem_obj)
        return mem_obj

    def news_answer_dist_from_memory(
        self,
        persona_sys: str,
        memory: str,
        mem_conf: float,
        q_text: str,
        options: List[str],
    ) -> Dict[str, Any]:
        options_str = format_options(options)
        detail_like = llmsim.is_detail_question(q_text)

        key = _sha1("news_ans|" + persona_sys + "|" + memory + f"|{mem_conf:.3f}|" + q_text + "|" + options_str)
        obj = self._cache_get(key)
        if obj is None:
            obj = llmsim.responses_create_json(
                self.client,
                persona_sys + "\n\n" + llmsim.NEWS_ANSWER_DIST_PROMPT,
                (
                    f"MEMORY TRACE:\n{memory}\n"
                    f"MEMORY_CONFIDENCE: {mem_conf:.2f}\n"
                    f"DETAIL_QUESTION: {detail_like}\n\n"
                    f"QUESTION:\n{q_text}\n\nOPTIONS:\n{options_str}\n\nReturn JSON only."
                ),
                "news_answer_dist_v3",
                llmsim.ANSWER_DIST_SCHEMA,
                temperature=self.theta_post.temp,
                max_output_tokens=160,
            )
            self._cache_put(key, obj)
        return obj

    def post_news_stats(
        self,
        persona_sys: str,
        news_text: str,
        q_text: str,
        options: List[str],
        correct_option: int,
        hard_mode: str = "argmax",
    ) -> Tuple[float, int]:
        idk_opt = idk_index(options)
        mem_obj = self.news_memory_dual(persona_sys, news_text)

        traces = mem_obj["traces"]
        tps = mem_obj["trace_probs"]
        tps = llmsim.clamp_probs([float(tps[0]), float(tps[1])], lo=self.theta_post.trace_lo, hi=self.theta_post.trace_hi)
        mem_conf = float(mem_obj["confidence"])

        # expected p_correct
        if self.theta_post.trace_mode == "sample":
            idx = 0 if self.rng.random() < tps[0] else 1
            dist = self.news_answer_dist_from_memory(persona_sys, traces[idx]["memory"], mem_conf, q_text, options)
            cands = [llmsim.safe_int(x) for x in dist["candidates"]]
            probs = [float(x) for x in dist["probs"]]
            pc, _, _ = dist_to_bucket_probs(cands, probs, correct_option, idk_opt, clamp=True)
        else:
            pc = 0.0
            for idx, w in enumerate(tps):
                dist = self.news_answer_dist_from_memory(persona_sys, traces[idx]["memory"], mem_conf, q_text, options)
                cands = [llmsim.safe_int(x) for x in dist["candidates"]]
                probs = [float(x) for x in dist["probs"]]
                pci, _, _ = dist_to_bucket_probs(cands, probs, correct_option, idk_opt, clamp=True)
                pc += w * pci
            pc = max(0.0, min(1.0, pc))

        # hard correctness (0/1): sample ONE trace, then choose ONE answer
        trace_draw = 0 if self.rng.random() < tps[0] else 1
        dist_h = self.news_answer_dist_from_memory(persona_sys, traces[trace_draw]["memory"], mem_conf, q_text, options)
        cands_h = [llmsim.safe_int(x) for x in dist_h["candidates"]]
        probs_h = [float(x) for x in dist_h["probs"]]
        hard = hard_correct_from_dist(cands_h, probs_h, correct_option, hard_mode, self.rng)

        return pc, hard

    # KG score

    def score(
        self,
        candidate_news: str,
        qa_annotations: List[Dict[str, Any]],
        persona_samples: int = 8,
        hard_mode: str = "argmax",
        return_breakdown: bool = False,
    ) -> Any:
        per_persona_rewards: List[float] = []
        breakdown: List[Dict[str, Any]] = []

        for _ in range(max(1, persona_samples)):
            cid = self.sample_persona_cluster()
            persona_sys = self.cluster_prompts[cid]
            deltas = []

            for qa in qa_annotations:
                q_text = qa.get("question-text") or qa.get("question_text") or ""
                options = qa.get("options") or []
                correct_opt = int(qa.get("correct_option", 0))
                if not q_text or not options or correct_opt <= 0:
                    continue

                pre_pc, pre_h = self.pre_stats(persona_sys, q_text, options, correct_opt, hard_mode=hard_mode)
                post_pc, post_h = self.post_news_stats(persona_sys, candidate_news, q_text, options, correct_opt, hard_mode=hard_mode)

                deltas.append(post_pc - pre_pc)

                if return_breakdown:
                    breakdown.append({
                        "cluster_id": cid,
                        "question_in_set": qa.get("question_in_set"),
                        "pre_p_correct": pre_pc,
                        "post_p_correct": post_pc,
                        "delta_p": post_pc - pre_pc,
                        "pre_hard_correct": pre_h,
                        "post_hard_correct": post_h,
                        "delta_h": post_h - pre_h,
                    })

            per_persona_rewards.append(float(sum(deltas) / max(1, len(deltas))))

        score = float(sum(per_persona_rewards) / max(1, len(per_persona_rewards)))
        if return_breakdown:
            return score, {"per_persona": per_persona_rewards, "per_question": breakdown}
        return score

def summarize_breakdown(per_question_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "pre_p": 0.0, "post_p": 0.0, "delta_p": 0.0,
                               "pre_a": 0.0, "post_a": 0.0, "delta_a": 0.0})
    for r in per_question_rows:
        q = r.get("question_in_set")
        if q is None:
            continue
        agg[q]["n"] += 1
        agg[q]["pre_p"] += float(r.get("pre_p_correct", 0.0))
        agg[q]["post_p"] += float(r.get("post_p_correct", 0.0))
        agg[q]["delta_p"] += float(r.get("delta_p", 0.0))
        agg[q]["pre_a"] += float(r.get("pre_hard_correct", 0.0))
        agg[q]["post_a"] += float(r.get("post_hard_correct", 0.0))
        agg[q]["delta_a"] += float(r.get("delta_h", 0.0))

    out = []
    for q in sorted(agg.keys()):
        n = agg[q]["n"]
        out.append({
            "question_in_set": q,
            "pre_p": agg[q]["pre_p"] / n,
            "post_p": agg[q]["post_p"] / n,
            "delta_p": agg[q]["delta_p"] / n,
            "pre_acc": agg[q]["pre_a"] / n,    # fraction correct across persona samples
            "post_acc": agg[q]["post_a"] / n,
            "delta_acc": agg[q]["delta_a"] / n,
        })
    return out


def print_question_table(qas: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> None:
    qtext_by_id = {qa.get("question_in_set"): (qa.get("question-text") or "") for qa in qas}
    print("\nPer-question breakdown (means over persona samples):")
    print("q  pre_p  post_p  dP     pre_acc post_acc dA     question")
    print("-- -----  ------  ------ ------- -------- ------ --------")
    for row in summary:
        q = row["question_in_set"]
        qt = qtext_by_id.get(q, "")
        qt = (qt[:90] + "…") if len(qt) > 91 else qt
        print(f"{q:>2} {row['pre_p']:>5.2f}  {row['post_p']:>6.2f}  {row['delta_p']:>6.3f} "
              f"{row['pre_acc']:>7.2f} {row['post_acc']:>8.2f} {row['delta_acc']:>6.3f}  {qt}")

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    txt = open(path, "r", encoding="utf-8").read().strip()
    if not txt:
        return []
    if "\n" in txt and txt.lstrip().startswith("{") and txt.rstrip().endswith("}"):
        try:
            obj = json.loads(txt)
            return [obj] if isinstance(obj, dict) else list(obj)
        except Exception:
            return [json.loads(line) for line in txt.splitlines() if line.strip()]
    obj = json.loads(txt)
    return [obj] if isinstance(obj, dict) else list(obj)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qgen", required=True, help="JSON/JSONL: record(s) containing qa_annotations and (optionally) candidates")
    ap.add_argument("--news_file", default="", help="Text file containing ONE candidate news article")
    ap.add_argument("--news", default="", help="ONE candidate news article passed inline")
    ap.add_argument("--persona_path", default=os.getenv("PERSONA_PATH", "./persona_cards.json"))
    ap.add_argument("--persona_samples", type=int, default=30)
    ap.add_argument("--trace_mode", choices=["expectation", "sample"], default="expectation")
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--record_idx", type=int, default=0, help="Which record in the JSON array to use")
    ap.add_argument("--top_k_generated", type=int, default=5, help="How many generated candidates (gen_*) to score")
    ap.add_argument("--table", action="store_true", help="Print per-question pre/post/delta table for the best candidate")
    ap.add_argument("--table_all", action="store_true", help="Print per-question tables for all scored candidates")

    args = ap.parse_args()

    recs = _read_json_or_jsonl(args.qgen)
    if not recs:
        raise SystemExit("No records found in --qgen file.")

    if args.record_idx < 0 or args.record_idx >= len(recs):
        raise SystemExit(f"--record_idx out of range (0..{len(recs)-1})")

    rec = recs[args.record_idx]
    qas = rec.get("qa_annotations") or []
    if not qas:
        raise SystemExit("Selected record has no qa_annotations.")

    scorer = KGScorer(
        client=OpenAI(),
        persona_path=args.persona_path,
        theta_pre=Theta(trace_mode="expectation"),
        theta_post=Theta(trace_mode=args.trace_mode),
    )

    # Mode 1: score ONE provided news text
    if args.news_file or args.news:
        if args.news_file:
            news_text = open(args.news_file, "r", encoding="utf-8").read()
        else:
            news_text = args.news

        if not news_text.strip():
            raise SystemExit("Empty --news/--news_file content.")

        if args.breakdown:
            s, extra = scorer.score(news_text, qas, persona_samples=args.persona_samples, return_breakdown=True)
            print(json.dumps({"score": s, **extra}, ensure_ascii=False, indent=2))
        else:
            s = scorer.score(news_text, qas, persona_samples=args.persona_samples, return_breakdown=False)
            print(json.dumps({"score": s}, ensure_ascii=False, indent=2))
        return

    # Mode 2: score ALL generated candidates in the record
    cands = rec.get("candidates") or []
    if not cands:
        raise SystemExit("No --news provided and record has no 'candidates' field to score.")

    # Only score generated ones: source starts with "gen_"
    gen_cands = [c for c in cands if str(c.get("source", "")).startswith("gen_")]
    gen_cands = gen_cands[: args.top_k_generated]

    if not gen_cands:
        raise SystemExit("No generated candidates found (expected sources like gen_0, gen_1, ...)")

    results = []
    detailed = []  # store breakdowns

    for c in gen_cands:
        text = (c.get("text") or "").strip()
        if not text:
            continue

        if args.table or args.table_all or args.breakdown:
            s, extra = scorer.score(text, qas, persona_samples=args.persona_samples, return_breakdown=True)
            summary = summarize_breakdown(extra["per_question"])
            detailed.append({"source": c.get("source", ""), "score": float(s), "summary": summary})
            results.append({"source": c.get("source", ""), "score": float(s)})
        else:
            s = scorer.score(text, qas, persona_samples=args.persona_samples, return_breakdown=False)
            results.append({"source": c.get("source", ""), "score": float(s)})

    results.sort(key=lambda x: x["score"], reverse=True)

    # JSON output (ranked)
    out_obj = {"record_idx": args.record_idx, "results": results}
    print(json.dumps(out_obj, ensure_ascii=False, indent=2))

    # Optional tables
    if args.table_all and detailed:
        detailed.sort(key=lambda x: x["score"], reverse=True)
        for d in detailed:
            print(f"\n=== {d['source']}  score={d['score']:.6f} ===")
            print_question_table(qas, d["summary"])
    elif args.table and detailed:
        detailed.sort(key=lambda x: x["score"], reverse=True)
        best = detailed[0]
        print(f"\n=== BEST: {best['source']}  score={best['score']:.6f} ===")
        print_question_table(qas, best["summary"])

    results.sort(key=lambda x: x["score"], reverse=True)
    print(json.dumps({"record_idx": args.record_idx, "results": results}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()