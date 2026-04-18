#!/usr/bin/env python3
"""
Generate ALL LLMSim ablation predictions, then evaluate them into ACL-ready
CSV/LaTeX tables.

Assumptions:
  - llmsim.py is importable, e.g. this script is run from repo root with
    --scripts_dir scripts
  - persona_cards.json exists
  - OPENAI_API_KEY is set

Example:
  python scripts/run_llmsim_ablation.py \
    --dataset data/kgain_annotated_dataset.json \
    --persona_path scripts/persona_cards.json \
    --scripts_dir scripts \
    --out_root runs/llmsim_ablation \
    --model gpt-4o-mini

Debug:
  python scripts/run_llmsim_ablation.py ... --max_docs 3 \
    --variants "Direct answer,Full LLMSim"
"""
from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, re, sys, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

IDK_TEXT = "I do not know the answer."
EPS = 1e-9

SINGLE_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}

BASE_SYSTEM = """You are a simulated survey participant. Answer as an ordinary reader, not as an expert annotator. Follow the task instructions exactly."""

DIRECT_PRE_PROMPT = """TASK: DIRECT PRE ANSWER
You are answering BEFORE seeing any supporting text. Choose exactly one option number.
{IDK_POLICY}
Return JSON only: {{"answer": <option_number>}}
"""

DIRECT_POST_PROMPT = """TASK: DIRECT POST ANSWER
You are answering AFTER reading the supporting text. Choose exactly one option number.
{IDK_POLICY}
Return JSON only: {{"answer": <option_number>}}
"""

TRACE_SINGLE_PROMPT = """TASK: MEMORY-BOTTLENECK ANSWER
Answer using ONLY the provided memory trace/interpretation. You do NOT have the original text.
Choose exactly one option number.
{IDK_POLICY}
Return JSON only: {{"answer": <option_number>}}
"""


def safe_int(x: Any, default: int = -1) -> int:
    try: return int(x)
    except Exception: return default


def find_idk(options: List[str]) -> int:
    try: return options.index(IDK_TEXT) + 1
    except ValueError: return len(options)


def options_text(options: List[str]) -> str:
    return "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))


def bucket(ans: int, correct: Optional[int], idk: int) -> int:
    if ans == idk: return 2
    if correct is not None and ans == correct: return 0
    return 1


def bucket_label(b: int) -> str:
    return ["correct", "incorrect", "dk"][b]


def first_non_idk(n: int, idk: int) -> int:
    for i in range(1, n + 1):
        if i != idk: return i
    return 1


def valid_answer(ans: int, n: int, idk: int, allow_idk: bool) -> int:
    if ans < 1 or ans > n: return idk if allow_idk else first_non_idk(n, idk)
    if ans == idk and not allow_idk: return first_non_idk(n, idk)
    return ans


def idk_policy(allow: bool) -> str:
    if allow:
        return "The IDK option is valid; choose it when evidence or memory is insufficient."
    return "Do NOT choose the IDK option. If unsure, make your best non-IDK guess."


def is_detail_question(q: str) -> bool:
    ql = q.lower()
    triggers = ["how many", "what year", "which year", "date", "percent", "%", "according to",
                "name of", "who was", "exact", "statistically", "p-value", "sample size",
                "randomized", "inclusion", "exclusion", "criteria", "confidence interval"]
    return any(t in ql for t in triggers)


def clamp_probs(ps: List[float], lo: float = 0.05, hi: float = 0.65) -> List[float]:
    xs = [min(hi, max(lo, float(p))) for p in ps]
    s = sum(xs)
    return [x / s for x in xs] if s > 0 else [1 / len(xs)] * len(xs)


def sample(cands: List[int], probs: List[float], rng: random.Random) -> int:
    probs = clamp_probs(probs)
    r, acc = rng.random(), 0.0
    for c, p in zip(cands, probs):
        acc += p
        if r <= acc: return int(c)
    return int(cands[-1])


def sample_non_idk(cands: List[int], probs: List[float], idk: int, rng: random.Random) -> int:
    keep = [(safe_int(c), float(p)) for c, p in zip(cands, probs) if safe_int(c) != idk]
    if not keep: return idk
    return sample([c for c, _ in keep], [p for _, p in keep], rng)


def stable_hash(obj: Any) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class JsonCache:
    def __init__(self, path: str):
        self.path = path; self.data = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    r = json.loads(line); self.data[r["key"]] = r["value"]
                except Exception: pass
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def get(self, k): return self.data.get(k)

    def set(self, k, v):
        if k in self.data: return
        self.data[k] = v
        self.f.write(json.dumps({"key": k, "value": v}, ensure_ascii=False) + "\n")


class LLM:
    def __init__(self, model: str, temp: float, cache_path: str, retries: int):
        self.client = OpenAI(); self.model = model; self.temp = temp; self.cache = JsonCache(cache_path); self.retries = retries

    def json(self, system: str, user: str, schema_name: str, schema: Dict[str, Any], max_tokens: int = 160) -> Dict[str, Any]:
        key = stable_hash({"m": self.model, "t": self.temp, "s": system, "u": user, "sn": schema_name, "schema": schema, "mt": max_tokens})
        got = self.cache.get(key)
        if got is not None: return got
        last = None
        for a in range(self.retries):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=self.temp,
                    max_output_tokens=max_tokens,
                    text={"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}},
                )
                obj = json.loads((resp.output_text or "").strip())
                self.cache.set(key, obj)
                return obj
            except Exception as e:
                last = e; time.sleep(min(8, 0.5 * (2 ** a)))
        raise RuntimeError(f"LLM call failed: {last}")


@dataclass(frozen=True)
class Variant:
    name: str
    slug: str
    persona: bool
    memory: bool
    idk: bool
    sampling: bool
    calibration: bool


VARIANTS = [
    Variant("Direct answer", "00_direct_answer", False, False, False, False, False),
    Variant("+ Persona mixture", "01_persona_mixture", True, False, False, False, False),
    Variant("+ Memory bottleneck", "02_memory_bottleneck", True, True, False, False, False),
    Variant("+ IDK option", "03_idk_option", True, True, True, False, False),
    Variant("+ Verbalized sampling", "04_verbalized_sampling", True, True, True, True, False),
    Variant("Full LLMSim", "05_full_llmsim", True, True, True, True, True),
]


def choose_variants(s: Optional[str]) -> List[Variant]:
    if not s: return VARIANTS
    wanted = [x.strip() for x in s.split(",") if x.strip()]
    lut = {v.name: v for v in VARIANTS} | {v.slug: v for v in VARIANTS}
    return [lut[x] for x in wanted]


class Runner:
    def __init__(self, args):
        if args.scripts_dir not in sys.path: sys.path.insert(0, args.scripts_dir)
        import llmsim as base
        self.base = base
        self.args = args
        self.raw = json.load(open(args.dataset, encoding="utf-8"))
        if args.max_docs: self.raw = self.raw[:args.max_docs]
        self.cluster_prompts, self.annotator_to_cluster = base.load_personas(args.persona_path)
        self.llm = LLM(args.model, args.temperature, os.path.join(args.out_root, "cache", "llm_cache.jsonl"), args.max_retries)

    def persona(self, v: Variant, aid: int) -> Tuple[str, str]:
        if not v.persona: return "C_direct", BASE_SYSTEM
        cid = self.annotator_to_cluster.get(int(aid), "C_average_gist")
        return cid, self.cluster_prompts[cid]

    def direct_answer(self, v: Variant, system: str, q: str, opts: List[str], content: Optional[str], media: str) -> int:
        idk = find_idk(opts); opt = options_text(opts)
        if content is None:
            prompt = DIRECT_PRE_PROMPT.format(IDK_POLICY=idk_policy(v.idk))
            user = f"QUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only."
            schema_name = "direct_pre_answer"
        else:
            prompt = DIRECT_POST_PROMPT.format(IDK_POLICY=idk_policy(v.idk))
            user = f"CONTENT TYPE: {media}\nSUPPORTING TEXT:\n{content}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only."
            schema_name = "direct_post_answer"
        obj = self.llm.json(system + "\n\n" + prompt, user, schema_name, SINGLE_ANSWER_SCHEMA, 80)
        return valid_answer(safe_int(obj.get("answer")), len(opts), idk, v.idk)

    def pre(self, v: Variant, system: str, q: str, opts: List[str], rng: random.Random) -> int:
        idk = find_idk(opts)
        if not v.idk:
            return self.direct_answer(v, system, q, opts, None, "pre")
        gate = self.llm.json(system + "\n\n" + self.base.PRE_GATE_PROMPT,
                             f"QUESTION:\n{q}\n\nReturn JSON only.", "pre_gate", self.base.PRE_GATE_SCHEMA, 80)
        if gate["familiarity"] == "technical_or_unknown": return idk
        if v.sampling:
            dist = self.llm.json(system + "\n\n" + self.base.PRE_ANSWER_PROMPT,
                f"FAMILIARITY: {gate['familiarity']}\nCONFIDENCE: {float(gate['confidence']):.2f}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{options_text(opts)}\n\nReturn JSON only.",
                "pre_answer_dist", self.base.ANSWER_DIST_SCHEMA, 120)
            ans = sample([safe_int(x) for x in dist["candidates"]], [float(x) for x in dist["probs"]], rng)
            return valid_answer(ans, len(opts), idk, True)
        prompt = DIRECT_PRE_PROMPT.format(IDK_POLICY=idk_policy(True))
        obj = self.llm.json(system + "\n\n" + prompt,
            f"FAMILIARITY: {gate['familiarity']}\nCONFIDENCE: {float(gate['confidence']):.2f}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{options_text(opts)}\n\nReturn JSON only.",
            "pre_answer_single_idk", SINGLE_ANSWER_SCHEMA, 80)
        return valid_answer(safe_int(obj.get("answer")), len(opts), idk, True)

    def memory(self, v: Variant, system: str, media: str, content: str, q: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        b = self.base
        if media == "news":
            mem = self.llm.json(system + "\n\n" + b.NEWS_MEMORY_DUAL_PROMPT, f"ARTICLE:\n{content}\n\nReturn JSON only.", "news_memory_dual", b.NEWS_MEMORY_DUAL_SCHEMA, 200)
            return mem, None
        if media == "abstract":
            mem = self.llm.json(system + "\n\n" + b.ABSTRACT_MEMORY_DUAL_PROMPT, f"ABSTRACT:\n{content}\n\nReturn JSON only.", "abstract_memory_dual", b.ABSTRACT_MEMORY_DUAL_SCHEMA, 200)
            return mem, None
        if media == "tweet":
            gate = {"explicitness": "implied", "confidence": 1.0}
            if v.idk:
                gate = self.llm.json(system + "\n\n" + b.TWEET_GATE_PROMPT,
                    f"TWEET:\n{content}\n\nQUESTION:\n{q}\n\nReturn JSON only.", "tweet_gate", b.TWEET_GATE_SCHEMA, 80)
            mem = self.llm.json(system + "\n\n" + b.TWEET_MEMORY_DUAL_PROMPT,
                f"EXPLICITNESS: {gate['explicitness']}\nQUESTION:\n{q}\n\nTWEET:\n{content}\n\nReturn JSON only.",
                "tweet_memory_dual", b.TWEET_MEMORY_DUAL_SCHEMA, 160)
            return mem, gate
        raise ValueError(f"unsupported media {media}")

    def choose_trace(self, v: Variant, media: str, mem: Dict[str, Any], gate: Optional[Dict[str, Any]], q: str, rng: random.Random) -> Tuple[str, float, str]:
        probs = clamp_probs([float(mem["trace_probs"][0]), float(mem["trace_probs"][1])], lo=0.10, hi=0.90)
        conf = float(mem["confidence"])
        if media == "tweet" and v.calibration and gate:
            exp, gconf = gate["explicitness"], float(gate["confidence"])
            idx = 1 if exp == "implied" and (gconf < 0.60 or conf < 0.55 or is_detail_question(q)) else (0 if rng.random() < probs[0] else 1)
        else:
            idx = 0 if rng.random() < probs[0] else 1
        tr = mem["traces"][idx]
        return tr["memory"], conf, tr.get("type", "trace")

    def answer_from_trace(self, v: Variant, system: str, media: str, trace: str, conf: float, q: str, opts: List[str], gate: Optional[Dict[str, Any]], rng: random.Random) -> int:
        b = self.base; idk = find_idk(opts); opt = options_text(opts)
        exp = gate.get("explicitness", "implied") if gate else "n/a"
        gconf = float(gate.get("confidence", 1.0)) if gate else 1.0
        if not v.sampling:
            prompt = TRACE_SINGLE_PROMPT.format(IDK_POLICY=idk_policy(v.idk))
            obj = self.llm.json(system + "\n\n" + prompt,
                f"MEDIA: {media}\nEXPLICITNESS: {exp}\nGATE_CONFIDENCE: {gconf:.2f}\nMEMORY TRACE:\n{trace}\nMEMORY_CONFIDENCE: {conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q)}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only.",
                "trace_single_answer", SINGLE_ANSWER_SCHEMA, 80)
            return valid_answer(safe_int(obj.get("answer")), len(opts), idk, v.idk)

        if media == "news":
            prompt, schema_name = b.NEWS_ANSWER_DIST_PROMPT, "news_answer_dist"
            user = f"MEMORY TRACE:\n{trace}\nMEMORY_CONFIDENCE: {conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q)}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only."
        elif media == "abstract":
            prompt, schema_name = b.ABSTRACT_ANSWER_DIST_PROMPT, "abstract_answer_dist"
            user = f"MEMORY TRACE:\n{trace}\nMEMORY_CONFIDENCE: {conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q)}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only."
        else:
            prompt, schema_name = b.TWEET_ANSWER_DIST_PROMPT, "tweet_answer_dist"
            user = f"EXPLICITNESS: {exp}\nGATE_CONFIDENCE: {gconf:.2f}\nINTERPRETATION:\n{trace}\nCONFIDENCE: {conf:.2f}\nDETAIL_QUESTION: {is_detail_question(q)}\n\nQUESTION:\n{q}\n\nOPTIONS:\n{opt}\n\nReturn JSON only."
        dist = self.llm.json(system + "\n\n" + prompt, user, schema_name, b.ANSWER_DIST_SCHEMA, 140)
        cands, probs = [safe_int(x) for x in dist["candidates"]], [float(x) for x in dist["probs"]]
        ans = sample(cands, probs, rng)
        if media == "tweet" and v.calibration and exp == "implied" and ans == idk:
            if not (gconf < self.args.tweet_gate_conf_th and conf < self.args.tweet_mem_conf_th):
                ans = sample_non_idk(cands, probs, idk, rng)
        return valid_answer(ans, len(opts), idk, True)

    def post(self, v: Variant, system: str, media: str, content: str, q: str, opts: List[str], rng: random.Random) -> Tuple[int, Dict[str, Any]]:
        idk = find_idk(opts)
        if media not in ("news", "abstract", "tweet"): return idk, {}
        if not v.memory:
            return self.direct_answer(v, system, q, opts, content, media), {}
        mem, gate = self.memory(v, system, media, content, q)
        diag = {}
        if media == "tweet" and v.idk and gate:
            diag.update({"tweet_explicitness": gate["explicitness"], "tweet_gate_confidence": float(gate["confidence"])})
            if gate["explicitness"] == "unclear": return idk, diag
        trace, conf, ttype = self.choose_trace(v, media, mem, gate, q, rng)
        diag.update({"memory": trace, "memory_confidence": conf, "trace_type": ttype})
        if media == "tweet" and v.calibration and gate:
            exp, gconf = gate["explicitness"], float(gate["confidence"])
            if exp == "implied" and gconf < self.args.tweet_gate_conf_th and conf < self.args.tweet_mem_conf_th: return idk, diag
            if exp == "explicit" and is_detail_question(q) and conf < self.args.tweet_expl_detail_memconf_th: return idk, diag
        return self.answer_from_trace(v, system, media, trace, conf, q, opts, gate, rng), diag

    def existing(self, path: str) -> set:
        out = set()
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                try:
                    r = json.loads(line); out.add((safe_int(r["doc_id"]), safe_int(r["question_index"]), safe_int(r["annotator_id"])))
                except Exception: pass
        return out

    def run_variant(self, v: Variant) -> str:
        rng = random.Random(self.args.seed + safe_int(re.sub(r"\D", "", v.slug), 0))
        outdir = os.path.join(self.args.out_root, v.slug); os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "predictions.jsonl")
        done = self.existing(path); wrote = 0
        with open(path, "a", encoding="utf-8", buffering=1) as w:
            for doc_id, doc in enumerate(self.raw):
                if doc_id % 10 == 0: print(f"[{v.name}] doc {doc_id}/{len(self.raw)} wrote={wrote}")
                media, content = doc["content-type"], doc.get("content", "")
                anns = doc.get("human_annotations") or []
                if not anns: continue
                for q_idx, qref in enumerate(anns[0]["qa_annotations"]):
                    q, opts = qref["question-text"], qref["options"]
                    correct = qref.get("correct_option"); correct = int(correct) if correct is not None else None
                    idk = find_idk(opts)
                    for ann in anns:
                        aid = int(ann["annotator_id"]); key = (doc_id, q_idx, aid)
                        if key in done: continue
                        if self.args.limit_obs and wrote >= self.args.limit_obs: return path
                        cid, system = self.persona(v, aid)
                        try: apre = self.pre(v, system, q, opts, rng)
                        except Exception as e:
                            print(f"pre failed {v.name} {key}: {e}"); apre = idk if v.idk else first_non_idk(len(opts), idk)
                        try: apost, diag = self.post(v, system, media, content, q, opts, rng)
                        except Exception as e:
                            print(f"post failed {v.name} {key}: {e}"); apost, diag = idk, {"error": str(e)}
                        bpre, bpost = bucket(apre, correct, idk), bucket(apost, correct, idk)
                        rec = {"variant": v.name, "doc_id": doc_id, "question_index": q_idx, "media": media,
                               "annotator_id": aid, "cluster_id": cid, "question": q, "correct_option": correct,
                               "model_pre_answer": int(apre), "model_post_answer": int(apost),
                               "classification_pre": bucket_label(bpre), "classification_post": bucket_label(bpost),
                               "diagnostics": diag}
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        done.add(key); wrote += 1
        return path

    def run(self, variants: List[Variant]) -> Dict[str, str]:
        return {v.name: self.run_variant(v) for v in variants}


# ---------------- Evaluation ----------------

def fmean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def raw_dist(c):
    s = sum(c)
    if s <= 0:
        return [float("nan")] * len(c)
    return [x / s for x in c]


def smooth_dist(c, alpha=1e-3):
    xs = [float(x) + alpha for x in c]
    s = sum(xs)
    return [x / s for x in xs]


def kl(p, q):
    return sum(
        max(EPS, pi) * math.log(max(EPS, pi) / max(EPS, qi))
        for pi, qi in zip(p, q)
    )


def pearson(xs, ys):
    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return float("nan")

    xs, ys = zip(*pairs)
    mx, my = fmean(xs), fmean(ys)

    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)

    if vx <= 0 or vy <= 0:
        return float("nan")

    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def load_human(dataset: str, max_docs: Optional[int]):
    raw = json.load(open(dataset, encoding="utf-8"))
    raw = raw[:max_docs] if max_docs else raw

    meta = {}
    hunit = defaultdict(lambda: [0, 0, 0])
    hpre = defaultdict(list)
    hpost = defaultdict(list)

    for did, doc in enumerate(raw):
        media = doc["content-type"]
        anns = doc.get("human_annotations") or []
        if not anns:
            continue

        for qi, qref in enumerate(anns[0]["qa_annotations"]):
            opts = qref["options"]
            idk = find_idk(opts)

            corr = qref.get("correct_option")
            corr = int(corr) if corr is not None else None

            for ann in anns:
                aid = int(ann["annotator_id"])
                qa = ann["qa_annotations"][qi]

                okey = (did, qi, aid)
                ukey = (media, did, qi)
                dkey = (media, did)

                meta[okey] = {
                    "media": media,
                    "correct": corr,
                    "idk": idk,
                    "unit": ukey,
                    "doc": dkey,
                }

                hpre[dkey].append(
                    int(corr is not None and safe_int(qa.get("human-answer-pre")) == corr)
                )

                if qa.get("human-answer-post") is not None and media in ("news", "abstract", "tweet"):
                    ans = safe_int(qa.get("human-answer-post"))
                    hunit[ukey][bucket(ans, corr, idk)] += 1
                    hpost[dkey].append(int(corr is not None and ans == corr))

    hkg = {
        d: fmean(hpost[d]) - fmean(hpre[d])
        for d in set(hpre) & set(hpost)
        if hpre[d] and hpost[d]
    }

    return meta, hunit, hkg


def load_model(path: str, meta):
    munit = defaultdict(lambda: [0, 0, 0])
    mpre = defaultdict(list)
    mpost = defaultdict(list)
    seen = set()

    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue

        key = (
            safe_int(r.get("doc_id")),
            safe_int(r.get("question_index")),
            safe_int(r.get("annotator_id")),
        )

        if key not in meta or key in seen:
            continue

        seen.add(key)
        m = meta[key]

        corr = m["correct"]
        idk = m["idk"]
        media = m["media"]

        pre_ans = safe_int(r.get("model_pre_answer"))
        post_ans = safe_int(r.get("model_post_answer"))

        mpre[m["doc"]].append(
            int(corr is not None and pre_ans == corr)
        )

        if media in ("news", "abstract", "tweet"):
            munit[m["unit"]][bucket(post_ans, corr, idk)] += 1
            mpost[m["doc"]].append(
                int(corr is not None and post_ans == corr)
            )

    mkg = {
        d: fmean(mpost[d]) - fmean(mpre[d])
        for d in set(mpre) & set(mpost)
        if mpre[d] and mpost[d]
    }

    return munit, mkg


def add_counts(a, b):
    for i in range(len(a)):
        a[i] += b[i]


def evaluate(dataset: str, pred_paths: Dict[str, str], out_root: str, max_docs: Optional[int]):
    meta, hunit, hkg = load_human(dataset, max_docs)
    rows = []

    for name, path in pred_paths.items():
        munit, mkg = load_model(path, meta)

        item_kls = []
        correct_maes = []
        idk_maes = []

        global_h = [0, 0, 0]
        global_m = [0, 0, 0]

        for u, hc in hunit.items():
            mc = munit.get(u)

            if not mc or not sum(mc) or not sum(hc):
                continue

            # Global aggregate KL: matches original llmsim.py evaluation style.
            add_counts(global_h, hc)
            add_counts(global_m, mc)

            # Item KL: stricter per document-question diagnostic.
            hp_smooth = smooth_dist(hc)
            mp_smooth = smooth_dist(mc)
            item_kls.append(kl(hp_smooth, mp_smooth))

            # Calibration errors use unsmoothed empirical rates.
            hp_raw = raw_dist(hc)
            mp_raw = raw_dist(mc)
            correct_maes.append(abs(hp_raw[0] - mp_raw[0]))
            idk_maes.append(abs(hp_raw[2] - mp_raw[2]))

        if sum(global_h) > 0 and sum(global_m) > 0:
            global_kl = kl(raw_dist(global_h), raw_dist(global_m))
            global_h_correct = raw_dist(global_h)[0]
            global_m_correct = raw_dist(global_m)[0]
            global_h_incorrect = raw_dist(global_h)[1]
            global_m_incorrect = raw_dist(global_m)[1]
            global_h_idk = raw_dist(global_h)[2]
            global_m_idk = raw_dist(global_m)[2]
        else:
            global_kl = float("nan")
            global_h_correct = global_m_correct = float("nan")
            global_h_incorrect = global_m_incorrect = float("nan")
            global_h_idk = global_m_idk = float("nan")

        docs = sorted(set(hkg) & set(mkg))

        rows.append({
            "variant": name,
            "global_kl": global_kl,
            "item_kl": fmean(item_kls),
            "kg_corr": pearson(
                [hkg[d] for d in docs],
                [mkg[d] for d in docs],
            ),
            "correct_mae": fmean(correct_maes),
            "idk_mae": fmean(idk_maes),
            "human_correct": global_h_correct,
            "model_correct": global_m_correct,
            "human_incorrect": global_h_incorrect,
            "model_incorrect": global_m_incorrect,
            "human_idk": global_h_idk,
            "model_idk": global_m_idk,
            "n_units": len(item_kls),
            "n_docs": len(docs),
            "predictions_path": path,
        })

    os.makedirs(os.path.join(out_root, "tables"), exist_ok=True)

    csv_path = os.path.join(out_root, "tables", "llmsim_ablations.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    write_tex(rows, os.path.join(out_root, "tables", "llmsim_ablations.tex"), compact=False)
    write_tex(rows, os.path.join(out_root, "tables", "llmsim_ablations_compact.tex"), compact=True)

    for r in rows:
        print(
            f"{r['variant']}: "
            f"Global KL={fmt(r['global_kl'])}, "
            f"Item KL={fmt(r['item_kl'])}, "
            f"KG corr={fmt(r['kg_corr'])}, "
            f"Correct MAE={fmt(r['correct_mae'])}, "
            f"IDK MAE={fmt(r['idk_mae'])}"
        )

        print(
            f"  Human vs model aggregate: "
            f"Correct {pct(r['human_correct'])} vs {pct(r['model_correct'])}, "
            f"Incorrect {pct(r['human_incorrect'])} vs {pct(r['model_incorrect'])}, "
            f"IDK {pct(r['human_idk'])} vs {pct(r['model_idk'])}"
        )


def fmt(x):
    return "--" if x is None or not math.isfinite(float(x)) else f"{float(x):.3f}"


def pct(x):
    return "--" if x is None or not math.isfinite(float(x)) else f"{100 * float(x):.1f}%"


def texesc(s):
    return (
        s.replace("&", r"\&")
        .replace("_", r"\_")
        .replace("%", r"\%")
    )


def write_tex(rows, path, compact=False):
    if compact:
        head = """\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lrrrr}
\\toprule
Simulator variant & Global KL $\\downarrow$ & Item KL $\\downarrow$ & KG corr. $\\uparrow$ & IDK MAE $\\downarrow$ \\\\
\\midrule
"""
        body = "".join(
            f"{texesc(r['variant'])} & {fmt(r['global_kl'])} & {fmt(r['item_kl'])} & {fmt(r['kg_corr'])} & {fmt(r['idk_mae'])} \\\\\n"
            for r in rows
        )
        cap = (
            "Ablation of LLMSim components. Global KL matches the aggregate "
            "Correct/Incorrect/IDK evaluation used in the main LLMSim validation; "
            "Item KL averages KL over document--question units."
        )
    else:
        head = """\\begin{table}[t]
\\centering
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Simulator variant & Global KL $\\downarrow$ & Item KL $\\downarrow$ & KG corr. $\\uparrow$ & Correct MAE $\\downarrow$ & IDK MAE $\\downarrow$ \\\\
\\midrule
"""
        body = "".join(
            f"{texesc(r['variant'])} & {fmt(r['global_kl'])} & {fmt(r['item_kl'])} & {fmt(r['kg_corr'])} & {fmt(r['correct_mae'])} & {fmt(r['idk_mae'])} \\\\\n"
            for r in rows
        )
        cap = (
            "Ablation of LLMSim components. Global KL measures aggregate alignment "
            "with the human Correct/Incorrect/IDK distribution. Item KL is a stricter "
            "document--question-level diagnostic. Correct MAE and IDK MAE measure "
            "calibration of simulated correctness and abstention rates."
        )

    tail = f"""\\bottomrule
\\end{{tabular}}
\\caption{{{cap}}}
\\label{{tab:llmsim_ablation}}
\\end{{table}}
"""

    open(path, "w", encoding="utf-8").write(head + body + tail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--persona_path", default="scripts/persona_cards.json")
    p.add_argument("--scripts_dir", default="scripts")
    p.add_argument("--out_root", default=None)
    p.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    p.add_argument("--temperature", type=float, default=float(os.getenv("TEMP", "1.7")))
    p.add_argument("--max_retries", type=int, default=int(os.getenv("MAX_RETRIES", "5")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "0")))
    p.add_argument("--max_docs", type=int, default=None)
    p.add_argument("--limit_obs", type=int, default=None)
    p.add_argument("--variants", default=None)
    p.add_argument("--evaluate_only", action="store_true")
    p.add_argument("--tweet_gate_conf_th", type=float, default=float(os.getenv("TWEET_GATE_CONF_TH", "0.30")))
    p.add_argument("--tweet_mem_conf_th", type=float, default=float(os.getenv("TWEET_MEM_CONF_TH", "0.20")))
    p.add_argument("--tweet_expl_detail_memconf_th", type=float, default=float(os.getenv("TWEET_EXPL_DETAIL_MEMCONF_TH", "0.40")))
    args = p.parse_args()
    if args.out_root is None: args.out_root = os.path.join("runs", "llmsim_ablation_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.out_root, exist_ok=True)
    variants = choose_variants(args.variants)
    if args.evaluate_only:
        pred_paths = {v.name: os.path.join(args.out_root, v.slug, "predictions.jsonl") for v in variants}
    else:
        pred_paths = Runner(args).run(variants)
    evaluate(args.dataset, pred_paths, args.out_root, args.max_docs)


if __name__ == "__main__":
    main()