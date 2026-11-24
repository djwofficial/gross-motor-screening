"""Agreement tables: app vs KEE, KEE vs KEE, and KEE's practice clips vs the answer key.
Owner: Theresia (#54).

BUILT AND TESTED ON FAKE DATA ONLY. Testing-day files (Nov 29) never go into an AI tool: run
this script on the live laptop / encrypted drive and keep the output there.

    python -m eval.agreement --fake runs/fake_agreement           # fake files + tables (demo)
    python -m eval.agreement --log runs/2026-11-29/log.csv --kee kee_scores.csv
    python -m eval.agreement --practice kee_practice.csv --key answer_key.csv

Inputs (CSV, UTF-8):
- app log: the app's log.csv (app.session.LOG_FIELDS). Uses child, task, try, score and
  check_by_teacher. A "check by teacher" try has no app score; it is counted, not compared.
- KEE sheet: the paper scoring sheets typed in, one row per try: code, task, try, kee1, kee2
  (the two KEE scorers, 0/1/2; blank = not scored).
- practice: clip, score (KEE's score for each practice clip; a "scorer" column is optional).
- answer key: clip, task, score (the dataset's expert score, kept away from KEE until the end).

Per task and for all tasks: n compared, exact agreement, within one point, quadratic weighted
kappa (sklearn cohen_kappa_score, the same as model/baseline.py) and the 3x3 table of counts.
Kappa is "n/a" when it is not defined (e.g. every score the same). Small n: read the counts,
not only the kappa. A screening aid: these tables say how often the app agrees with KEE, nothing
about any child.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCORES = (0, 1, 2)
SCORE_NAMES = {0: "0 Bad", 1: "1 Good", 2: "2 Perfect"}
TASK_NAMES = {1: "stand on one leg", 2: "hop on one leg", 3: "long jump"}
KEE_FIELDS = ["code", "task", "try", "kee1", "kee2"]


def score(value) -> int | None:
    """'0' / '1' / '2' (also '2.0', ' 1 ') -> int; anything else (blank, 'check') -> None."""
    v = str(value if value is not None else "").strip()
    try:
        f = float(v)
    except ValueError:
        return None
    return int(f) if f in SCORES else None


def _read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _task(value) -> int:
    return int(str(value).strip()[0])  # "1", "1 one leg" -> 1


# ------------------------------------------------------------------ the numbers


def compare(a, b) -> dict:
    """Two score lists (None = not scored) -> n, exact, within_1, kappa, table (3x3 counts)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    table = np.zeros((3, 3), dtype=int)
    for x, y in pairs:
        table[x, y] += 1
    n = len(pairs)
    res = {"n": n, "table": table, "exact": None, "within_1": None, "kappa": None}
    if n == 0:
        return res
    x = np.array([p[0] for p in pairs])
    y = np.array([p[1] for p in pairs])
    res["exact"] = float(np.mean(x == y))
    res["within_1"] = float(np.mean(np.abs(x - y) <= 1))
    if len(set(x) | set(y)) > 1:
        import warnings

        from sklearn.metrics import cohen_kappa_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = cohen_kappa_score(x, y, labels=list(SCORES), weights="quadratic")
        res["kappa"] = None if np.isnan(k) else float(k)
    return res


def by_task(rows, a: str, b: str) -> dict:
    """{1: compare, 2: ..., 3: ..., "all": ...} for columns a and b of rows (with "task")."""
    out = {}
    for task in (1, 2, 3):
        rs = [r for r in rows if r["task"] == task]
        out[task] = compare([r[a] for r in rs], [r[b] for r in rs])
    out["all"] = compare([r[a] for r in rows], [r[b] for r in rows])
    return out


# ------------------------------------------------------------------ reading the files


def join(log_path, kee_path) -> dict:
    """App log + KEE sheet -> rows {code, task, try, app, check, kee1, kee2} and what didn't match."""
    app = {}
    for r in _read_csv(log_path):
        key = (r["child"].strip(), _task(r["task"]), int(r["try"]))
        check = str(r.get("check_by_teacher", "")).strip() in ("1", "True", "true", "yes")
        app[key] = {"app": None if check else score(r.get("score")), "check": check}
    rows, kee_only = [], []
    for r in _read_csv(kee_path):
        if not (r.get("code") or "").strip():
            continue
        key = (r["code"].strip(), _task(r["task"]), int(r["try"]))
        a = app.pop(key, None)
        if a is None:
            kee_only.append(key)
            continue
        rows.append(
            {
                "code": key[0],
                "task": key[1],
                "try": key[2],
                **a,
                "kee1": score(r.get("kee1")),
                "kee2": score(r.get("kee2")),
            }
        )
    return {"rows": rows, "app_only": sorted(app), "kee_only": kee_only}


def practice(practice_path, key_path) -> dict:
    """KEE's practice scores vs the answer key -> {"rows", "tables" (per scorer), "missing"}."""
    key = {r["clip"].strip(): (_task(r["task"]), score(r["score"])) for r in _read_csv(key_path)}
    by_scorer: dict[str, list[dict]] = {}
    missing = []
    for r in _read_csv(practice_path):
        clip = (r.get("clip") or "").strip()
        if not clip:
            continue
        if clip not in key:
            missing.append(clip)
            continue
        task, expert = key[clip]
        who = (r.get("scorer") or "KEE").strip() or "KEE"
        by_scorer.setdefault(who, []).append(
            {"clip": clip, "task": task, "kee": score(r.get("score")), "key": expert}
        )
    tables = {who: by_task(rs, "kee", "key") for who, rs in by_scorer.items()}
    return {"rows": by_scorer, "tables": tables, "missing": missing}


# ------------------------------------------------------------------ printing


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.0f}%"


def _k(v):
    return "n/a" if v is None else f"{v:.2f}"


def table_md(title: str, res: dict, a_name: str, b_name: str) -> str:
    lines = [
        f"### {title}",
        "",
        "| Task | n | Exact | Within 1 | Kappa (quadratic) |",
        "|---|---|---|---|---|",
    ]
    for task in (1, 2, 3, "all"):
        c = res[task]
        name = "all" if task == "all" else f"{task} {TASK_NAMES[task]}"
        lines.append(
            f"| {name} | {c['n']} | {_pct(c['exact'])} | {_pct(c['within_1'])} | {_k(c['kappa'])} |"
        )
    t = res["all"]["table"]
    lines += [
        "",
        f"Counts, all tasks (rows {a_name}, columns {b_name}):",
        "",
        "| | " + " | ".join(SCORE_NAMES[s] for s in SCORES) + " |",
        "|---|---|---|---|",
    ]
    for s in SCORES:
        lines.append(f"| {a_name} {SCORE_NAMES[s]} | " + " | ".join(str(v) for v in t[s]) + " |")
    return "\n".join(lines)


def testing_day_report(joined: dict) -> str:
    rows = joined["rows"]
    n_check = sum(r["check"] for r in rows)
    for r in rows:  # KEE consensus: the score both scorers gave, else not compared
        r["kee_both"] = r["kee1"] if r["kee1"] is not None and r["kee1"] == r["kee2"] else None
    parts = [
        "## Agreement: app vs KEE, KEE vs KEE",
        "",
        f"{len(rows)} tries in both files; {n_check} were 'check by teacher' (no app score, not "
        "compared).",
        "",
        table_md("App vs KEE scorer 1", by_task(rows, "app", "kee1"), "app", "KEE 1"),
        "",
        table_md("App vs KEE scorer 2", by_task(rows, "app", "kee2"), "app", "KEE 2"),
        "",
        table_md(
            "App vs KEE where both scorers agree", by_task(rows, "app", "kee_both"), "app", "KEE"
        ),
        "",
        table_md("KEE scorer 1 vs KEE scorer 2", by_task(rows, "kee1", "kee2"), "KEE 1", "KEE 2"),
    ]
    if joined["app_only"] or joined["kee_only"]:
        parts += [
            "",
            f"Not matched: {len(joined['app_only'])} tries only in the app log, "
            f"{len(joined['kee_only'])} only on the KEE sheet (check the codes and try numbers).",
        ]
    return "\n".join(parts)


def practice_report(p: dict) -> str:
    parts = ["## KEE practice clips vs the answer key"]
    for who, res in p["tables"].items():
        parts += ["", table_md(f"{who} vs answer key", res, who, "key")]
    if p["missing"]:
        parts += ["", f"Clips not in the answer key: {', '.join(p['missing'])}"]
    return "\n".join(parts)


# ------------------------------------------------------------------ fake files (tests, demo)


def make_fake_files(out_dir, n_children: int = 12, tries: int = 4, seed: int = 0) -> dict:
    """FAKE app log, KEE sheet, practice sheet and answer key, to build and test the tables.

    Codes F01.. are made up. The app agrees with KEE about 70% of the time (a made-up number).
    """
    from app.session import LOG_FIELDS

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def near(s, p_same):
        return int(s) if rng.random() < p_same else int(np.clip(s + rng.choice([-1, 1]), 0, 2))

    log_rows, kee_rows = [], []
    for c in range(1, n_children + 1):
        code = f"F{c:02d}"
        for task in (1, 2, 3):
            for k in range(1, tries + 1):
                truth = int(rng.choice(SCORES, p=[0.3, 0.4, 0.3]))
                check = rng.random() < 0.1
                log_rows.append(
                    {
                        "child": code,
                        "task": task,
                        "try": k,
                        "leg": "" if task == 3 else "LR"[k % 2],
                        "score": "" if check else near(truth, 0.7),
                        "check_by_teacher": int(check),
                    }
                )
                kee_rows.append(
                    {
                        "code": code,
                        "task": task,
                        "try": k,
                        "kee1": near(truth, 0.85),
                        "kee2": near(truth, 0.85),
                    }
                )
    paths = {"log": out / "log.csv", "kee": out / "kee_scores.csv"}
    with open(paths["log"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, restval="")
        w.writeheader()
        w.writerows(log_rows)
    with open(paths["kee"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEE_FIELDS)
        w.writeheader()
        w.writerows(kee_rows)

    key_rows = [
        {"clip": f"clip_{i:02d}", "task": 1 + (i - 1) % 3, "score": int(rng.choice(SCORES))}
        for i in range(1, 21)
    ]
    paths["key"] = out / "answer_key.csv"
    paths["practice"] = out / "kee_practice.csv"
    with open(paths["key"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "task", "score"])
        w.writeheader()
        w.writerows(key_rows)
    with open(paths["practice"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "scorer", "score"])
        w.writeheader()
        for who in ("KEE 1", "KEE 2"):
            for r in key_rows:
                w.writerow({"clip": r["clip"], "scorer": who, "score": near(r["score"], 0.8)})
    return paths


# ------------------------------------------------------------------ command line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Agreement tables (app vs KEE, KEE vs KEE, practice)")
    ap.add_argument("--log", help="the app's log.csv")
    ap.add_argument("--kee", help="KEE scoring sheets typed in (code, task, try, kee1, kee2)")
    ap.add_argument("--practice", help="KEE practice scores (clip, [scorer], score)")
    ap.add_argument("--key", help="answer key of the practice clips (clip, task, score)")
    ap.add_argument("--fake", help="write FAKE files into this folder and print their tables")
    ap.add_argument("-o", "--out", help="also write the tables to this .md file")
    args = ap.parse_args(argv)
    if args.fake:
        p = make_fake_files(args.fake)
        args.log, args.kee, args.practice, args.key = p["log"], p["kee"], p["practice"], p["key"]
    parts = []
    if args.log or args.kee:
        if not (args.log and args.kee):
            ap.error("--log and --kee go together")
        parts.append(testing_day_report(join(args.log, args.kee)))
    if args.practice or args.key:
        if not (args.practice and args.key):
            ap.error("--practice and --key go together")
        parts.append(practice_report(practice(args.practice, args.key)))
    if not parts:
        ap.error("give --log and --kee, --practice and --key, or --fake FOLDER")
    text = "\n\n".join(parts)
    if args.fake:
        text = "**FAKE DATA (made-up scores, for building the tables only)**\n\n" + text
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
