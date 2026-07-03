#!/usr/bin/env python3
"""Blocker census: invert the propagation direction.
Build a demand table: which single not-yet-existing mechanism would unlock the
most total bytes/points across the 400 tasklogs' recorded blockers.

ANALYSIS-ONLY. Writes reports/blocker_census.md and reports/blocker_census.json.
"""
import json, re, math, os, glob

ROOT = "/Users/minseong/project/neurogolf"
TLOG = os.path.join(ROOT, "reports/tasklog")
MANIFEST = os.path.join(ROOT, "reports/manifest.json")

# ---------------------------------------------------------------- taxonomy
# Each category: list of (weight, regex) signal phrases. Case-insensitive.
# Weights: 3 = near-definitive phrase, 2 = strong, 1 = weak/supporting.
TAX = {
 "per_cell_detection_read": [
   (3, r"per-?cell colour"), (3, r"per-?cell detection"), (3, r"detection floor"),
   (3, r"g[²\^2]?\s*[x×]\s*4"), (2, r"colour-?index plane"), (2, r"color-?index plane"),
   (2, r"detection read"), (2, r"3600\s*b"), (2, r"per-?cell read"),
   (1, r"detection/reconstruction"), (1, r"detection tier"), (1, r"read the (input )?colour"),
   (2, r"one unavoidable fp32 entry plane"), (1, r"native.crop"),
 ],
 "full_output_carrier": [
   (3, r"output carrier"), (3, r"full-?output carrier"), (3, r"900\s*b .{0,25}carrier"),
   (2, r"pad carrier"), (2, r"one-?hot emission"), (2, r"index plane"),
   (2, r"carrier .{0,15}(floor|required|unavoidable|900)"), (2, r"emission .{0,10}floor"),
   (2, r"output carrier, required"), (2, r"900\s*b"),
 ],
 "interval_fill_band": [
   (3, r"interval[- ]fill"), (3, r"interval fill"), (3, r"band floor"),
   (2, r"prefix/?suffix"), (2, r"prefix-?or"), (2, r"suffix-?or"), (2, r"segment between"),
   (2, r"fill the whole segment"), (2, r"beam"), (1, r"1-?d .{0,15}profile"),
   (1, r"row/?col occupancy"), (1, r"triangular"), (2, r"data-?dependent 2-?d interval"),
 ],
 "sprite_stamp_machinery": [
   (3, r"sprite[- ]?stamp"), (3, r"template kernel"), (3, r"dilation bank"), (3, r"scale bank"),
   (2, r"candidate bank"), (2, r"repeat bank"), (2, r"repeat-?d\d"), (2, r"stamp"),
   (2, r"qlinearconv repeat"), (2, r"template bank"), (2, r"sprite"), (1, r"dilation"),
   (1, r"kernel bank"), (1, r"footprint"), (1, r"scale"), (1, r"rotated .{0,15}(body|shape|copy)"),
 ],
 "assignment_correspondence": [
   (3, r"assignment wall"), (3, r"shape-?correspondence"), (3, r"content-?matched"),
   (3, r"correspondence"), (2, r"hash-?match"), (2, r"component pairing"), (2, r"assignment"),
   (2, r"matching is .{0,15}ambiguous"), (2, r"sprite.{0,4}box"), (2, r"pair"),
   (1, r"which sprite"), (1, r"template-?recovery"), (2, r"3.3 match"), (1, r"matcher"),
 ],
 "connectivity_flood": [
   (3, r"flood-?fill"), (3, r"connected[- ]?component"), (3, r"connectivity"),
   (2, r"flood"), (2, r"iterative scan"), (2, r"8-?neighbor"), (2, r"8-?neighbour"),
   (2, r"cc-?label"), (1, r"enclosed"), (1, r"seed/?dilation"), (1, r"propagat"),
   (2, r"closure"), (1, r"iteration depth"),
 ],
 "ambiguity_info_wall": [
   (3, r"information floor"), (3, r"info-?bottleneck"), (3, r"information wall"),
   (3, r"irreducible ambiguity"), (3, r"info-?bottleneck wall"), (2, r"infeasible"),
   (2, r"ambiguous"), (2, r"not encoded in the input"), (2, r"not recoverable from"),
   (2, r"no net .{0,20}can pass"), (2, r"information ceiling"), (2, r"uniformly at random"),
   (2, r"hidden .{0,10}pairing"), (1, r"ambiguity"), (1, r"discarded by the generator"),
   (2, r"impossible"), (1, r"no signal in the input"),
 ],
 "mem0_param_game": [
   (3, r"mem[= ]0\b"), (2, r"params-?only"), (2, r"param-?table"), (2, r"param table"),
   (1, r"params-only floor"), (2, r"mem 0\b"), (1, r"dedup .{0,10}param"),
 ],
 "single_op_floor": [
   (3, r"single-?op floor"), (3, r"already at .{0,15}single"), (2, r"single-?pass"),
   (2, r"single conv"), (2, r"onnx floor"), (2, r"single[- ]op"), (2, r"already at (the )?floor"),
   (1, r"at floor"), (1, r"minimal"), (1, r"no cheaper"),
 ],
 "opset_ort_dtype": [
   (3, r"grader-?killer"), (3, r"cross-?session .{0,15}alias"), (3, r"weight aliasing"),
   (2, r"opset"), (2, r"dtype constraint"), (2, r"uint8 topk"), (2, r"aliasing"),
   (2, r"dirty[- ]?(gate|flip|process)"), (2, r"ort .{0,10}(1\.2|constraint|bug)"),
   (1, r"int64"),
 ],
 "ood_bundled_overlay": [
   (3, r"hardcoded .{0,15}(original|arc|patch)"), (3, r"bundled overlay"), (2, r"ood"),
   (2, r"original-?arc"), (2, r"hardcoded"), (2, r"overlay"), (1, r"stress shape"),
   (2, r"out-of-?distribution"),
 ],
}

CAT_LABEL = {
 "per_cell_detection_read": "per-cell-detection-read floor (G^2x4 fp32 off free input)",
 "full_output_carrier": "full-output-carrier floor (900B index / one-hot emission)",
 "interval_fill_band": "data-dependent 2D interval-fill band floor (~3000B)",
 "sprite_stamp_machinery": "arbitrary-sprite-stamp machinery (dilation/scale/template banks)",
 "assignment_correspondence": "content-matched assignment / correspondence",
 "connectivity_flood": "connectivity / flood iteration state",
 "ambiguity_info_wall": "ambiguity / information wall (unfixable)",
 "mem0_param_game": "mem0 param-game (params-only)",
 "single_op_floor": "already-at-single-op floor",
 "opset_ort_dtype": "opset / ORT dtype constraint",
 "ood_bundled_overlay": "OOD-bundled-example overlay (hardcoded patches)",
 "UNKNOWN": "UNKNOWN / stale-log (thin or no log)",
}

# ---------------------------------------------------------------- helpers
DATE_HDR = re.compile(r"^#{1,3}\s+(20\d\d-\d\d-\d\d|S\d{1,2}\b)", re.M)

def split_sections(text):
    """Return list of (header, body). If no dated headers, one section = whole."""
    idxs = [m.start() for m in DATE_HDR.finditer(text)]
    if not idxs:
        return [("(whole)", text)]
    secs = []
    for i, s in enumerate(idxs):
        e = idxs[i+1] if i+1 < len(idxs) else len(text)
        line_end = text.find("\n", s)
        hdr = text[s:line_end].strip("# \n")
        secs.append((hdr, text[s:e]))
    # prepend any preamble before first dated header (contains **Current** etc.)
    if idxs[0] > 0:
        secs.insert(0, ("(preamble)", text[:idxs[0]]))
    return secs

FLOOR_REGION = re.compile(
    r"(irreducible-?floor analysis|why infeasible|dominant cost|dominant tensor|"
    r"floor analysis|open angles|status:|infeasible|it can'?t be removed|"
    r"the true wall|the .{0,10}floor is)", re.I)

def floor_region_text(text):
    """Concatenate lines in/after floor/why-infeasible/status/open-angles blocks.
    These state the BLOCKER (vs body which states mechanism/attempts)."""
    lines = text.splitlines()
    keep, on = [], False
    for ln in lines:
        if FLOOR_REGION.search(ln):
            on = True
        elif re.match(r"^#{1,3}\s", ln) and not FLOOR_REGION.search(ln):
            on = False  # a new non-floor header ends the region
        if on:
            keep.append(ln)
    return "\n".join(keep)

def classify(text, newest_body, floor_txt):
    """Score categories over full text; newest section 2x, floor-region 3x extra."""
    scores = {c: 0 for c in TAX}
    lo, nlo, flo = text.lower(), newest_body.lower(), floor_txt.lower()
    for c, sigs in TAX.items():
        for w, pat in sigs:
            scores[c] += w * len(re.findall(pat, lo))        # base
            scores[c] += w * len(re.findall(pat, nlo))       # newest x2
            scores[c] += 3 * w * len(re.findall(pat, flo))   # floor region heavy
    return scores

# byte extraction: find largest explicit "NNNN B" figure inside floor/dominant regions
FLOOR_HDR = re.compile(r"(irreducible-?floor|dominant cost|dominant tensor|floor analysis|"
                       r"current graph anatomy|dominant remaining|why infeasible|"
                       r"irreducible-floor analysis)", re.I)
BYTES_RE = re.compile(r"(\d{3,6})\s*b\b", re.I)

def extract_blocker_bytes(text, mem):
    """Try explicit dominant-intermediate byte figure near floor language; else mem*0.6."""
    cands = []
    # scan line windows around floor headers / floor-ish lines
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        window = " ".join(lines[i:i+6])
        if FLOOR_HDR.search(ln) or re.search(r"dominant|floor|irreducible|carrier|entry plane", ln, re.I):
            for m in BYTES_RE.finditer(window):
                v = int(m.group(1))
                if 50 <= v <= max(60000, mem):  # plausible intermediate size
                    cands.append(v)
    if cands and mem > 0:
        # A mechanism-removable blocker is a MEMORY plane => cap at mem. For mem=0
        # (params-only single-op/param floor) there is NO removable plane: the cost
        # IS the irreducible op weights; explicit prose "900B carrier" is a rejected
        # alternative, not a removable intermediate. -> blocker_bytes=0.
        b = min(max(cands), int(mem))
        if b >= 50:
            return b, "explicit"
    # fallback (also yields 0 when mem==0, correctly)
    return int(round(mem * 0.6)), "mem*0.6"

# ---------------------------------------------------------------- load manifest
man = json.load(open(MANIFEST))["tasks"]

def cost_of(t):
    return (t.get("memory", 0) or 0) + (t.get("params", 0) or 0)

def points_of(cost):
    return 25.0 - math.log(max(1, cost))

# ---------------------------------------------------------------- process
records = []
for num in range(1, 401):
    key = str(num)
    m = man.get(key, {})
    mem = m.get("memory", 0) or 0
    params = m.get("params", 0) or 0
    cost = mem + params
    cur_pts = m.get("points", points_of(cost) if cost else 0)
    fp = os.path.join(TLOG, f"task{num:03d}.md")
    rec = {
        "task": num, "arc_method": m.get("method", ""),
        "manifest_mem": mem, "manifest_params": params,
        "manifest_cost": cost, "current_points": round(cur_pts, 4),
        "blocker": "UNKNOWN", "secondary": [], "blocker_bytes": 0,
        "blocker_bytes_src": "none", "confidence": "low",
        "stale": False, "log_present": False, "source_lines": [],
    }
    if not os.path.exists(fp):
        rec["blocker_bytes"] = int(round(mem * 0.6))
        rec["blocker_bytes_src"] = "mem*0.6"
        rec["note"] = "no tasklog file"
        records.append(rec)
        continue
    text = open(fp, encoding="utf-8", errors="replace").read()
    rec["log_present"] = True
    secs = split_sections(text)
    newest_body = secs[-1][1] if secs else text
    # if newest is preamble-only tiny, use whole
    thin = len(re.sub(r"\s+", "", text)) < 400
    floor_txt = floor_region_text(text)
    scores = classify(text, newest_body, floor_txt)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_cat, top_score = ranked[0]
    # decide dominant
    if thin or top_score < 3:
        rec["blocker"] = "UNKNOWN"
        rec["confidence"] = "low"
        rec["note"] = "thin/weak-signal log" if thin else "no category cleared threshold"
    else:
        rec["blocker"] = top_cat
        rec["secondary"] = [c for c, s in ranked[1:3] if s >= max(3, top_score * 0.5)]
        # confidence from margin + score
        second = ranked[1][1]
        margin = top_score - second
        if top_score >= 8 and margin >= 3:
            rec["confidence"] = "high"
        elif top_score >= 5:
            rec["confidence"] = "med"
        else:
            rec["confidence"] = "low"
    # blocker bytes
    bb, src = extract_blocker_bytes(text, mem)
    rec["blocker_bytes"] = bb
    rec["blocker_bytes_src"] = src
    # staleness: gather ALL stated mem figures; NOT stale if any is within 25% of
    # manifest mem (log records the current cost somewhere). Stale only if every
    # stated mem is far from manifest (log describes a retired net).
    logmems = [int(x) for x in re.findall(r"mem(?:ory)?[=:\s]+(\d{2,6})", text, re.I)]
    rec["log_stated_mems"] = logmems[:8]
    if logmems and mem > 0:
        near = min(abs(lm - mem) / mem for lm in logmems)
        if near > 0.25:
            rec["stale"] = True
            if rec["confidence"] == "high": rec["confidence"] = "med"
            elif rec["confidence"] == "med": rec["confidence"] = "low"
            rec["note"] = rec.get("note", "") + f" [stale: no log mem near manifest {mem}]"
    # capture a couple of source lines for the blocker
    src_lines = []
    for ln in text.splitlines():
        if rec["blocker"] != "UNKNOWN":
            for w, pat in TAX[rec["blocker"]]:
                if w >= 2 and re.search(pat, ln, re.I) and len(ln.strip()) > 15:
                    src_lines.append(ln.strip()[:200]); break
        if len(src_lines) >= 3:
            break
    rec["source_lines"] = src_lines
    records.append(rec)

# ---------------------------------------------------------------- aggregate
def unlock_pts(rec):
    cost = rec["manifest_cost"]
    if cost <= 1:
        return 0.0
    bb = min(rec["blocker_bytes"], cost - 1)
    new_cost = max(1, cost - bb)
    return (25.0 - math.log(new_cost)) - rec["current_points"]

for r in records:
    r["unlock_points"] = round(unlock_pts(r), 4)

agg = {}
for r in records:
    c = r["blocker"]
    a = agg.setdefault(c, {"tasks": [], "bytes": 0, "unlock": 0.0})
    a["tasks"].append(r["task"])
    a["bytes"] += r["blocker_bytes"]
    a["unlock"] += r["unlock_points"]

# ---------------------------------------------------------------- write JSON
out_json = {
    "methodology": {
        "cost_model": "cost = mem + params; points = 25 - ln(max(1,cost))",
        "unlock_model": "per task: (25 - ln(max(1, cost - blocker_bytes))) - current_points; blocker_bytes capped at cost-1",
        "blocker_bytes": "explicit largest plausible '<N>B' figure near floor/dominant/carrier language; else mem*0.6 (conservative estimate, flagged)",
        "classification": "weighted keyword taxonomy over full log, newest dated section double-weighted; dominant = top-scoring category if top_score>=3 and log not thin (<400 non-ws chars), else UNKNOWN",
        "staleness": "log-stated 'mem' vs manifest mem mismatch >25% => stale flag, confidence downgraded",
        "caveats": [
            "Keyword classification is heuristic; freeform logs vary in age/quality. Confidence field encodes reliability.",
            "blocker_bytes with src='mem*0.6' is an ESTIMATE, not read from the log. ~majority of tasks use the fallback.",
            "unlock_points is a hypothetical upper bound assuming a mechanism removes the dominant blocker ENTIRELY while leaving the rest of the graph intact; many blockers are proven-irreducible (ambiguity/info walls) => their 'unlock' is aspirational and physically 0.",
            "34 tasks have no tasklog file; bucketed UNKNOWN with mem*0.6 bytes.",
        ],
    },
    "aggregate": {c: {
        "label": CAT_LABEL.get(c, c), "num_tasks": len(a["tasks"]),
        "total_blocker_bytes": a["bytes"], "total_unlock_points": round(a["unlock"], 3),
        "tasks": sorted(a["tasks"]),
    } for c, a in agg.items()},
    "tasks": records,
}
json.dump(out_json, open(os.path.join(ROOT, "reports/blocker_census.json"), "w"), indent=1)

# ---------------------------------------------------------------- write MD
ranked_cats = sorted(agg.items(), key=lambda kv: -kv[1]["unlock"])

def top_members(cat, n=8):
    ms = [r for r in records if r["blocker"] == cat]
    ms.sort(key=lambda r: -r["unlock_points"])
    return ms[:n]

L = []
L.append("# Blocker Census — inverted demand table\n")
L.append("_Generated by reports/scripts/blocker_census.py. ANALYSIS-ONLY._\n")
L.append("**Question inverted:** not 'new insight -> which tasks' but 'which SINGLE not-yet-existing "
         "mechanism would unlock the most total points across the 400 tasklogs' recorded blockers'.\n")
L.append("## Cost / unlock model\n")
L.append("- `cost = mem + params`; `points = 25 - ln(max(1,cost))` (verified: task001 mem0/params240 -> 19.52).")
L.append("- Per-task hypothetical unlock = `(25 - ln(max(1, cost - blocker_bytes))) - current_points`, "
         "blocker_bytes capped at `cost-1`.")
L.append("- `blocker_bytes`: explicit largest plausible `<N>B` figure near floor/dominant/carrier language "
         "where the log states one; otherwise the conservative fallback `mem*0.6` (flagged per task).\n")

tot_unlock = sum(a["unlock"] for a in agg.values())
L.append(f"**Total hypothetical unlock across all categories: {tot_unlock:.1f} pts** "
         f"(upper bound; see caveats — info/ambiguity walls are physically un-unlockable).\n")

L.append("## Ranked demand table\n")
L.append("| rank | blocker category | #tasks | total blocker bytes | est. unlock pts (upper bound) |")
L.append("|---:|---|---:|---:|---:|")
for i, (c, a) in enumerate(ranked_cats, 1):
    L.append(f"| {i} | {CAT_LABEL.get(c,c)} | {len(a['tasks'])} | {a['bytes']:,} | {a['unlock']:.1f} |")
L.append("")

# confidence + bytes-source breakdown
n_explicit = sum(1 for r in records if r["blocker_bytes_src"] == "explicit")
n_fallback = sum(1 for r in records if r["blocker_bytes_src"] == "mem*0.6")
conf = {}
for r in records:
    conf[r["confidence"]] = conf.get(r["confidence"], 0) + 1
stale_n = sum(1 for r in records if r["stale"])
L.append("## Interpretation — where the demand is REAL vs a proven wall\n")
L.append("The two biggest categories are the structural read/write floors that project memory "
         "already marks PROVEN-IRREDUCIBLE:")
L.append("- **full-output-carrier** and **per-cell-detection-read** heavily CO-OCCUR (a task pays "
         "both a 900B one-hot/label carrier to emit AND a G^2x4 fp32 slice to read). Their combined "
         "~300pt 'unlock' is the single largest demand — but memory `structural-ceiling-7800` / "
         "`detection-floor measured 7 ways` says no cheaper ONNX read/write path exists. So a mechanism "
         "that broke the 30x30 detection+carrier floor would be the highest-value invention by far, "
         "yet is the one most tested-against and believed impossible. Treat these two rows as the "
         "'wall demand': aspirational ceiling, not a shovel-ready lever.")
L.append("- The **actionable** demand — mechanisms not yet proven impossible — is concentrated in "
         "**sprite-stamp machinery** (dilation/scale/template banks materialized because ONNX has no "
         "control-flow branch-select), **connectivity/flood iteration state**, and the "
         "**data-dependent 2D interval-fill band**. A single mechanism collapsing the candidate-bank "
         "materialization (compute only the selected branch) is the most credible net-new lever.")
L.append("- **ambiguity/information wall** and **already-at-single-op / mem0 param-game** rows are "
         "floors by construction: their 'unlock' is ~0 by design (info not in the input; params ARE "
         "the irreducible op). Listed for completeness; do not chase.\n")

L.append("## Reliability snapshot\n")
L.append(f"- blocker_bytes source: **{n_explicit} explicit** (read from log), **{n_fallback} mem*0.6 fallback** (estimate).")
L.append(f"- classification confidence: " + ", ".join(f"{k}={v}" for k, v in sorted(conf.items())) + ".")
L.append(f"- stale logs flagged (log mem vs manifest mismatch >25%): **{stale_n}**.\n")

L.append("## Per-category member lists (top-8 by unlock size)\n")
for c, a in ranked_cats:
    L.append(f"### {CAT_LABEL.get(c,c)}  —  {len(a['tasks'])} tasks, {a['unlock']:.1f} pts upper-bound\n")
    L.append("| task | cost(mem+params) | cur pts | blocker bytes | src | unlock pts | conf | note |")
    L.append("|---:|---:|---:|---:|:--:|---:|:--:|---|")
    for r in top_members(c):
        note = (r.get("note", "") or "").strip()[:60]
        L.append(f"| {r['task']} | {r['manifest_cost']:,} | {r['current_points']:.2f} | "
                 f"{r['blocker_bytes']:,} | {r['blocker_bytes_src'][:3]} | {r['unlock_points']:.3f} | "
                 f"{r['confidence']} | {note} |")
    L.append("")

L.append("## Methodology caveats\n")
for cv in out_json["methodology"]["caveats"]:
    L.append(f"- {cv}")
L.append("")
L.append("Files: `reports/blocker_census.md`, `reports/blocker_census.json`, generator `reports/scripts/blocker_census.py`.")

open(os.path.join(ROOT, "reports/blocker_census.md"), "w").write("\n".join(L))

# ---------------------------------------------------------------- console summary
print("=== RANKED DEMAND TABLE (by upper-bound unlock pts) ===")
for i, (c, a) in enumerate(ranked_cats, 1):
    print(f"{i:2d}. {CAT_LABEL.get(c,c)[:52]:52s} #{len(a['tasks']):3d}  {a['unlock']:7.1f} pts  {a['bytes']:>8,}B")
print()
biggest = ranked_cats[0]
print(f"BIGGEST DEMAND: {CAT_LABEL[biggest[0]]}")
print("  top members:", [(r['task'], round(r['unlock_points'],2)) for r in top_members(biggest[0])])
uk = agg.get("UNKNOWN", {"tasks": [], "bytes": 0, "unlock": 0})
print(f"UNKNOWN bucket: {len(uk['tasks'])} tasks, {uk['bytes']:,}B, {uk['unlock']:.1f} pts")
print(f"bytes-src: explicit={n_explicit} fallback={n_fallback}; conf={conf}; stale={stale_n}")
print(f"TOTAL upper-bound unlock = {tot_unlock:.1f} pts")
