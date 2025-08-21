import os, io, json, time, zipfile, pathlib, hashlib, re, datetime as dt
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
from shutil import copyfile

# ----- Paths & config -----
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "dist"
WEB  = ROOT / "web"
CONFIG = json.loads((ROOT / "config" / "settings.json").read_text())
REPOS  = [l.strip() for l in (ROOT / "config" / "repos.txt").read_text().splitlines()
          if l.strip() and not l.strip().startswith("#")]
OUT.mkdir(parents=True, exist_ok=True)

# ----- GitHub API -----
API = "https://api.github.com"
TOKEN = os.getenv("ORG_WIDE_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
HDRS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HDRS["Authorization"] = f"Bearer {TOKEN}"

def gh_get(url: str) -> Any:
    r = requests.get(url, headers=HDRS, timeout=60); r.raise_for_status(); time.sleep(0.2); return r.json()

def gh_get_bin(url: str) -> bytes:
    r = requests.get(url, headers=HDRS, timeout=180); r.raise_for_status(); time.sleep(0.2); return r.content

# ----- artifact selection -----
ARTIFACT_PREFIX = CONFIG["artifact_name_prefix"]
CSV_CANDS = CONFIG["csv_name_candidates"]
FAIL_IF_MISSING_CSV = CONFIG.get("fail_if_missing_csv", True)
_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

def parse_name_ts(name: str) -> Optional[dt.datetime]:
    m = _TS_RE.search(name or ""); 
    if not m: return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)

def list_artifacts(owner: str, repo: str):
    page=1
    while True:
        data = gh_get(f"{API}/repos/{owner}/{repo}/actions/artifacts?per_page=100&page={page}")
        items = data.get("artifacts", []) or []
        if not items: break
        for a in items: yield a
        if len(items)<100: break
        page += 1

def looks_right(a: dict, repo: str) -> bool:
    n = a.get("name","")
    return n.startswith(ARTIFACT_PREFIX) and not a.get("expired") and (f"-{repo}-" in n)

def latest_history(owner: str, repo: str) -> Optional[dict]:
    cands=[]
    for a in list_artifacts(owner, repo):
        if not looks_right(a, repo): continue
        ts = parse_name_ts(a.get("name",""))
        created = a.get("created_at")
        created_dt = None
        if created:
            try: created_dt = dt.datetime.fromisoformat(created.replace("Z","+00:00"))
            except: pass
        score = ts or created_dt or dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
        cands.append((a, score))
    if not cands: return None
    cands.sort(key=lambda t: t[1], reverse=True)
    return cands[0][0]

def find_csv(zip_bytes: bytes):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for want in CSV_CANDS:
            if want in names:
                with zf.open(want) as f: return want, f.read()
        for n in names:
            if n.lower().endswith(".csv"):
                with zf.open(n) as f: return n, f.read()
    return None

# ----- parsing violations_by_location -----
ENTRY_SPLIT = re.compile(r"\s*;\s*")
ENTRY_RE = re.compile(r"^(?P<spec>[^:]+):(?P<file>.*):(?P<line>\d+)(?:=(?P<count>\d+))?$")

def stable_id(file: str, line: int) -> str:
    return hashlib.sha1(f"{file}|{line}".encode("utf-8")).hexdigest()[:12]

def parse_vloc_cell(project: str, commit: str, algo: str, cell: str) -> List[dict]:
    tool = "pymop" if "pymop" in algo.lower() else ("dylin" if "dylin" in algo.lower() else algo.lower())
    out=[]
    if not cell: return out
    for entry in ENTRY_SPLIT.split(cell.strip("; ")):
        if not entry: continue
        m = ENTRY_RE.match(entry)
        if not m: continue
        g = m.groupdict()
        out.append({
            "project": project, "commit": commit, "tool": tool,
            "spec": g["spec"], "file": g["file"], "line": int(g["line"]),
            "count": int(g.get("count") or 1),
        })
    return out

def build():
    projects_out = []
    total_commits = 0
    total_locations = 0

    for full in REPOS:
        owner, repo = full.split("/", 1)
        art = latest_history(owner, repo)
        proj = {
            "slug": full.replace("/","-").lower(),
            "full_name": full,
            "latest_artifact_name": art.get("name") if art else None,
            "commits": []  # filled below
        }

        parsed_rows=[]
        if art:
            zip_bytes = gh_get_bin(art["archive_download_url"])
            found = find_csv(zip_bytes)
            if not found:
                if FAIL_IF_MISSING_CSV: 
                    raise FileNotFoundError(f"No CSV found in artifact {art['name']}")
            else:
                csv_name, csv_bytes = found
                df = pd.read_csv(io.BytesIO(csv_bytes))
                df.columns = [c.strip().lower() for c in df.columns]
                need = {"project","commit_sha","algorithm","violations_by_location"}
                if not need.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")
                df_use = df[df["algorithm"].str.lower().str.contains("pymop|dylin", na=False)]
                for _, r in df_use.iterrows():
                    parsed_rows.extend(
                        parse_vloc_cell(str(r["project"]), str(r["commit_sha"]), str(r["algorithm"]), str(r["violations_by_location"]))
                    )

        if parsed_rows:
            agg = (
                pd.DataFrame(parsed_rows)
                .groupby(["project","commit","tool","file","line","spec"], dropna=False)["count"]
                .sum().reset_index()
            )
            # totals per tool per location
            tool_totals = (
                agg.groupby(["project","commit","tool","file","line"], dropna=False)["count"]
                .sum().rename("tool_total").reset_index()
            )
            tool_pivot = tool_totals.pivot_table(
                index=["project","commit","file","line"], columns="tool", values="tool_total", fill_value=0
            ).reset_index()
            for t in ["pymop","dylin"]:
                if t not in tool_pivot.columns: tool_pivot[t]=0
            tool_pivot["_id"] = tool_pivot.apply(lambda r: stable_id(r["file"], int(r["line"])), axis=1)
            tool_pivot["flagged_by"] = tool_pivot.apply(
                lambda r: "both" if r["pymop"]>0 and r["dylin"]>0 else ("pymop" if r["pymop"]>0 else ("dylin" if r["dylin"]>0 else "none")),
                axis=1
            )

            # spec breakdown by location
            spec_breakdown = (
                agg.pivot_table(index=["project","commit","file","line","spec"], columns="tool", values="count", fill_value=0)
                .reset_index()
            )
            for t in ["pymop","dylin"]:
                if t not in spec_breakdown.columns: spec_breakdown[t]=0

            # commit list for this project
            commit_keys = sorted(tool_pivot["commit"].unique().tolist())
            for sha in commit_keys:
                rows = tool_pivot[tool_pivot["commit"]==sha].copy().sort_values(["file","line"])
                merged = rows.merge(
                    spec_breakdown, on=["project","commit","file","line"], how="left", suffixes=("","_spec")
                )
                specs_series = (
                    merged.groupby(["file","line","_id"])
                    .apply(lambda g: ";".join(sorted([str(s) for s in g["spec"].unique() if str(s) not in ["nan","None",""]])))
                    .rename("specs").reset_index()
                )
                rows = rows.merge(specs_series, on=["file","line","_id"], how="left")

                commit_obj = {
                    "sha": sha,
                    "counts": {
                        "locations": int(rows.shape[0]),
                        "both": int((rows["flagged_by"]=="both").sum()),
                        "pymop_only": int((rows["flagged_by"]=="pymop").sum()),
                        "dylin_only": int((rows["flagged_by"]=="dylin").sum())
                    },
                    "violations": []
                }

                # attach violation detail + per-spec breakdown
                for _, r in rows.iterrows():
                    loc_specs = spec_breakdown[
                        (spec_breakdown["project"]==proj["full_name"]) &
                        (spec_breakdown["commit"]==sha) &
                        (spec_breakdown["file"]==r["file"]) &
                        (spec_breakdown["line"]==r["line"])
                    ][["spec","pymop","dylin"]]
                    commit_obj["violations"].append({
                        "id": r["_id"], "file": r["file"], "line": int(r["line"]),
                        "flagged_by": r["flagged_by"], "pymop": int(r.get("pymop",0)), "dylin": int(r.get("dylin",0)),
                        "specs": r.get("specs","") or "",
                        "breakdown": [
                            {"spec": str(x["spec"]), "pymop": int(x["pymop"]), "dylin": int(x["dylin"])}
                            for _, x in loc_specs.iterrows()
                        ]
                    })

                proj["commits"].append(commit_obj)

            total_commits += len(commit_keys)
            total_locations += int(tool_pivot.shape[0])

        projects_out.append(proj)

    # write data.json
    payload = {
        "generated_at": int(time.time()),
        "projects": projects_out,
        "totals": {"commits": total_commits, "locations": total_locations}
    }
    (OUT / "data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # copy SPA
    copyfile(WEB / "index.html", OUT / "index.html")

if __name__ == "__main__":
    build()