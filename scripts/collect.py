import os, io, json, time, zipfile, pathlib, hashlib, re, datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
from shutil import copyfile

# ---------- Paths & config ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "dist"
WEB  = ROOT / "web"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = json.loads((ROOT / "config" / "settings.json").read_text())
REPOS  = [l.strip() for l in (ROOT / "config" / "repos.txt").read_text().splitlines()
          if l.strip() and not l.strip().startswith("#")]

MON_PREFIX = CONFIG["monitoring_prefix"]
HIS_PREFIX = CONFIG["history_prefix"]
CSV_CANDS  = CONFIG["csv_name_candidates"]
FAIL_IF_MISSING_CSV = CONFIG.get("fail_if_missing_csv", True)

# ---------- GitHub API ----------
API = "https://api.github.com"
TOKEN = os.getenv("ORG_WIDE_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
HDRS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HDRS["Authorization"] = f"Bearer {TOKEN}"

def gh_get(url: str) -> Any:
    r = requests.get(url, headers=HDRS, timeout=60)
    r.raise_for_status()
    time.sleep(0.2)  # politeness
    return r.json()

def gh_get_bin(url: str) -> bytes:
    r = requests.get(url, headers=HDRS, timeout=180)
    r.raise_for_status()
    time.sleep(0.2)
    return r.content

# ---------- Artifact helpers ----------
_TS_IN_NAME = re.compile(r"(\d{8}T\d{6}Z)")

def parse_name_ts(name: str) -> Optional[dt.datetime]:
    m = _TS_IN_NAME.search(name or "")
    if not m: return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)

def list_artifacts(owner: str, repo: str):
    page = 1
    while True:
        data = gh_get(f"{API}/repos/{owner}/{repo}/actions/artifacts?per_page=100&page={page}")
        items = data.get("artifacts", []) or []
        if not items: break
        for a in items: yield a
        if len(items) < 100: break
        page += 1

def looks_with_prefix(a: dict, repo: str, prefix: str) -> bool:
    n = a.get("name", "")
    return n.startswith(prefix) and not a.get("expired") and (f"-{repo}-" in n)

def latest_with_prefix(owner: str, repo: str, prefix: str) -> Optional[dict]:
    cands = []
    for a in list_artifacts(owner, repo):
        if not looks_with_prefix(a, repo, prefix): continue
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

def find_csv(zip_bytes: bytes, candidates: List[str]) -> Optional[Tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for want in candidates:
            if want in names:
                with zf.open(want) as f: return want, f.read()
        # fallback: any .csv
        for n in names:
            if n.lower().endswith(".csv"):
                with zf.open(n) as f: return n, f.read()
    return None

# ---------- Data parsing ----------
ENTRY_SPLIT = re.compile(r"\s*;\s*")
# e.g. "Spec:/path/file.py:123" or "Spec:/path/file.py:123=COUNT"
ENTRY_RE = re.compile(r"^(?P<spec>[^:]+):(?P<file>.*):(?P<line>\d+)(?:=(?P<count>\d+))?$")

def stable_loc_id(file: str, line: int) -> str:
    return hashlib.sha1(f"{file}|{line}".encode("utf-8")).hexdigest()[:12]

def parse_vloc_cell(cell: str) -> List[dict]:
    out = []
    if not cell: return out
    for entry in ENTRY_SPLIT.split(str(cell).strip("; ")):
        if not entry: continue
        m = ENTRY_RE.match(entry)
        if not m: continue
        g = m.groupdict()
        out.append({
            "spec": g["spec"],
            "file": g["file"],
            "line": int(g["line"]),
            "count": int(g.get("count") or 1),
        })
    return out

def to_epoch(ts_val) -> Optional[int]:
    """Parse various timestamp formats to epoch seconds (UTC)."""
    if ts_val is None: return None
    s = str(ts_val).strip()
    if not s or s.lower()=="nan": return None
    if s.isdigit() and len(s) in (10, 13):
        # 10 = seconds, 13 = milliseconds
        return int(int(s) / (1000 if len(s)==13 else 1))
    # Normalize trailing Z
    s_norm = s.replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dtm = dt.datetime.strptime(s_norm, fmt)
            if dtm.tzinfo is None: dtm = dtm.replace(tzinfo=dt.timezone.utc)
            return int(dtm.timestamp())
        except Exception:
            pass
    try:
        d = pd.to_datetime([s], utc=True, errors="coerce")[0]
        if pd.notna(d): return int(d.to_pydatetime().timestamp())
    except Exception:
        pass
    return None

def build_dataset(prefix: str) -> dict:
    projects_out = []
    total_commits = 0
    total_locations = 0

    for full in REPOS:
        owner, repo = full.split("/", 1)
        art = latest_with_prefix(owner, repo, prefix)

        proj = {
            "slug": full.replace("/","-").lower(),
            "full_name": full,
            "latest_artifact_name": art.get("name") if art else None,
            "commits": []
        }

        if art:
            zip_bytes = gh_get_bin(art["archive_download_url"])
            found = find_csv(zip_bytes, CSV_CANDS)
            if not found:
                if FAIL_IF_MISSING_CSV:
                    raise FileNotFoundError(f"No CSV found in artifact {art['name']}")
            else:
                csv_name, csv_bytes = found
                df = pd.read_csv(io.BytesIO(csv_bytes))
                df.columns = [c.strip().lower() for c in df.columns]

                required = {"project","commit_sha","timestamp","violations_by_location"}
                if not required.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns; found {df.columns.tolist()}")

                # Parse all rows into (commit -> violations)
                commits_map: Dict[str, dict] = {}

                for _, r in df.iterrows():
                    sha = str(r["commit_sha"])
                    ts  = to_epoch(r["timestamp"])
                    locs = parse_vloc_cell(r["violations_by_location"])

                    if sha not in commits_map:
                        commits_map[sha] = {
                            "sha": sha,
                            "ts": ts,
                            "violations_raw": []  # collect per-row entries; aggregate later
                        }
                    # Prefer the latest timestamp seen for same sha
                    if ts and (commits_map[sha]["ts"] or 0) < ts:
                        commits_map[sha]["ts"] = ts
                    commits_map[sha]["violations_raw"].extend(locs)

                # Aggregate per commit by (file,line) and per-spec
                for sha, obj in commits_map.items():
                    # group by location
                    by_loc: Dict[Tuple[str,int], dict] = {}
                    for v in obj["violations_raw"]:
                        key = (v["file"], v["line"])
                        if key not in by_loc:
                            by_loc[key] = {
                                "file": v["file"], "line": v["line"],
                                "count": 0,
                                "spec_counts": {}  # spec -> total
                            }
                        by_loc[key]["count"] += int(v["count"])
                        by_loc[key]["spec_counts"][v["spec"]] = by_loc[key]["spec_counts"].get(v["spec"], 0) + int(v["count"])

                    # build violations array with stable ids
                    violations = []
                    for (f, ln), rec in sorted(by_loc.items(), key=lambda t: (t[0][0], t[0][1])):
                        specs = ";".join(sorted(rec["spec_counts"].keys()))
                        breakdown = [{"spec": s, "count": int(c)} for s, c in sorted(rec["spec_counts"].items())]
                        violations.append({
                            "id": hashlib.sha1(f"{f}|{ln}".encode("utf-8")).hexdigest()[:12],
                            "file": f,
                            "line": int(ln),
                            "count": int(rec["count"]),
                            "specs": specs,
                            "breakdown": breakdown
                        })

                    proj["commits"].append({
                        "sha": sha,
                        "ts": obj["ts"],
                        "counts": {"locations": len(violations)},
                        "violations": violations
                    })

                # Totals
                total_commits += len(commits_map)
                total_locations += sum(c["counts"]["locations"] for c in proj["commits"])

        projects_out.append(proj)

    return {
        "projects": projects_out,
        "totals": {"commits": total_commits, "locations": total_locations}
    }

def build():
    payload = {
        "generated_at": int(time.time()),
        "datasets": {
            "monitoring": build_dataset(MON_PREFIX),
            "history":    build_dataset(HIS_PREFIX)
        }
    }
    (OUT / "data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Copy SPA
    copyfile(WEB / "index.html", OUT / "index.html")

if __name__ == "__main__":
    build()