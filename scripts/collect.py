import os, io, json, time, zipfile, pathlib, hashlib, re, datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd
from shutil import copyfile

# ---------- Paths & config ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT  = ROOT / "dist"
WEB  = ROOT / "web"
DATA = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = json.loads((ROOT / "config" / "settings.json").read_text())

# Parse repos.txt: format is "owner/repo;url;date" or variations
REPOS_WITH_METADATA = []
for l in (ROOT / "config" / "repos.txt").read_text().splitlines():
    l = l.strip()
    if not l or l.startswith("#"):
        continue
    parts = [p.strip() for p in l.split(";")]
    repo_name = parts[0] if len(parts) > 0 else ""
    url = parts[1] if len(parts) > 1 else None
    date = parts[2] if len(parts) > 2 else None
    REPOS_WITH_METADATA.append((repo_name, url, date))
REPOS = [repo for repo, _, _ in REPOS_WITH_METADATA]

def _ensure_prefix_list(value) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    raise TypeError(f"Expected string or list of strings for prefix configuration, got {type(value).__name__}")

MON_PREFIXES = _ensure_prefix_list(CONFIG["monitoring_prefix"])
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
    time.sleep(0.2)
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
        # fallback: any CSV
        for n in names:
            if n.lower().endswith(".csv"):
                with zf.open(n) as f: return n, f.read()
    return None

# ---------- Data parsing ----------
ENTRY_SEP = re.compile(r"\s*;\s*")

def _normalize_path(p: str) -> str:
    try:
        if p.startswith("/") or p.startswith("./") or p.startswith("../"):
            return os.path.normpath(p)
    except Exception:
        pass
    return p

def parse_vloc_cell(cell: str) -> List[dict]:
    """
    Parse strings like:
      Spec:/path/file.py:34;OtherSpec:/usr/lib/.../x.py:495
    Returns: list of dicts {spec, file, line}
    """
    out = []
    if cell is None:
        return out
    s = str(cell).strip()
    if not s:
        return out

    for raw in ENTRY_SEP.split(s.strip("; ")):
        if not raw:
            continue
        # Expect "<spec>:<path>:<line>" but <path> may contain colons
        if ":" not in raw:
            continue
        left2, line_s = raw.rsplit(":", 1)
        try:
            line = int(line_s.strip())
        except Exception:
            continue

        if ":" in left2:
            spec, file_part = left2.split(":", 1)
        else:
            spec, file_part = "", left2

        spec = (spec or "Unknown").strip()
        file_part = _normalize_path(file_part.strip())
        if not file_part:
            continue

        out.append({"spec": spec, "file": file_part, "line": line})
    return out

def to_epoch(ts_val) -> Optional[int]:
    """
    Parse many timestamp formats to epoch seconds (UTC).
    Supports:
      - epoch seconds (10) or milliseconds (13)
      - ISO-like strings with/without 'Z'
      - 'YYYYMMDD_HHMM' (e.g. 20250821_1229)
      - 'YYYYMMDD_HHMMSS' (e.g. 20250821_122901)
    """
    if ts_val is None:
        return None
    s = str(ts_val).strip()
    if not s or s.lower() == "nan":
        return None

    # 1) epoch seconds or milliseconds
    if s.isdigit() and len(s) in (10, 13):
        return int(int(s) / (1000 if len(s) == 13 else 1))

    # 2) 'YYYYMMDD_HHMMSS' or 'YYYYMMDD_HHMM'
    m6 = re.match(r"^(\d{8})_(\d{6})$", s)
    if m6:
        ymd, hms = m6.groups()
        y, mth, d = int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])
        hh, mm, ss = int(hms[0:2]), int(hms[2:4]), int(hms[4:6])
        try:
            dtm = dt.datetime(y, mth, d, hh, mm, ss, tzinfo=dt.timezone.utc)
            return int(dtm.timestamp())
        except Exception:
            return None
    m4 = re.match(r"^(\d{8})_(\d{4})$", s)
    if m4:
        ymd, hm = m4.groups()
        y, mth, d = int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8])
        hh, mm = int(hm[0:2]), int(hm[2:4])
        try:
            dtm = dt.datetime(y, mth, d, hh, mm, tzinfo=dt.timezone.utc)
            return int(dtm.timestamp())
        except Exception:
            return None

    # 3) ISO-ish
    s_norm = s.replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dtm = dt.datetime.strptime(s_norm, fmt)
            if dtm.tzinfo is None: dtm = dtm.replace(tzinfo=dt.timezone.utc)
            return int(dtm.timestamp())
        except Exception:
            pass

    # 4) pandas fallback
    try:
        d = pd.to_datetime([s], utc=True, errors="coerce")[0]
        if pd.notna(d): return int(d.to_pydatetime().timestamp())
    except Exception:
        pass
    return None

def build_dataset(prefixes: List[str], compute_after_first: bool = False) -> dict:
    """
    Reads artifacts with the first available prefix (tries each in order).
    Requires CSV columns (lowercased after read):
      timestamp, current_commit_sha, new_violations
    Uses ONLY 'new_violations' (blank => zero violations).
    Aggregates distinct specs per (file,line).
    """
    projects_out, total_commits, total_locations = [], 0, 0
    total_new_after_first = 0

    # Create mappings from repo name to metadata
    repo_url_map = {repo: url for repo, url, _ in REPOS_WITH_METADATA}
    repo_date_map = {repo: date for repo, _, date in REPOS_WITH_METADATA}
    
    # Iterate through repos in order, using line number (1-based) to determine prefix
    # Projects on lines 1-45 use "continuous-analysis-future-filtered-results-"
    # Projects on lines 46+ use "continuous-analysis-filtered-results-"
    for idx, full in enumerate(REPOS):
        owner, repo = full.split("/", 1)
        art = None
        
        # Line number is idx + 1 (1-based)
        line_number = idx + 1
        if line_number <= 45:
            # Projects on lines 1-45 use future prefix
            selected_prefixes = ["continuous-analysis-future-filtered-results-"]
        else:
            # Projects on lines 46+ use regular prefix
            selected_prefixes = ["continuous-analysis-filtered-results-"]
        
        for prefix in selected_prefixes:
            art = latest_with_prefix(owner, repo, prefix)
            if art:
                break

        proj = {
            "slug": full.replace("/","-").lower(),
            "full_name": full,
            "latest_artifact_name": art.get("name") if art else None,
            "url": repo_url_map.get(full),
            "shadowed_repo": full,
            "date_shadowing_started": repo_date_map.get(full),
            "commits": []
        }

        if art:
            zip_bytes = gh_get_bin(art["archive_download_url"])
            found = find_csv(zip_bytes, CSV_CANDS)
            if not found:
                if FAIL_IF_MISSING_CSV:
                    raise FileNotFoundError(f"No CSV found in artifact {art['name']}")
            else:
                _, csv_bytes = found
                # Force string dtype and fill NaN so new_violations is never lost
                df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str).fillna("")
                df.columns = [c.strip().lower() for c in df.columns]

                required = {"timestamp", "current_commit_sha", "new_violations"}
                if not required.issubset(df.columns):
                    raise ValueError(f"CSV missing required columns for {full} ({art}); found {df.columns.tolist()}")

                # Build commits map strictly from new_violations
                commits_map: Dict[str, dict] = {}
                for _, r in df.iterrows():
                    sha = str(r.get("current_commit_sha", "")).strip()
                    if not sha:
                        continue
                    ts  = to_epoch(r.get("timestamp", ""))
                    new_v = str(r.get("new_violations", "") or "").strip()
                    coverage = r.get("coverage", "")
                    commit_msg = r.get("current_commit_message", "")
                    commit_ts = r.get("current_commit_timestamp", "")
                    locs = parse_vloc_cell(new_v) if new_v else []
                    
                    # Read violation counts from CSV
                    num_current = r.get("num_current_violations", "")
                    num_new = r.get("num_new_violations", "")
                    num_old = r.get("num_old_violations", "")
                    
                    def parse_int_or_zero(val):
                        if val is None or str(val).strip() == "" or str(val).strip().lower() == "nan":
                            return 0
                        try:
                            return int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            return 0
                    
                    num_current_int = parse_int_or_zero(num_current)
                    num_new_int = parse_int_or_zero(num_new)
                    num_old_int = parse_int_or_zero(num_old)

                    if sha not in commits_map:
                        commits_map[sha] = {
                            "sha": sha, 
                            "ts": ts, 
                            "violations_raw": [], 
                            "coverage": None,
                            "current_commit_message": None,
                            "current_commit_timestamp": None,
                            "num_current_violations": 0,
                            "num_new_violations": 0,
                            "num_old_violations": 0
                        }
                    if ts and (commits_map[sha]["ts"] or 0) < ts:
                        commits_map[sha]["ts"] = ts
                    if locs:
                        commits_map[sha]["violations_raw"].extend(locs)
                    
                    # Store violation counts (use latest values if multiple rows for same SHA)
                    commits_map[sha]["num_current_violations"] = num_current_int
                    commits_map[sha]["num_new_violations"] = num_new_int
                    commits_map[sha]["num_old_violations"] = num_old_int
                    
                    # Store coverage if present
                    if coverage and str(coverage).strip() and str(coverage).strip().lower() != "nan":
                        try:
                            # Try to parse as float first, then as is
                            coverage_val = float(coverage)
                            commits_map[sha]["coverage"] = coverage_val
                        except (ValueError, TypeError):
                            commits_map[sha]["coverage"] = str(coverage).strip()
                    
                    # Store commit message if present
                    if commit_msg and str(commit_msg).strip() and str(commit_msg).strip().lower() != "nan":
                        commits_map[sha]["current_commit_message"] = str(commit_msg).strip()
                    
                    # Store commit timestamp if present
                    if commit_ts and str(commit_ts).strip() and str(commit_ts).strip().lower() != "nan":
                        commit_ts_epoch = to_epoch(commit_ts)
                        if commit_ts_epoch:
                            commits_map[sha]["current_commit_timestamp"] = commit_ts_epoch

                # Aggregate to output format
                EXCLUDED_PATH = "specs-new/NLTK_NonterminalSymbolMutability.py"

                for sha, obj in commits_map.items():
                    by_loc: Dict[Tuple[str,int], dict] = {}
                    for v in obj["violations_raw"]:
                        # Skip excluded file paths
                        if EXCLUDED_PATH in v["file"]:
                            continue
                        key = (v["file"], v["line"])
                        rec = by_loc.setdefault(key, {"file": v["file"], "line": v["line"], "specs": set()})
                        rec["specs"].add(v["spec"])

                    violations = []
                    for (f, ln), rec in sorted(by_loc.items(), key=lambda t: (t[0][0], t[0][1])):
                        spec_list = sorted(rec["specs"])
                        violations.append({
                            "id": hashlib.sha1(f"{f}|{ln}".encode("utf-8")).hexdigest()[:12],
                            "file": f,
                            "line": int(ln),
                            "count": len(spec_list), 
                            "specs": ";".join(spec_list),
                            "breakdown": [{"spec": s, "count": 1} for s in spec_list]
                        })

                    # Use num_current_violations from CSV for counts.locations (total current violations)
                    # The violations list only contains new violations from new_violations column, not all current violations
                    commit_data = {
                        "sha": sha,
                        "current_commit_sha": sha,
                        "ts": obj["ts"],
                        "counts": {"locations": obj.get("num_current_violations", 0)},
                        "violations": violations,
                        "num_current_violations": obj.get("num_current_violations", 0),
                        "num_new_violations": obj.get("num_new_violations", 0),
                        "num_old_violations": obj.get("num_old_violations", 0)
                    }
                    
                    # Add coverage if available
                    if obj.get("coverage") is not None:
                        commit_data["coverage"] = obj["coverage"]
                    
                    # Add commit message if available
                    if obj.get("current_commit_message") is not None:
                        commit_data["current_commit_message"] = obj["current_commit_message"]
                    
                    # Add commit timestamp if available
                    if obj.get("current_commit_timestamp") is not None:
                        commit_data["current_commit_timestamp"] = obj["current_commit_timestamp"]
                    
                    proj["commits"].append(commit_data)

                total_commits += len(commits_map)
                total_locations += sum(c["counts"]["locations"] for c in proj["commits"])

                # History-specific aggregate: unique locations that appear after the first commit
                if compute_after_first and proj["commits"]:
                    # The "first" commit is the first row in the CSV (already in chronological order)
                    first_commit = proj["commits"][0]
                    base_ids = {v.get("id") for v in first_commit.get("violations", [])}
                    
                    # Collect ALL violation IDs from ALL commits
                    all_violation_ids: set = set()
                    for c in proj["commits"]:
                        for v in c.get("violations", []):
                            vid = v.get("id")
                            if vid:
                                all_violation_ids.add(vid)
                    
                    # New violations = violations that exist in the overall history but NOT in the first commit
                    new_after_first = len(all_violation_ids - base_ids)
                    # Removed violations = violations that exist in the first commit but NOT in the overall history
                    removed_after_first = len(base_ids - all_violation_ids)
                    
                    # Debug logging
                    print(f"DEBUG {proj['full_name']}:")
                    print(f"  Total commits: {len(proj['commits'])}")
                    print(f"  First commit: {first_commit['sha'][:8]} (ts: {first_commit['ts']}, violations: {len(first_commit.get('violations', []))})")
                    print(f"  Base IDs count: {len(base_ids)}")
                    print(f"  All violation IDs count: {len(all_violation_ids)}")
                    print(f"  New after first: {new_after_first}")
                    print(f"  Removed after first: {removed_after_first}")
                    
                    # Attach per-project counts
                    proj.setdefault("counts", {})["new_locations_after_first"] = new_after_first
                    proj.setdefault("counts", {})["removed_locations_after_first"] = removed_after_first
                    total_new_after_first += new_after_first

        projects_out.append(proj)

    totals = {"commits": total_commits, "locations": total_locations}
    if compute_after_first:
        totals["new_locations_after_first"] = total_new_after_first
    return {"projects": projects_out, "totals": totals}

def build_dataset_from_local(compute_after_first: bool = False) -> dict:
    """
    Reads CSV files from local data directory.
    CSV filename should match the repo name (part after '/' in full repo name).
    Requires CSV columns (lowercased after read):
      timestamp, current_commit_sha, new_violations
    Uses ONLY 'new_violations' (blank => zero violations).
    Aggregates distinct specs per (file,line).
    """
    projects_out, total_commits, total_locations = [], 0, 0
    total_new_after_first = 0

    # Track repo names that are explicitly listed in repos.txt
    configured_repo_names = set()

    def process_one_project(full: str, csv_path: pathlib.Path, url: Optional[str] = None, date: Optional[str] = None):
        """
        Process a single CSV file into the unified project/commit structure.
        Mutates projects_out / total_* counters in the outer scope.
        """
        nonlocal total_commits, total_locations, total_new_after_first

        # full may be either "owner/repo" (from repos.txt) or just a repo/file name
        proj = {
            "slug": full.replace("/","-").lower(),
            "full_name": full,
            "latest_artifact_name": csv_path.name if csv_path.exists() else None,  # Use CSV filename for history runs
            "url": url,
            "shadowed_repo": full,
            "date_shadowing_started": date,
            "commits": []
        }

        if csv_path.exists():
            try:
                # Force string dtype and fill NaN so new_violations is never lost
                df = pd.read_csv(csv_path, dtype=str).fillna("")
                df.columns = [c.strip().lower() for c in df.columns]

                required = {"timestamp", "current_commit_sha", "new_violations"}
                if not required.issubset(df.columns):
                    print(f"WARNING: CSV missing required columns for {full} ({csv_path}); found {df.columns.tolist()}")
                    projects_out.append(proj)
                    return

                # Build commits map strictly from new_violations
                commits_map: Dict[str, dict] = {}
                for _, r in df.iterrows():
                    sha = str(r.get("current_commit_sha", "")).strip()
                    if not sha:
                        continue
                    ts  = to_epoch(r.get("timestamp", ""))
                    new_v = str(r.get("new_violations", "") or "").strip()
                    coverage = r.get("coverage", "")
                    commit_msg = r.get("current_commit_message", "")
                    commit_ts = r.get("current_commit_timestamp", "")
                    num_python_files = r.get("num_python_file_changed", "")
                    github_url_val = r.get("github_url", "")
                    locs = parse_vloc_cell(new_v) if new_v else []

                    # Read violation counts from CSV
                    num_current = r.get("num_current_violations", "")
                    num_new = r.get("num_new_violations", "")
                    num_old = r.get("num_old_violations", "")

                    def parse_int_or_zero(val):
                        if val is None or str(val).strip() == "" or str(val).strip().lower() == "nan":
                            return 0
                        try:
                            return int(float(str(val).strip()))
                        except (ValueError, TypeError):
                            return 0

                    num_current_int = parse_int_or_zero(num_current)
                    num_new_int = parse_int_or_zero(num_new)
                    num_old_int = parse_int_or_zero(num_old)

                    if sha not in commits_map:
                        commits_map[sha] = {
                            "sha": sha,
                            "ts": ts,
                            "violations_raw": [],
                            "coverage": None,
                            "current_commit_message": None,
                            "current_commit_timestamp": None,
                            "num_current_violations": 0,
                            "num_new_violations": 0,
                            "num_old_violations": 0,
                            "num_python_file_changed": None,
                            "github_url": None
                        }
                    if ts and (commits_map[sha]["ts"] or 0) < ts:
                        commits_map[sha]["ts"] = ts
                    if locs:
                        commits_map[sha]["violations_raw"].extend(locs)

                    # Store violation counts (use latest values if multiple rows for same SHA)
                    commits_map[sha]["num_current_violations"] = num_current_int
                    commits_map[sha]["num_new_violations"] = num_new_int
                    commits_map[sha]["num_old_violations"] = num_old_int

                    # Store coverage if present
                    if coverage and str(coverage).strip() and str(coverage).strip().lower() != "nan":
                        try:
                            # Try to parse as float first, then as is
                            coverage_val = float(coverage)
                            commits_map[sha]["coverage"] = coverage_val
                        except (ValueError, TypeError):
                            commits_map[sha]["coverage"] = str(coverage).strip()

                    # Store commit message if present
                    if commit_msg and str(commit_msg).strip() and str(commit_msg).strip().lower() != "nan":
                        commits_map[sha]["current_commit_message"] = str(commit_msg).strip()

                    # Store commit timestamp if present
                    if commit_ts and str(commit_ts).strip() and str(commit_ts).strip().lower() != "nan":
                        commit_ts_epoch = to_epoch(commit_ts)
                        if commit_ts_epoch:
                            commits_map[sha]["current_commit_timestamp"] = commit_ts_epoch
                    
                    # Store num_python_file_changed if present
                    if num_python_files and str(num_python_files).strip() and str(num_python_files).strip().lower() != "nan":
                        try:
                            commits_map[sha]["num_python_file_changed"] = int(float(str(num_python_files).strip()))
                        except (ValueError, TypeError):
                            pass
                    
                    # Store github_url if present
                    if github_url_val and str(github_url_val).strip() and str(github_url_val).strip().lower() != "nan":
                        commits_map[sha]["github_url"] = str(github_url_val).strip()

                for sha, obj in commits_map.items():
                    by_loc: Dict[Tuple[str,int], dict] = {}
                    for v in obj["violations_raw"]:
                        key = (v["file"], v["line"])
                        rec = by_loc.setdefault(key, {"file": v["file"], "line": v["line"], "specs": set()})
                        rec["specs"].add(v["spec"])

                    violations = []
                    for (f, ln), rec in sorted(by_loc.items(), key=lambda t: (t[0][0], t[0][1])):
                        spec_list = sorted(rec["specs"])
                        violations.append({
                            "id": hashlib.sha1(f"{f}|{ln}".encode("utf-8")).hexdigest()[:12],
                            "file": f,
                            "line": int(ln),
                            "count": len(spec_list),
                            "specs": ";".join(spec_list),
                            "breakdown": [{"spec": s, "count": 1} for s in spec_list]
                        })

                    # Use num_current_violations from CSV for counts.locations (total current violations)
                    commit_data = {
                        "sha": sha,
                        "current_commit_sha": sha,
                        "ts": obj["ts"],
                        "counts": {"locations": obj.get("num_current_violations", 0)},
                        "violations": violations,
                        "num_current_violations": obj.get("num_current_violations", 0),
                        "num_new_violations": obj.get("num_new_violations", 0),
                        "num_old_violations": obj.get("num_old_violations", 0)
                    }

                    # Add coverage if available
                    if obj.get("coverage") is not None:
                        commit_data["coverage"] = obj["coverage"]

                    # Add commit message if available
                    if obj.get("current_commit_message") is not None:
                        commit_data["current_commit_message"] = obj["current_commit_message"]

                    # Add commit timestamp if available
                    if obj.get("current_commit_timestamp") is not None:
                        commit_data["current_commit_timestamp"] = obj["current_commit_timestamp"]
                    
                    # Add num_python_file_changed if available (for history runs)
                    if obj.get("num_python_file_changed") is not None:
                        commit_data["num_python_file_changed"] = obj["num_python_file_changed"]
                    
                    # Add github_url if available (for history runs)
                    if obj.get("github_url") is not None:
                        commit_data["github_url"] = obj["github_url"]
                    
                    proj["commits"].append(commit_data)

                total_commits += len(commits_map)
                total_locations += sum(c["counts"]["locations"] for c in proj["commits"])

                # History-specific aggregate: unique locations that appear after the first commit
                if compute_after_first and proj["commits"]:
                    # The "first" commit is the first row in the CSV (already in chronological order)
                    first_commit = proj["commits"][0]
                    base_ids = {v.get("id") for v in first_commit.get("violations", [])}

                    # Collect ALL violation IDs from ALL commits
                    all_violation_ids: set = set()
                    for c in proj["commits"]:
                        for v in c.get("violations", []):
                            vid = v.get("id")
                            if vid:
                                all_violation_ids.add(vid)

                    # New violations = violations that exist in the overall history but NOT in the first commit
                    new_after_first = len(all_violation_ids - base_ids)
                    # Removed violations = violations that exist in the first commit but NOT in the overall history
                    removed_after_first = len(base_ids - all_violation_ids)
                    
                    # Debug logging
                    print(f"DEBUG {proj['full_name']}:")
                    print(f"  Total commits: {len(proj['commits'])}")
                    print(f"  First commit: {first_commit['sha'][:8]} (ts: {first_commit['ts']}, violations: {len(first_commit.get('violations', []))})")
                    print(f"  Base IDs count: {len(base_ids)}")
                    print(f"  All violation IDs count: {len(all_violation_ids)}")
                    print(f"  New after first: {new_after_first}")
                    print(f"  Removed after first: {removed_after_first}")
                    
                    # Attach per-project counts
                    proj.setdefault("counts", {})["new_locations_after_first"] = new_after_first
                    proj.setdefault("counts", {})["removed_locations_after_first"] = removed_after_first
                    total_new_after_first += new_after_first
            except Exception as e:
                print(f"ERROR: Failed to process {csv_path} for {full}: {e}")
            # Only append the project if we actually found a CSV file.
            projects_out.append(proj)
        else:
            # If there's no CSV for this project, just warn and skip it entirely
            # so it does not appear in the history tab.
            print(f"WARNING: CSV file not found for {full}: {csv_path}")

    # Create mappings from repo name to metadata
    repo_url_map = {repo: url for repo, url, _ in REPOS_WITH_METADATA}
    repo_date_map = {repo: date for repo, _, date in REPOS_WITH_METADATA}
    
    # First, process all projects that are explicitly configured in repos.txt
    for full in REPOS:
        owner, repo = full.split("/", 1)
        repo = repo.strip()  # Remove any trailing spaces
        configured_repo_names.add(repo)

        # CSV filename is the repo name
        csv_path = DATA / f"{repo}.csv"
        process_one_project(full, csv_path, repo_url_map.get(full), repo_date_map.get(full))

    # Then, add any remaining CSV files in the data directory as standalone projects
    # (these may not appear in repos.txt). The file stem becomes the repo name.
    for csv_path in DATA.glob("*.csv"):
        repo_name = csv_path.stem
        if repo_name in configured_repo_names:
            continue  # already processed via repos.txt mapping

        # Use the filename as the "repo" identifier for history runs not in repos.txt
        full = repo_name
        process_one_project(full, csv_path, None, None)

    totals = {"commits": total_commits, "locations": total_locations}
    if compute_after_first:
        totals["new_locations_after_first"] = total_new_after_first
    return {"projects": projects_out, "totals": totals}

def build():
    payload = {
        "generated_at": int(time.time()),
        "datasets": {
            "monitoring": build_dataset(MON_PREFIXES),
            "history":    build_dataset_from_local(compute_after_first=True)
        }
    }
    (OUT / "data.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    copyfile(WEB / "index.html", OUT / "index.html")

if __name__ == "__main__":
    build()
