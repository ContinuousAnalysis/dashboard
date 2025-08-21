import os, io, json, time, zipfile, pathlib, hashlib, re, datetime as dt
from typing import Dict, Any, List, Optional
import requests
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "settings.json").read_text())
REPOS = [l.strip() for l in (ROOT / "config" / "repos.txt").read_text().splitlines()
         if l.strip() and not l.strip().startswith("#")]

OUT = ROOT / "dist"
PROJ_DIR = OUT / "projects"
VIOL_DIR = OUT / "violations"
ASSETS_DIR = OUT / "assets"
for d in (OUT, PROJ_DIR, VIOL_DIR, ASSETS_DIR):
    d.mkdir(parents=True, exist_ok=True)

env = Environment(
    loader=FileSystemLoader(str(ROOT / "site_template")),
    autoescape=select_autoescape(["html", "xml"])
)

ARTIFACT_PREFIX = CONFIG["artifact_name_prefix"]
CSV_CANDIDATES = CONFIG["csv_name_candidates"]
FAIL_IF_MISSING_CSV = CONFIG.get("fail_if_missing_csv", True)

API = "https://api.github.com"
TOKEN = os.getenv("ORG_WIDE_TOKEN") or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
HDRS = {"Accept": "application/vnd.github+json"}
if TOKEN:
    HDRS["Authorization"] = f"Bearer {TOKEN}"

_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")

def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9\-]+","-", s.lower())

def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def parse_name_timestamp(name: str) -> Optional[dt.datetime]:
    m = _TS_RE.search(name or "")
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)

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

def list_all_artifacts(owner: str, repo: str):
    page = 1
    per_page = 100
    while True:
        url = f"{API}/repos/{owner}/{repo}/actions/artifacts?per_page={per_page}&page={page}"
        data = gh_get(url)
        items = data.get("artifacts", []) or []
        if not items:
            break
        for a in items:
            yield a
        if len(items) < per_page:
            break
        page += 1

def looks_like_history(a: dict, repo: str) -> bool:
    name = a.get("name","")
    if not name.startswith(ARTIFACT_PREFIX): return False
    if a.get("expired"): return False
    # defend against cross-repo collisions; many names encode "-<repo>-"
    return f"-{repo}-" in name

def latest_history_artifact(owner: str, repo: str) -> Optional[dict]:
    cands: List[tuple[dict, dt.datetime]] = []
    for a in list_all_artifacts(owner, repo):
        if not looks_like_history(a, repo):
            continue
        ts = parse_name_timestamp(a.get("name",""))
        created = a.get("created_at")
        created_dt = None
        if created:
            try: created_dt = dt.datetime.fromisoformat(created.replace("Z","+00:00"))
            except Exception: created_dt = None
        score = ts or created_dt or dt.datetime.fromtimestamp(0, tz=dt.timezone.utc)
        cands.append((a, score))
    if not cands:
        return None
    cands.sort(key=lambda t: t[1], reverse=True)
    return cands[0][0]

def find_csv_in_zip(zip_bytes: bytes) -> tuple[str, bytes] | None:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for want in CSV_CANDIDATES:
            if want in names:
                with zf.open(want) as f:
                    return want, f.read()
        # fallback: any .csv
        for n in names:
            if n.lower().endswith(".csv"):
                with zf.open(n) as f:
                    return n, f.read()
    return None

# ---------- parsing violations_by_location ----------
ENTRY_SPLIT = re.compile(r"\s*;\s*")
# "Spec:/path/to/file.py:123" or "Spec:/path/file.py:123=COUNT"
ENTRY_RE = re.compile(r"^(?P<spec>[^:]+):(?P<file>.*):(?P<line>\d+)(?:=(?P<count>\d+))?$")

def parse_vloc_cell(project: str, commit: str, algo: str, cell: str) -> List[dict]:
    algo_norm = ("pymop" if "pymop" in algo.lower() else ("dylin" if "dylin" in algo.lower() else algo.lower()))
    out = []
    if not cell: return out
    for entry in ENTRY_SPLIT.split(cell.strip("; ")):
        if not entry: continue
        m = ENTRY_RE.match(entry)
        if not m:
            # keep malformed entries if needed
            continue
        g = m.groupdict()
        out.append({
            "project": project,
            "commit": commit,
            "tool": algo_norm,
            "spec": g["spec"],
            "file": g["file"],
            "line": int(g["line"]),
            "count": int(g.get("count") or 1),
        })
    return out

def stable_loc_id(file: str, line: int) -> str:
    h = hashlib.sha1(f"{file}|{line}".encode("utf-8")).hexdigest()
    return h[:12]

# ---------- site building ----------
def render_template(name: str, **ctx) -> str:
    tpl = env.get_template(name)
    return tpl.render(generated=now_str(), **ctx)

def build():
    all_projects_meta = []
    total_locations = 0
    total_commits = 0

    for full in REPOS:
        owner, repo = full.split("/", 1)
        slug = slugify(full)
        art = latest_history_artifact(owner, repo)
        proj_meta = {
            "full_name": full,
            "slug": slug,
            "latest_artifact_name": art.get("name") if art else None,
            "latest_commit": None,
            "commit_count": 0,
            "location_count": 0,
            "csv_source": None,
        }

        parsed_rows: List[dict] = []
        spec_rows: List[dict] = []

        if art:
            zip_bytes = gh_get_bin(art["archive_download_url"])
            found = find_csv_in_zip(zip_bytes)
            if not found:
                if FAIL_IF_MISSING_CSV:
                    raise FileNotFoundError("No CSV found in artifact")
            else:
                csv_name, csv_bytes = found
                # save original CSV for download
                csv_path = ASSETS_DIR / f"{slug}-{csv_name.split('/')[-1]}"
                csv_path.write_bytes(csv_bytes)
                proj_meta["csv_source"] = f"../assets/{csv_path.name}"

                df = pd.read_csv(io.BytesIO(csv_bytes))
                df.columns = [c.strip().lower() for c in df.columns]

                # expected columns
                project_col = "project"
                commit_col  = "commit_sha"
                algo_col    = "algorithm"
                vloc_col    = "violations_by_location"

                if not set([project_col, commit_col, algo_col, vloc_col]).issubset(set(df.columns)):
                    raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")

                # keep pymop/dylin rows only
                df_use = df[df[algo_col].str.lower().isin(["pymop","dylin"])].copy()
                # if algorithms named differently but contain substrings, normalize
                if df_use.empty:
                    df_use = df[df[algo_col].str.lower().str.contains("pymop|dylin")].copy()

                # parse violations_by_location
                for _, r in df_use.iterrows():
                    cells = parse_vloc_cell(str(r[project_col]), str(r[commit_col]), str(r[algo_col]), str(r[vloc_col]))
                    parsed_rows.extend(cells)

                if parsed_rows:
                    agg = (
                        pd.DataFrame(parsed_rows)
                        .groupby(["project","commit","tool","file","line","spec"], dropna=False)["count"]
                        .sum()
                        .reset_index()
                    )

                    # tool totals per location
                    tool_totals = (
                        agg.groupby(["project","commit","tool","file","line"], dropna=False)["count"]
                        .sum()
                        .rename("tool_total")
                        .reset_index()
                    )
                    tool_pivot = tool_totals.pivot_table(
                        index=["project","commit","file","line"],
                        columns="tool", values="tool_total", fill_value=0
                    ).reset_index()

                    for t in ["pymop","dylin"]:
                        if t not in tool_pivot.columns:
                            tool_pivot[t] = 0

                    tool_pivot["flagged_by"] = tool_pivot.apply(
                        lambda r: "both" if r["pymop"]>0 and r["dylin"]>0 else ("pymop" if r["pymop"]>0 else ("dylin" if r["dylin"]>0 else "none")), axis=1
                    )
                    tool_pivot["_id"] = tool_pivot.apply(lambda r: stable_loc_id(r["file"], int(r["line"])), axis=1)

                    # spec breakdown per location
                    spec_breakdown = (
                        agg.pivot_table(index=["project","commit","file","line","spec"], columns="tool", values="count", fill_value=0)
                        .reset_index()
                    )
                    for t in ["pymop","dylin"]:
                        if t not in spec_breakdown.columns:
                            spec_breakdown[t] = 0

                    # derive commit summary counts
                    commit_summary = (
                        tool_pivot.groupby(["project","commit"], dropna=False)
                        .agg(
                            locations=("file","count"),
                            flagged_by_both=("flagged_by", lambda s: int((s == "both").sum())),
                            flagged_by_pymop_only=("flagged_by", lambda s: int((s == "pymop").sum())),
                            flagged_by_dylin_only=("flagged_by", lambda s: int((s == "dylin").sum()))
                        )
                        .reset_index()
                    )

                    # write per-project site
                    project_dir = PROJ_DIR / slug
                    (project_dir / "commits").mkdir(parents=True, exist_ok=True)
                    # overall per project stats
                    proj_meta["commit_count"] = int(commit_summary["commit"].nunique()) if not commit_summary.empty else 0
                    proj_meta["location_count"] = int(tool_pivot.shape[0])
                    if not commit_summary.empty:
                        proj_meta["latest_commit"] = str(commit_summary["commit"].iloc[-1])

                    # render project page
                    html = render_template("project.html.j2",
                        project=proj_meta,
                        commits=[
                            {
                                "commit": row["commit"],
                                "locations": int(row["locations"]),
                                "flagged_by_both": int(row["flagged_by_both"]),
                                "flagged_by_pymop_only": int(row["flagged_by_pymop_only"]),
                                "flagged_by_dylin_only": int(row["flagged_by_dylin_only"]),
                            } for _, row in commit_summary.sort_values("commit").iterrows()
                        ]
                    )
                    (project_dir / "index.html").write_text(html, encoding="utf-8")

                    # render per-commit pages + violation pages
                    for commit in commit_summary["commit"].tolist():
                        rows_loc = tool_pivot[tool_pivot["commit"] == commit].copy()
                        rows_loc = rows_loc.sort_values(["file","line"])
                        # collect specs per location for quick view
                        merged = rows_loc.merge(
                            spec_breakdown, on=["project","commit","file","line"], how="left", suffixes=("","_spec")
                        )
                        # per row: join specs where count>0 in either tool
                        specs_series = (
                            merged.groupby(["file","line","_id"])
                            .apply(lambda g: ";".join(sorted([str(s) for s in g["spec"].unique() if str(s) not in ["nan","None",""]])))
                            .rename("specs")
                            .reset_index()
                        )
                        rows_with_specs = rows_loc.merge(specs_series, on=["file","line","_id"], how="left")
                        # write commit page
                        commit_dir = project_dir / "commits"
                        counts = {
                            "both": int((rows_with_specs["flagged_by"] == "both").sum()),
                            "pymop_only": int((rows_with_specs["flagged_by"] == "pymop").sum()),
                            "dylin_only": int((rows_with_specs["flagged_by"] == "dylin").sum()),
                        }
                        commit_html = render_template("commit.html.j2",
                            project=proj_meta,
                            commit=commit,
                            counts=counts,
                            rows=[
                                {
                                    "_id": r["_id"],
                                    "file": r["file"],
                                    "line": int(r["line"]),
                                    "flagged_by": r["flagged_by"],
                                    "specs": rows_with_specs.loc[rows_with_specs["_id"]==r["_id"], "specs"].iloc[0] or ""
                                } for _, r in rows_with_specs.iterrows()
                            ]
                        )
                        (commit_dir / f"{commit}.html").write_text(commit_html, encoding="utf-8")

                        # violation detail pages with per-location spec breakdown
                        vio_base = VIOL_DIR / slug / commit
                        vio_base.mkdir(parents=True, exist_ok=True)
                        for _, r in rows_with_specs.iterrows():
                            loc_specs = spec_breakdown[
                                (spec_breakdown["project"]==proj_meta["full_name"]) &
                                (spec_breakdown["commit"]==commit) &
                                (spec_breakdown["file"]==r["file"]) &
                                (spec_breakdown["line"]==r["line"])
                            ][["spec","pymop","dylin"]]
                            vhtml = render_template("violation.html.j2",
                                project=proj_meta,
                                commit=commit,
                                violation={
                                    "_id": r["_id"],
                                    "file": r["file"],
                                    "line": int(r["line"]),
                                    "flagged_by": r["flagged_by"],
                                    "pymop": int(r.get("pymop",0)),
                                    "dylin": int(r.get("dylin",0)),
                                },
                                breakdown=[
                                    {"spec": str(x["spec"]), "pymop": int(x["pymop"]), "dylin": int(x["dylin"])}
                                    for _, x in loc_specs.iterrows()
                                ]
                            )
                            (vio_base / f"{r['_id']}.html").write_text(vhtml, encoding="utf-8")

                    # keep project-level meta for index
                    all_projects_meta.append(proj_meta)
                    total_locations += int(tool_pivot.shape[0])
                    total_commits += int(commit_summary["commit"].nunique())
                else:
                    all_projects_meta.append(proj_meta)  # no rows parsed
        else:
            all_projects_meta.append(proj_meta)  # no artifact

    # index page
    idx_html = render_template("index.html.j2",
        projects=all_projects_meta,
        total_commits=total_commits,
        total_locations=total_locations
    )
    (OUT / "index.html").write_text(idx_html, encoding="utf-8")

    # Also expose a JSON snapshot
    (OUT / "data.json").write_text(json.dumps({
        "generated_at": int(time.time()),
        "projects": all_projects_meta
    }, indent=2), encoding="utf-8")

if __name__ == "__main__":
    build()