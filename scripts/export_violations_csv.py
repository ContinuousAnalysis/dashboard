#!/usr/bin/env python3
"""
Export unique violation counts per commit per project from history runs to CSV.
Recalculates unique violations from current_violations column using the same method as the dashboard.
"""
import pathlib
import pandas as pd
from collect import parse_vloc_cell_with_occurrence, to_epoch, REPOS, DATA

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
EXCLUDED_PATH = "specs-new/NLTK_NonterminalSymbolMutability.py"

def count_unique_violations(locs_list):
    """
    Count unique violations by (file, line) - not by spec.
    Same logic as in collect.py
    """
    unique_locs = set()
    for v in locs_list:
        if EXCLUDED_PATH not in v.get("file", ""):
            unique_locs.add((v["file"], v["line"]))
    return len(unique_locs)

def export_violations_to_csv():
    """
    Extract unique violation counts for each commit in each project from history runs
    and export to a CSV file.
    
    Recalculates unique violations from current_violations column using the same
    parsing and counting logic as the dashboard.
    
    Follows the same logic as build_dataset_from_local: processes repos.txt projects first,
    then any remaining CSV files in the data directory.
    """
    # Collect all rows
    rows = []
    
    # Track repo names that are explicitly listed in repos.txt
    configured_repo_names = set()
    
    def process_one_project(project_name: str, csv_path: pathlib.Path):
        """Process a single CSV file and add rows to the rows list."""
        if not csv_path.exists():
            print(f"WARNING: CSV file not found for {project_name}: {csv_path}")
            return
        
        try:
            # Read CSV with string dtype to preserve all data
            df = pd.read_csv(csv_path, dtype=str).fillna("")
            df.columns = [c.strip().lower() for c in df.columns]
            
            required = {"timestamp", "current_commit_sha", "current_violations"}
            if not required.issubset(df.columns):
                print(f"WARNING: CSV missing required columns for {project_name} ({csv_path}); found {df.columns.tolist()}")
                return
            
            # Process each row (commit)
            for _, r in df.iterrows():
                sha = str(r.get("current_commit_sha", "")).strip()
                if not sha:
                    continue
                
                # Parse current_violations and count unique violations
                current_v = str(r.get("current_violations", "") or "").strip()
                current_locs = parse_vloc_cell_with_occurrence(current_v) if current_v else []
                num_unique_violations = count_unique_violations(current_locs)
                
                # Get timestamp
                ts = to_epoch(r.get("timestamp", ""))
                
                rows.append({
                    "project": project_name,
                    "commit_sha": sha,
                    "timestamp": ts if ts else "",
                    "num_unique_violations": num_unique_violations,
                })
                
        except Exception as e:
            print(f"ERROR: Failed to process {csv_path} for {project_name}: {e}")
    
    # First, process all projects that are explicitly configured in repos.txt
    for full in REPOS:
        owner, repo = full.split("/", 1)
        repo = repo.strip()  # Remove any trailing spaces
        configured_repo_names.add(repo)
        
        # CSV filename is the repo name
        csv_path = DATA / f"{repo}.csv"
        process_one_project(full, csv_path)
    
    # Then, add any remaining CSV files in the data directory as standalone projects
    # (these may not appear in repos.txt). The file stem becomes the repo name.
    for csv_path in DATA.glob("*.csv"):
        repo_name = csv_path.stem
        if repo_name in configured_repo_names:
            continue  # already processed via repos.txt mapping
        
        # Use the filename as the "repo" identifier for history runs not in repos.txt
        project_name = repo_name
        process_one_project(project_name, csv_path)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    
    # Sort by project name and timestamp
    df = df.sort_values(["project", "timestamp"])
    
    # Write to CSV
    output_path = OUT / "violations_history.csv"
    df.to_csv(output_path, index=False)
    print(f"Exported {len(rows)} commit records from {len(set(df['project']))} projects to {output_path}")
    print(f"Columns: {', '.join(df.columns.tolist())}")

if __name__ == "__main__":
    export_violations_to_csv()

