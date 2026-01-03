#!/usr/bin/env python3
"""
Workflow script to download latest artifacts for future runs and create a GitHub release.

Usage:
    python scripts/create_daily_release.py

Downloads the latest artifact for each future run project and creates a release.
"""

import os
import sys
import time
import pathlib
import datetime as dt
import importlib.util
from typing import Dict, List, Tuple
import requests

# Add parent directory to path to import from collect.py
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import shared utilities from collect.py
spec = importlib.util.spec_from_file_location("collect", ROOT / "scripts" / "collect.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)

API = collect.API
TOKEN = collect.TOKEN
HDRS = collect.HDRS
gh_get = collect.gh_get
gh_get_bin = collect.gh_get_bin
parse_name_ts = collect.parse_name_ts
list_artifacts = collect.list_artifacts
looks_with_prefix = collect.looks_with_prefix
latest_with_prefix = collect.latest_with_prefix
REPOS_WITH_METADATA = collect.REPOS_WITH_METADATA
REPOS = collect.REPOS

# ---------- Configuration ----------
FUTURE_PREFIX = "continuous-analysis-future-filtered-results-"
REGULAR_PREFIX = "continuous-analysis-filtered-results-"

# Get current repo info (assumes running from the dashboard repo)
def get_current_repo() -> Tuple[str, str]:
    """Get owner/repo for the current repository."""
    # Try to get from environment (GitHub Actions)
    repo_env = os.getenv("GITHUB_REPOSITORY")
    if repo_env:
        owner, repo = repo_env.split("/", 1)
        return owner, repo
    
    # Try to get from git config
    import subprocess
    try:
        remote_url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        # Handle both https://github.com/owner/repo.git and git@github.com:owner/repo.git
        if "github.com" in remote_url:
            parts = remote_url.replace(".git", "").split("github.com/")[-1].split("/")
            if len(parts) >= 2:
                return parts[0], parts[1]
    except:
        pass
    
    # Default fallback (update if needed)
    return "Zhuohang-Shen", "continuous_analysis"

# ---------- Artifact Download ----------
def get_latest_artifacts() -> Dict[str, List[dict]]:
    """
    Get the latest artifact for each project.
    Returns a dict mapping repo_name -> list with single artifact dict.
    Uses the same logic as collect.py to determine which prefix to use for each project.
    """
    artifacts_by_repo = {}
    
    # Use the same logic as collect.py: iterate through REPOS and check line number
    # Projects on lines 1-45 use "continuous-analysis-future-filtered-results-"
    # Projects on lines 46+ use "continuous-analysis-filtered-results-"
    print(f"Processing {len(REPOS)} repositories...")
    
    for idx, repo_name in enumerate(REPOS):
        try:
            owner, repo = repo_name.split("/", 1)
            
            # Line number is idx + 1 (1-based)
            line_number = idx + 1
            if line_number <= 45:
                # Projects on lines 1-45 use future prefix
                prefix = FUTURE_PREFIX
            else:
                # Projects on lines 46+ use regular prefix
                prefix = REGULAR_PREFIX
            
            print(f"\nChecking {repo_name} (line {line_number}, prefix: {prefix})...")
            
            # Get the latest artifact with the appropriate prefix
            artifact = latest_with_prefix(owner, repo, prefix)
            
            if artifact:
                artifacts_by_repo[repo_name] = [artifact]
                print(f"  Found latest artifact: {artifact['name']} (created: {artifact.get('created_at', 'N/A')})")
            else:
                print(f"  No artifacts found")
        
        except Exception as e:
            print(f"  Error processing {repo_name}: {e}")
            continue
    
    return artifacts_by_repo

def download_artifact_zip(artifact: dict) -> bytes:
    """Download the artifact zip file."""
    download_url = artifact.get("archive_download_url")
    if not download_url:
        raise ValueError(f"Artifact {artifact.get('name')} has no download URL")
    
    print(f"  Downloading {artifact['name']}...")
    return gh_get_bin(download_url)

# ---------- GitHub Release Creation ----------
def gh_post(url: str, data: dict) -> dict:
    """POST to GitHub API."""
    r = requests.post(url, headers=HDRS, json=data, timeout=60)
    if r.status_code == 422:
        # Check if it's a "release already exists" error
        error_data = r.json()
        if "already_exists" in str(error_data).lower() or "already exists" in str(error_data):
            raise ValueError(f"Release with tag '{data.get('tag_name')}' already exists")
    r.raise_for_status()
    time.sleep(0.2)
    return r.json()

def gh_post_upload(url: str, file_data: bytes, content_type: str, filename: str) -> dict:
    """Upload a file to GitHub API."""
    headers = HDRS.copy()
    headers["Content-Type"] = content_type
    r = requests.post(
        f"{url}?name={filename}",
        headers=headers,
        data=file_data,
        timeout=300
    )
    r.raise_for_status()
    time.sleep(0.2)
    return r.json()

def create_release(
    owner: str,
    repo: str,
    tag: str,
    name: str,
    body: str,
    artifacts_by_repo: Dict[str, List[dict]]
) -> dict:
    """
    Create a GitHub release with artifacts as attachments.
    
    Args:
        owner: Repository owner
        repo: Repository name
        tag: Release tag (e.g., "artifacts-20250115-120000")
        name: Release name
        body: Release body/description
        artifacts_by_repo: Dict mapping repo_name -> list of artifact dicts
    
    Returns:
        Release dict
    """
    # Create the release
    release_data = {
        "tag_name": tag,
        "name": name,
        "body": body,
        "draft": False,
        "prerelease": False
    }
    
    print(f"\nCreating release {tag}...")
    try:
        release = gh_post(f"{API}/repos/{owner}/{repo}/releases", release_data)
        release_id = release["id"]
        upload_url = release["upload_url"].replace("{?name,label}", "")
        print(f"Release created: {release['html_url']}")
    except ValueError as e:
        if "already exists" in str(e):
            print(f"Release {tag} already exists. Skipping creation.")
            # Try to get the existing release
            try:
                releases = gh_get(f"{API}/repos/{owner}/{repo}/releases/tags/{tag}")
                release = releases
                release_id = release["id"]
                upload_url = release["upload_url"].replace("{?name,label}", "")
                print(f"Using existing release: {release['html_url']}")
            except Exception as get_err:
                print(f"Could not retrieve existing release: {get_err}")
                raise
        else:
            raise
    
    # Upload artifacts as release assets
    uploaded_count = 0
    for repo_name, artifacts in artifacts_by_repo.items():
        for artifact in artifacts:
            try:
                # Download artifact
                zip_data = download_artifact_zip(artifact)
                
                # Create a zip file with a descriptive name
                safe_repo_name = repo_name.replace("/", "-")
                artifact_filename = f"{safe_repo_name}-{artifact['name']}.zip"
                
                print(f"  Uploading {artifact_filename}...")
                gh_post_upload(
                    upload_url,
                    zip_data,
                    "application/zip",
                    artifact_filename
                )
                uploaded_count += 1
                print(f"    ✓ Uploaded {artifact_filename}")
            
            except Exception as e:
                print(f"    ✗ Failed to upload artifact {artifact.get('name')}: {e}")
                continue
    
    print(f"\n✓ Release created with {uploaded_count} artifacts")
    return release

# ---------- Main Workflow ----------
def main():
    if not TOKEN:
        print("Error: GitHub token not found. Set ORG_WIDE_TOKEN, GITHUB_TOKEN, or GH_TOKEN")
        sys.exit(1)
    
    # Get latest artifacts for each project
    print("=" * 60)
    print("Latest Artifact Release Workflow")
    print("=" * 60)
    artifacts_by_repo = get_latest_artifacts()
    
    if not artifacts_by_repo:
        print("\nNo artifacts found")
        sys.exit(0)
    
    # Get current repo info
    owner, repo = get_current_repo()
    print(f"\nCurrent repository: {owner}/{repo}")
    
    # Create release tag and name using timestamp
    now = dt.datetime.now(dt.timezone.utc)
    tag = f"artifacts-{now.strftime('%Y%m%d-%H%M%S')}"
    name = f"Latest Artifacts - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    # Create release body
    total_repos_num = len(REPOS)
    repos_with_artifacts_num = len(artifacts_by_repo)
    repos_without_artifacts = [repo for repo in REPOS if repo not in artifacts_by_repo]
    repos_without_artifacts_num = len(repos_without_artifacts)
    
    body_lines = [
        "Latest artifacts from all repositories",
        "",
        "## Summary",
        "",
        f"- **Total number of repositories**: {total_repos_num}",
        f"- **Number of repositories with artifacts**: {repos_with_artifacts_num}",
        f"- **Number of repositories without artifacts**: {repos_without_artifacts_num}",
        ""
    ]
    
    if repos_without_artifacts:
        body_lines.extend([
            f"## Repositories without artifacts:",
            f"- {', '.join(repos_without_artifacts)}",
            ""
        ])
    
    body_lines.extend([
        f"## Repositories with artifacts:",
        f"- {', '.join(artifacts_by_repo.keys())}",
        ""
    ])
    
    body = "\n".join(body_lines)
    
    # Create the release
    release = create_release(
        owner,
        repo,
        tag,
        name,
        body,
        artifacts_by_repo
    )
    
    print(f"\n✓ Success! Release: {release['html_url']}")

if __name__ == "__main__":
    main()

