#!/usr/bin/env bash
set -euo pipefail

LIST_FILE="projects.txt"
WORKDIR="repos"
TEMPLATE_DIR="templates"
LOGDIR="logs"
COMMIT_AND_PUSH_CHANGES="1"

# Overwrite existing directories
rm -rf "$WORKDIR" "$LOGDIR"
mkdir "$WORKDIR" "$LOGDIR"

# ---- helper: extract UPSTREAM_REPO and BRANCH from .github/workflows/monitor-upstream-and-analyze.yml (always exists) ----
extract_repo_branch() {
  local file="$1"
  local upstream_repo="" branch=""
  if [[ -f "$file" ]]; then
    upstream_repo=$(grep 'UPSTREAM_REPO: "' "$file" 2>/dev/null | sed -n 's/.*UPSTREAM_REPO:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
    branch=$(grep 'BRANCH: "' "$file" 2>/dev/null | sed -n 's/.*BRANCH:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
  fi
  echo "${upstream_repo:-databricks/databricks-ai-bridge}"
  echo "${branch:-main}"
}

# ---- helper: apply/overwrite workflow templates ----
apply_workflow_templates() {
  local repo_dir
  repo_dir="$(cd "$1" && pwd)"
  local workflows_dir="$repo_dir/.github/workflows"

  mkdir -p "$workflows_dir"

  # Run files: copy template directly (replace or add)
  for f in run-analysis.yml run-filter.yml; do
    if [[ ! -f "$TEMPLATE_DIR/$f" ]]; then
      echo "ERROR: missing template: $TEMPLATE_DIR/$f" >&2
      return 1
    fi
    cp -f "$TEMPLATE_DIR/$f" "$workflows_dir/$f"
  done

  # Read UPSTREAM_REPO and BRANCH from repo's .github/workflows/monitor-upstream-and-analyze.yml (always exists)
  local upstream_repo branch
  { read -r upstream_repo; read -r branch; } < <(extract_repo_branch "$workflows_dir/monitor-upstream-and-analyze.yml")

  # Monitor files: write template with repo-specific values in all three
  for f in \
    monitor-open-prs-and-analyze.yml \
    monitor-upstream-and-analyze.yml \
    monitor-upstream.yml
  do
    if [[ ! -f "$TEMPLATE_DIR/$f" ]]; then
      echo "ERROR: missing template: $TEMPLATE_DIR/$f" >&2
      return 1
    fi

    sed -e "s#UPSTREAM_REPO: \"databricks/databricks-ai-bridge\"#UPSTREAM_REPO: \"$upstream_repo\"#" \
        -e "s#BRANCH: \"main\"#BRANCH: \"$branch\"#" \
        "$TEMPLATE_DIR/$f" > "$workflows_dir/$f"
  done
}

while IFS= read -r line; do
  [[ -z "${line// }" ]] && continue

  # Parse: first field is owner/repo
  IFS=';' read -r owner_repo _rest <<<"$line"
  if [[ -z "${owner_repo:-}" ]]; then
    echo "[SKIP] malformed line: $line" | tee -a "$LOGDIR/summary.log"
    continue
  fi

  repo_slug="${owner_repo//\//-}"   # owner-repo
  repo_dir="$WORKDIR/$repo_slug"
  repo_log="$LOGDIR/$repo_slug.log"
  clone_url="git@github.com-research:${owner_repo}.git"
  

  echo "=== $owner_repo ===" | tee -a "$LOGDIR/summary.log"

  # Clone or update
  if [[ ! -d "$repo_dir/.git" ]]; then
    echo "[CLONE] $clone_url" | tee -a "$LOGDIR/summary.log"
    if ! git clone --quiet "$clone_url" "$repo_dir" &>>"$repo_log"; then
      echo "[FAIL] clone $owner_repo" | tee -a "$LOGDIR/summary.log"
      continue
    fi
  else
    echo "[FETCH] $owner_repo" | tee -a "$LOGDIR/summary.log"
    git -C "$repo_dir" fetch --all --prune &>>"$repo_log" || true
  fi

  # Checkout default branch (best-effort)
  default_branch="$(git -C "$repo_dir" remote show origin 2>>"$repo_log" | awk '/HEAD branch/ {print $NF}' || true)"
  if [[ -n "${default_branch:-}" ]]; then
    git -C "$repo_dir" checkout -q "$default_branch" &>>"$repo_log" || true
    git -C "$repo_dir" pull -q --ff-only &>>"$repo_log" || true
  fi

  # Apply templates (replace-or-add)
  if apply_workflow_templates "$repo_dir" &>>"$repo_log"; then
    echo "[OK] workflows updated" | tee -a "$LOGDIR/summary.log"
  else
    echo "[FAIL] workflows update" | tee -a "$LOGDIR/summary.log"
    continue
  fi

  # Optionally commit if changed
  if [[ "$COMMIT_AND_PUSH_CHANGES" == "1" ]]; then
    if ! git -C "$repo_dir" diff --quiet; then
      git -C "$repo_dir" add .github/workflows &>>"$repo_log" || true
      git -C "$repo_dir" commit -m "chore: update GitHub Actions workflows schedule" &>>"$repo_log" || true
      git -C "$repo_dir" push &>>"$repo_log" || true
      echo "[COMMIT] created commit and pushed" | tee -a "$LOGDIR/summary.log"
    else
      echo "[NOOP] no changes" | tee -a "$LOGDIR/summary.log"
    fi
  fi

done < "$LIST_FILE"

echo "Done. Summary: $LOGDIR/summary.log"
