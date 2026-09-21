#!/usr/bin/env bash
# Creates/pushes https://github.com/kveni12/close-incidental-findings
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PATH="${HOME}/bin:${PATH}"

if ! command -v gh >/dev/null 2>&1; then
  echo "Install GitHub CLI (gh) first." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "Authenticating gh (SSH + repo scope)..."
  gh auth login -h github.com -p ssh -s repo --skip-ssh-key
fi

if gh repo view kveni12/close-incidental-findings >/dev/null 2>&1; then
  git remote remove origin 2>/dev/null || true
  git remote add origin git@github.com:kveni12/close-incidental-findings.git
  git push -u origin main
else
  gh repo create kveni12/close-incidental-findings \
    --public \
    --description "CLOSE: Capture, Link, Orchestrate for incidental findings (research prototype)" \
    --source . \
    --remote origin \
    --push
fi

echo "Live at: https://github.com/kveni12/close-incidental-findings"
