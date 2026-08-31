#!/usr/bin/env bash
# Regenerate tests/fixtures/cython_baseline/*.npz from the old C/Cython
# extension, per the steps documented in gen_golden.py's docstring.
set -euo pipefail

BASELINE_COMMIT="9a11eed"
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_SCRIPT="${SCRIPT_DIR}/gen_golden.py"
WORKTREE_DIR="$(mktemp -d -t pf-baseline-XXXXXX)"

cleanup() {
  cd "${REPO_ROOT}"
  git worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

echo "== worktree @ ${BASELINE_COMMIT} -> ${WORKTREE_DIR}"
git -C "${REPO_ROOT}" worktree add "${WORKTREE_DIR}" "${BASELINE_COMMIT}"

cd "${WORKTREE_DIR}"
uv venv .venv
uv pip install --python .venv/bin/python -e .

echo "== running gen_golden.py"
"${WORKTREE_DIR}/.venv/bin/python" "${GEN_SCRIPT}"

echo "== done, fixtures in ${REPO_ROOT}/tests/fixtures/cython_baseline/"
