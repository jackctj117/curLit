#!/usr/bin/env bash
# CL-tsj: git-based deploy workflow with signed commits, test gate,
# and graceful restart.
#
# Run from a controller machine. Pulls latest main on the production
# server, runs tests, applies migrations, and graceful-restarts the
# engine via systemd.
#
# Usage:
#   ./deploy.sh [--target fx-server] [--branch main] [--skip-tests]
#
# Required:
#   - SSH access to $TARGET as fx-operator
#   - Signed commits enforced on origin/main (server validates)
#   - .gnupg trust ring on the server includes maintainer keys

set -euo pipefail

TARGET=${TARGET:-fx-server}
BRANCH=${BRANCH:-main}
SKIP_TESTS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)     TARGET=$2; shift 2 ;;
        --branch)     BRANCH=$2; shift 2 ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

echo "[deploy] Target: $TARGET  Branch: $BRANCH"

# ----------------------------------------------------------------------
# 1. Verify the controller's current main is signed and up to date.
# ----------------------------------------------------------------------
git fetch origin "$BRANCH"
HEAD_SHA=$(git rev-parse "origin/$BRANCH")
echo "[deploy] Deploying $HEAD_SHA"

if ! git verify-commit "$HEAD_SHA" >/dev/null 2>&1; then
    echo "[deploy] ERROR: $HEAD_SHA is not a signed commit. Refusing to deploy." >&2
    echo "[deploy] Re-sign the tip with: git commit --amend -S --no-edit" >&2
    exit 1
fi
echo "[deploy] Signature verified"

# ----------------------------------------------------------------------
# 2. Run the test gate locally on the controller.
# ----------------------------------------------------------------------
if [[ $SKIP_TESTS -eq 0 ]]; then
    echo "[deploy] Running tests"
    .venv/bin/python -m pytest tests/unit/ -q
    .venv/bin/ruff check src/ scripts/
    .venv/bin/mypy src/
    echo "[deploy] Tests passed"
fi

# ----------------------------------------------------------------------
# 3. Push the code to the server.
# ----------------------------------------------------------------------
echo "[deploy] Pushing to $TARGET"
ssh "$TARGET" "cd /opt/curlit && \
    sudo -u fx-operator git fetch --quiet origin '$BRANCH' && \
    sudo -u fx-operator git verify-commit '$HEAD_SHA' && \
    sudo -u fx-operator git checkout '$BRANCH' && \
    sudo -u fx-operator git reset --hard '$HEAD_SHA'"

# ----------------------------------------------------------------------
# 4. Apply migrations + dependency updates.
# ----------------------------------------------------------------------
echo "[deploy] Applying migrations + updating dependencies"
ssh "$TARGET" "cd /opt/curlit && \
    sudo -u fx-operator /opt/curlit/.venv/bin/pip install --quiet -e . && \
    sudo -u fx-operator /opt/curlit/.venv/bin/python -m migrations.run"

# ----------------------------------------------------------------------
# 5. Graceful restart — engine drains the OMS first, then exits.
#    systemd's Restart=on-failure brings it back on the new code.
# ----------------------------------------------------------------------
echo "[deploy] Triggering graceful restart"
ssh "$TARGET" "sudo systemctl reload-or-restart fx-live-engine"

# ----------------------------------------------------------------------
# 6. Wait for healthcheck.
# ----------------------------------------------------------------------
echo "[deploy] Waiting for engine to come back healthy"
for i in {1..12}; do
    sleep 5
    if ssh "$TARGET" "curl -sf -o /dev/null http://localhost:8099/metrics"; then
        echo "[deploy] Engine healthy on $TARGET — deploy complete"
        exit 0
    fi
    echo "[deploy]   attempt $i/12: not ready yet"
done

echo "[deploy] ERROR: engine did not become healthy within 60s" >&2
exit 1
