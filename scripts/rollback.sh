#!/usr/bin/env bash
# Rollback to a previous deploy tag
set -euo pipefail

TAG="${1:-}"

if [ -z "$TAG" ]; then
    echo "Usage: rollback.sh <deploy-tag>"
    echo "Available deploy tags:"
    git tag --list "deploy-*" | sort -r | head -10
    exit 1
fi

if ! git tag --list | grep -q "^${TAG}$"; then
    echo "ERROR: tag $TAG not found"
    exit 1
fi

echo "Rolling back to $TAG..."
git checkout "$TAG"
pip install -r requirements-lock.txt 2>/dev/null || pip install -e .
if systemctl is-active fx-live-engine &>/dev/null; then
    systemctl restart fx-live-engine
fi
echo "Rollback complete. Verify with: systemctl status fx-live-engine"
