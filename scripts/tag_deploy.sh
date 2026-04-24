#!/usr/bin/env bash
# Tag current HEAD as a deploy snapshot
set -euo pipefail
TAG="deploy-$(date +%Y%m%d-%H%M%S)"
git tag "$TAG"
echo "Tagged: $TAG"
