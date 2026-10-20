# curLit supervised Beads workflow (CL-doqh)

AGENTS.md is the shared development policy. Use Beads for all issue tracking;
do not create markdown TODO lists or a second tracker. Work only on the issue
assigned by the operator. Development remains credential-free; operational
publishing follows the scoped authorization in AGENTS.md, not Beads defaults.

## Implementer commands

Use sandbox mode to disable automatic synchronization, and JSON for tool output:

```bash
bd --sandbox ready --json
bd --sandbox show <id> --json
bd --sandbox update <id> --claim --json
bd --sandbox create "Finding" -t bug -p 1 --deps discovered-from:<id> --description "Evidence and scope" --design "Proposed fix" --acceptance "Observable checks" --validate --json
bd --sandbox close <id> --reason "Implemented and verified; awaiting operator review" --json
```

Claim only assigned work. Keep incomplete work open with an accurate status.
Local issue completion alone is not merge, deployment, or trading authorization.
The operator's October 20 standing request authorizes commit/push/targeted deploy
after verified fixes; failed checks or unresolved rollout blockers stop deployment.

## Reviewer commands

Reviewers do not modify files or issue state. Inspect with:

```bash
bd --sandbox --readonly show <id> --json
bd --sandbox --readonly ready --json
```

## Handoff

Report the patch summary, exact check commands and exit codes, baseline versus
new failures, and remaining concerns. Stop edits before independent review.
After verified assigned fixes, commit, push without force, and deploy affected
services under the operator's standing request in AGENTS.md. Record remote SHA,
runtime verification and Beads status. Never bypass safety/rollout gates; no
unrelated service restart or trading-policy change. Merging, destructive cleanup
and remote issue synchronization still need separate authorization. Do not use
operational access for development tests. A model approval never replaces tests
or the operator's authority.
