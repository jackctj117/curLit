# curLit supervised Beads workflow (CL-doqh)

AGENTS.md is the shared development policy. Use Beads for all issue tracking;
do not create markdown TODO lists or a second tracker. Work only on the issue
assigned by the operator. Runtime and broker access are out of scope.

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
Local issue completion is not merge, deployment, or trading authorization.

## Reviewer commands

Reviewers do not modify files or issue state. Inspect with:

```bash
bd --sandbox --readonly show <id> --json
bd --sandbox --readonly ready --json
```

## Handoff

Report the patch summary, exact check commands and exit codes, baseline versus
new failures, and remaining concerns. Stop edits before independent review.
Publishing, merging, deployment, and remote issue synchronization belong to the
operator. Do not push, rebase, deploy, clear stashes, or prune branches as part of
completion. Do not access broker accounts, operational databases, or trading
daemons. A model approval never replaces tests or operator approval.
