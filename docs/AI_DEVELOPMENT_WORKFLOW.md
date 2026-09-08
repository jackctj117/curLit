# Supervised AI development (CL-doqh)

One assigned Beads issue → Claude patch → checks → independent Codex review →
corrections and repeated checks/review → operator approval. Fable is an optional,
task-specific escalation, not a scheduled agent. No orchestration framework or
automatic publishing is configured.

## Prepare a credential-free environment

Use a development container, VM, or separate OS account that cannot read the
trading installation, its `.env`, vault, credentials, database, or service sockets.
Do not mount the production checkout, home directory, Docker socket, or trading
volumes into it. Keep write-capable external connectors out of the reviewer.
A different checkout and virtualenv alone do not provide this isolation.
This repository configuration does not create or verify that OS boundary.
Operator activation and isolation evidence are tracked in CL-xpsp.

The operator can install the clients in that environment if needed. Check versions
with `claude --version` and `codex --version`; account sign-in and model entitlement
remain operator steps. Do not upgrade tools in the running trading installation.

From the isolated environment, the operator prepares a separate checkout:

```bash
git clone https://github.com/jackctj117/curLit.git curLit-dev
cd curLit-dev
git switch -c chore/ai-development-workflow
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Do not copy production environment variables or databases. Integration tests use
a disposable test database only. Keep the operational paper installation unchanged.

## Configuration and verification

`AGENTS.md` owns the shared safety, review, and completion policy. `CLAUDE.md`
imports it using `@AGENTS.md` and keeps the architecture below it. Claude supports
these imports in its [memory documentation](https://code.claude.com/docs/en/memory).

For a new checkout, merge `.claude/settings.local.example.json` into
`.claude/settings.local.json`; create the local file from the example only if it
does not exist. Preserve unrelated settings, review old pre-approvals, and remove
permissions for operational access, publishing, destructive cleanup, and broad
shell/interpreter execution. Do not copy another checkout's accumulated approvals.
The local file is ignored by Git, so review it locally without publishing it.
The example, shared hooks, and Codex config are the portable part of this setup.

Claude uses `claude-opus-5`, default/manual permission mode, and disables bypass
and auto modes. Matching credential reads, pushes, daemon commands, merges, and
destructive Git cleanup are denied. These rules are defense in depth, not a
complete shell or credential isolation mechanism. Review effective permissions
with `/permissions`, including user settings and connectors; approve only commands
needed by the assigned issue. See [permissions](https://code.claude.com/docs/en/permissions)
and [local settings](https://code.claude.com/docs/en/settings).

Both Claude hooks run `bd --sandbox --readonly prime`. The supported
`.beads/PRIME.md` override replaces Beads' default push-based completion policy;
`--sandbox` disables automatic synchronization. Local backup pushing is disabled
in `.beads/config.yaml`. Check effective guidance with:

```bash
bd --sandbox --readonly prime
```

Do not replace this with `bd prime --export`, which bypasses the override. Check
the effective output again after a Beads upgrade or integration regeneration.
Implementers use `bd --sandbox ... --json` for local issue updates; reviewers add
`--readonly` and do not claim, close, or otherwise mutate issues. Operator-controlled
remote issue sync is separate from the agent handoff.

`.codex/config.toml` sets both `model` and `review_model` to `gpt-6-astra`, with
`sandbox_mode = "read-only"` and `approval_policy = "never"`. Here `never` prevents
escalation requests, not sandbox restrictions. Project configuration requires a
trusted project. Before reviewing, verify effective settings in the client; resolve
any higher-priority override or incompatible permission profile without weakening
read-only review. These settings apply to new sessions, not an already-running
implementation session. See the [Codex configuration reference](https://developers.openai.com/codex/config-reference).

Model names are requested defaults, not proof of account entitlement. Confirm the
selected model in the client. If unavailable, the operator chooses an available
model while retaining the role and permission restrictions; do not silently fall
back or launch paid calls merely to test availability.

## Implement, verify, review

The operator assigns one bounded issue. The implementer inspects and claims it:

```bash
bd --sandbox ready --json
bd --sandbox show <issue-id> --json
bd --sandbox update <issue-id> --claim --json
claude --model claude-opus-5
```

Capture the baseline before behavior changes, then repeat relevant regression
tests and these repository checks after the patch. Record every command's exit
code and test counts, with existing failures separated from new failures:

```bash
.venv/bin/pytest tests/unit -q
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/
```

For this setup's configuration regressions:

```bash
.venv/bin/pytest tests/unit/test_ai_development_workflow.py -q
```

Do not use production access to make a test pass. Tests/CI remain required.
CL-0deu.5.1 adds [PR unit, lint, and type checks](CI.md); publishing the workflow,
verifying its hosted run, and making checks required are operator steps. The
remaining CI/security/migration work is tracked by CL-0deu.5. Until activation is
verified, the operator must check the evidence before approving a merge.

Stop Claude's editing session, ensure the diff is stable, and choose exactly one
review target. For an uncommitted patch:

```bash
codex review --uncommitted
```

For committed changes on a feature branch relative to main:

```bash
codex review --base main
```

Do not combine target flags or append a custom prompt to them. Persistent criteria
are in `AGENTS.md`. The [CLI reference](https://developers.openai.com/codex/cli/reference)
documents these mutually exclusive targets. Reviewers do not repair the patch or
rerun write-producing tests in the read-only checkout; they inspect the diff and
test evidence and report any verification gap. Ignored local settings are outside
the Git diff and need a separate local operator inspection.

Send concrete findings back to the implementer; repeat checks and review after
corrections. Hand off changed files, behavior rationale, exact verification,
unresolved concerns, and local Beads status. Closing an implementation bead is not
approval to merge. The operator publishes, approves merging, and separately decides
deployment. Neither passing tests nor a review with no findings is sufficient alone.

## Escalate only when needed

For a specifically assigned hard design/debugging question, use a separate session:

```bash
claude --model fable
```

With a supported client (2.1.255+) and no alias override, `fable` selects Fable 5.1.
Confirm the actual selection and billing before proceeding; non-interactive Fable
requests can bill usage credits without a prompt. See [model configuration](https://code.claude.com/docs/en/model-config).

Example bounded escalation: inspect the order-reconciliation design without edits;
identify counterexamples involving partial fills, submission timeouts, duplicate
events, and crashes between persistence steps; propose the smallest correction.
Return the result to the assigned Beads issue and the normal implementation loop.

## Runtime research is separate

This setup does not replace Kimi, change `configs/research_agents.yaml`, change
`NICHE_TOOL_AGENT_ENABLED`, alter runtime cost reporting, or restart daemons.
Provider-neutral niche discovery, shadow comparisons, and unknown-versus-measured
cost accounting need separately assigned implementation work. Do not deploy these
development settings into the trading installation as a model migration.

The immediate Alpaca position-lookup interlock was already implemented in
CL-0deu.1.1; do not redo it from the earlier example prompt. Verify current code
before taking the next assigned hardening issue. Pending-order exposure and
reservations remain part of CL-0deu.1 and related roadmap work.
