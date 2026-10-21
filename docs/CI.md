# Development checks (CL-0deu.5.1)

## Local offline container checks (CL-i6ke)

When host isolation blocks fixture Unix sockets, do not skip the vault tests or
weaken the sandbox. Use a disposable development container instead:

```bash
.venv/bin/python scripts/test_isolated.py \
  --context desktop-linux --image curlit-review-tests:oct6 --check vault
.venv/bin/python scripts/test_isolated.py \
  --context desktop-linux --image curlit-review-tests:oct6 --check unit
.venv/bin/python scripts/test_isolated.py \
  --context desktop-linux --image curlit-review-tests:oct6 --check lint
.venv/bin/python scripts/test_isolated.py \
  --context desktop-linux --image curlit-review-tests:oct6 --check typecheck
```

The context/image names above identify the existing local development image,
not a published dependency lock or an instruction to use the production image.
Use an explicitly selected **credential-free test image**, prepared separately
with Python, Git, and the core/dev/audio/Polymarket test dependencies.
The runner never builds, downloads, upgrades, or starts a trading image command.
It resolves the local image to an immutable ID and prints that ID, the source
archive hash, and the check's actual exit code.

Only an explicit source/fixture allowlist is archived, including current tracked
edits and non-ignored new Python files/documentation. New data/report exports are
not included automatically; fixtures of other types must be explicitly tracked.
Deleted files remain deleted. Private
environment files, runtime `data/`, logs, Git/Dolt stores, nested worktrees, and
host dependency directories are excluded. Symlinks and special files are refused.
The archive is streamed into a fresh container-private `/tmp`; **no host directory,
Docker socket, broker volume, or credential is mounted**. The process runs as a
non-root UID with dropped capabilities, no-new-privileges, no external network,
a read-only image filesystem, resource limits, and a cleared test environment.
Image entrypoint/command and healthchecks are overridden. Real AF_UNIX sockets
inside that container remain usable. Test failure is propagated, not suppressed.

The private temporary filesystem remains **noexec**. PyTensor therefore uses its
supported Python implementation (`PYTENSOR_FLAGS=...,cxx=`), which still runs the
Bayesian fitting and numerical assertions; this is printed before each check.
See [PyTensor's compiler configuration](https://pytensor.readthedocs.io/en/latest/library/config.html#config.cxx).
No tests are skipped to achieve this. The run does **not** certify the C++ backend;
validate that separately in an approved environment with native compilation.
Do not change runtime model configuration to match this test-only choice.

**Observed October 20, 2026:** all 29 vault tests pass in this container. The
Python-only Bayesian sampler exceeds the existing 120-second test deadline
(confirmed on `TestFit.test_returns_per_pair_betas`); the full container suite
is therefore **not green**. The same 15 Bayesian tests pass with native
compilation in the unchanged credential-free macOS sandbox. That host sandbox
still denies the 13 vault socket-bind cases. CL-i6ke retains this verification
gap pending operator approval for a container-private executable compiler cache.
Neither the timeout nor the filesystem restriction has been weakened.

The source allowlist and container arguments have regression tests in
`test_isolated_runner.py`. This boundary does not inspect arbitrary image contents
for secrets: vet the image before use. It also does not make dependencies locked,
certify macOS-specific behavior, or establish hosted CI/branch protection.

## Hosted checks

`.github/workflows/ci.yml` defines credential-free hosted checks for all pull
requests, pushes to `main`, merge queues, and manual dispatch. It does not publish,
deploy, operate trading services, or invoke coding/research models. The jobs use
Ubuntu 24.04 and Python 3.11, the project's declared minimum version.

| Stable check name | Command in an activated development environment |
| --- | --- |
| `unit-tests` | `make test-unit` |
| `lint` | `make lint` |
| `typecheck` | `make typecheck` |
| `recovery-integration` | `python -m pytest tests/integration/test_alpaca_ledger_postgres.py tests/integration/test_alpaca_entry_reservation_postgres.py tests/integration/test_recovery_postgres.py tests/integration/test_niche_audit_postgres.py -q -o addopts=''` |

The existing Make targets run the complete unit directory, Ruff across `src/`
and `tests/`, and strict mypy across `src/`. The jobs run independently: a failure
does not cancel the others, and none uses `continue-on-error` or an exit-zero
override. PRs have no path filters that could leave a required check pending.

Each hosted job installs `.[dev]`. Linux CI preinstalls a CPU-only PyTorch wheel
from its official index to avoid downloading unused CUDA libraries; no dependency
is omitted. `pytest-timeout` is now declared in the dev extra so the existing
120-second per-test timeout is enforced. Do not upgrade the operational `.venv`
to activate this: use a disposable development environment. The dependency lock
is separate work (CL-0deu.10); installing the same broad requirements is not yet
a guarantee of a bit-for-bit reproducible environment.

`recovery-integration` creates a disposable PostgreSQL 15 service with public
fixture credentials and database name `curlit_test_ci`. Its database URL is set
only for the integration step, never read from the operational environment. The
suite covers additive migrations, ledger replay, recovery safeguards, audit
persistence and writer fencing. It is not a full TimescaleDB backup/restore drill
or permission to test against an operational database.

## Blocking security checks (CL-0deu.5.2)

The security workflow defines independent `security-bandit`,
`security-dependencies`, and `security-secrets` jobs. All run on PRs, main pushes,
merge queues and manual dispatch, with a weekly scheduled scan as well. One
failure cannot prevent the other scans from running. Each uses read-only GitHub
permissions and checkout without persisted credentials.

- **Bandit 1.9.4:** `python -m bandit -r src/ -ll -f json -o <temporary-report>`
  rejects medium/high findings at every confidence level. A separate always-run
  summary rejects parsing errors or unreadable reports and prints only finding
  locations, rule IDs and severity/confidence, not source snippets. The original
  scan failure is not suppressed by the summary. No blanket baseline is used.
- **pip-audit 2.10.1:** strict auditing uses an inventory generated by
  `scripts/ci_audit_manifest.py`, containing every installed third-party core/dev,
  transitive and scanner dependency. Only the private curLit root is omitted.
  `--no-deps --disable-pip` avoids re-resolving an already complete installed
  inventory; it does not omit transitive packages. Collection errors and known
  vulnerabilities fail. No automatic fixes or vulnerability ignores are used.
  Official PyTorch `+cpu` builds are explicitly mapped to the same public
  release for PyPI advisory lookup (the full build version remains in a manifest
  comment); PyPI returns 404 for the suffixed identity. No other local build is
  accepted. This is advisory matching, not a wheel-integrity audit. See
  [PyTorch's release/build installation matrix](https://pytorch.org/get-started/previous-versions/).
  Both test and audit runners also install the audio and Polymarket extras
  exercised by existing fixture-backed unit tests; no account keys are supplied.
  Both CI workflows install pip 26.2.1 instead of retaining the previous
  CVE-2026-3219 exception. See the [pip release notes](https://pip.pypa.io/en/stable/news/).
  They also upgrade setuptools to at least 83.0.0: the first complete hosted
  audit flagged the runner's bundled 79.0.1 under PYSEC-2026-3447. Neither
  advisory is ignored.
- **Gitleaks 8.30.1:** the Linux binary is verified against a committed SHA-256
  before extraction. `gitleaks git --config .gitleaks.toml --redact=100 --no-banner
  --verbose --ignore-gitleaks-allow --log-opts=--all .` scans fetched Git history
  across all file types, including deleted content; it does not load the ignored
  working-copy `.env` or verify credentials against a broker/provider. Checkout
  fetches complete available history. This is not a working-tree scan: local
  uncommitted/untracked changes are outside it until the operator commits them.
  Use only isolated fixture directories for local `gitleaks dir` experiments,
  never the operational checkout containing ignored credentials.

Scanner pins and their exceptions require review on updates. These tools report
evidence, not proof of exploitability or an exhaustive security audit. See the
[Bandit CLI](https://bandit.readthedocs.io/en/latest/man/bandit.html),
[pip-audit documentation](https://github.com/pypa/pip-audit), and
[Gitleaks documentation](https://github.com/gitleaks/gitleaks).

### Exception policy and observed baseline

Fix genuine defects first. An exception must name a Bead, exact rule/location or
package/advisory, evidence, reviewer rationale, and a recheck condition. Operator
review of the patch includes its exceptions. Do not ignore entire directories,
commits, rule families, or all current findings to obtain a green check. Do not
use inline `gitleaks:allow` comments; those are disabled. Do not publish raw secret
reports or validate discovered credentials online. Real credentials require
operator-led revocation/rotation, not merely deletion from the current file.

One exception matches `generic-api-key` AND the exact secret value
`PLACEHOLDER_ECB_Q3_YES` AND `configs/polymarket_markets.yaml`. Inspection of
commit `6512b361b2dc5336a29d3a91d3510bd4c58088a0` confirms it is a literal public
instrument-ID placeholder. Revisit the exception when the sample value, file,
or detector changes. Tests with another path and a token-shaped replacement both
still fail. The 395-commit local history scan then reports no findings; this does
not certify uncommitted changes, unfetched history, or remote credentials.

CL-esh6 also scopes the exact comment text `Telegram/Moonshot` to
`src/research/llm/claude_code.py` and its security-workflow test. This is a vendor
name pair misclassified in a comment, not an actual credential. Changed values
and other paths remain subject to scanning; regression tests cover both rules.

**Bandit findings resolved (CL-u59z):** the initial scan reported 16 findings,
not 16 confirmed exploits. Ten now have code fixes and six SQL constructions
have narrow, evidence-backed dispositions. The medium-and-higher scan exits
zero with no parser errors. See [individual dispositions and regression
evidence](SECURITY_DISPOSITIONS.md). Existing model identities and persisted idea
IDs are preserved. A vulnerable/patched pip
fixture proves auditor failure/success. The complete fresh hosted dependency
environment was audited successfully on October 20 (run linked below); this
does not establish that the operational virtualenv has the same inventory.

The security workflow uses read-only repository permissions, pinned action
commits, and checkout without persisted Git credentials. It has no deployment
environment, database service, or broker keys. External smoke tests stay opt-in and model
downloads are disabled. These settings do not prove that arbitrary test code
cannot use the network; credential-free hosted runners remain important.

## Operator activation

The workflows are published and running. To enforce successful checks before
merging, require `unit-tests`, `lint`, `typecheck`, `recovery-integration`, `security-bandit`,
`security-dependencies`, and `security-secrets` in the repository's branch
rules/ruleset, with the GitHub Actions app
as the expected source. Confirm the exact names in the checks UI. Require human
PR approval and verify that a deliberately failing check prevents merging using
a disposable branch. Model review is additional evidence, not a merge authority.

**Read-only verification October 20, 2026:** the main-branch protection API
returned `404 Branch not protected` (exit 1), and the effective branch-rules API
returned an empty list (exit 0). Required checks and human-approval enforcement
are therefore **not active**. No repository setting was modified. Enabling and
testing those rules is tracked in **CL-lqor**, an operator administrative step; successful workflow
execution is not merge enforcement. Do not bypass failing jobs to merge.

## Verification and remaining scope

Configuration regression tests run in the normal unit suite:

```bash
.venv/bin/pytest tests/unit/test_ci_workflow.py -q
.venv/bin/pytest tests/unit/test_security_workflow.py -q
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/
```

When reviewing changes to CI, confirm that deliberate failing test, lint, and
type fixtures each produce a nonzero exit. Put these fixtures in a disposable
directory, not in the repository suite; never commit credentials or weaken an
existing test to exercise a failure path.

**Hosted evidence, October 20, 2026, source `63fb4a399344f60a3c5ad409adbc4b6f5f890d57`:**
[CI run 39493644618](https://github.com/jackctj117/curLit/actions/runs/39493644618)
passed the complete unit suite (**4,708 passed, two skipped, one warning** in
246.44 seconds), lint, typecheck, and disposable recovery integration (**27
passed** in 7.30 seconds). The two skips are the existing opt-in alternative-data
network test and credential-dependent Grok smoke test, not behavioral regressions
disabled for CI. [Security run 39493644568](https://github.com/jackctj117/curLit/actions/runs/39493644568)
passed Bandit, complete installed dependency auditing, and history secret scans
at the same commit. Logs and conclusions were rechecked through read-only GitHub
commands. These results do not certify later uncommitted changes.

CL-0deu.5 now has hosted installation, migration and scanner evidence; required
branch checks remain explicitly outstanding as permitted by its acceptance
criteria. CL-i6ke separately retains the local offline-container compiler
limitation described above; passing hosted tests does not make that runner green.
CL-dzt6 tracks fresh-environment dependency/audit repairs. CL-6sdh records
Linux/concurrency fixture assumptions exposed by the first full hosted suite;
the operator authorized their correction on 2026-09-08. Four concurrent workers
now rendezvous before checking thread-local isolation (avoiding recycled OS
thread IDs). Unauthorized vault peers must receive **zero raw response bytes**
on either EOF or reset, and must never reach the protocol decoder. Authorized
requests still require complete frames. Mutation cases reject a shared client
and partial/full leaked responses; no production auth or cache behavior changed.
Publishing the hardening code does not mean deployment has passed these gates.
CL-xpsp separately tracks the credential-free coding/review environment. None of
these checks authorizes paper-account resets, daemon restarts, or live trading.
