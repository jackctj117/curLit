# Isolated research runtime (CL-esnh / CL-e1nr)

The October 23 staging environment is for the event pipeline only. It does not
replace the shared trading `.venv`, change risk controls, or activate the
shadow-only hop graph. Read CURRENT_OPERATIONS.md for the actual cutover state.

## Artifact identity

- Application source: immutable Git export of `0604f65247de4b9612c1fa09642fb02d1917cde7`.
- Interpreter: existing CPython 3.14.6, macOS arm64; no system Python upgrade.
- Package lock: [research-macos-cp314.lock](../requirements/research-macos-cp314.lock).
- Lock SHA256: `4ded5744b517a29a051f8de745af6704f9017f6dbe5df1f2302c64b453b869ca`.
- 141 core packages resolve at the exact versions from passing CI run 39642476681;
  the installed bootstrap `wheel==0.46.3` is also pinned/audited, making 142 total.
- Installed audit SHA256: `121dd6101ff11a09697dd5fca88f081c635baa01c449218ebe5e92b52b750a31`.

This is a platform-specific artifact lock, not a universal deployment profile.
All ordinary wheels came from PyPI. `hmmlearn==0.3.3` lacks a CPython 3.14 Mac
wheel, so one source build was performed in a credential-denied temporary
builder, using the reviewed CI constraints. No other source builds were allowed.
Those constraints did not pin `pybind11` or `setuptools_scm`; reproducibility
therefore means retaining the reviewed wheel artifact, not claiming a bit-exact
source rebuild.

- Source archive SHA256: `1d3c5dc4c5257e0c238dc1fe5387700b8cb987eab808edb3e0c73829f1cc44ec`.
- Built `hmmlearn-0.3.3-cp314-cp314-macosx_26_0_arm64.whl` SHA256:
  `6e9a6946b763307104ced8e1727171d928353859a61e55f6fcf0cc66e5b7f943`.

Retain that reviewed wheel in the release wheelhouse. A rebuild on another host
may have a different hash; it requires review, not an automatic hash exception.
Native validation compared a two-state, three-observation HMM's result against
explicit enumeration of all eight state paths (probability 0.088658).

## Installation and verification boundaries

The source export and new virtualenv are under
`data/releases/research-0604f65-20261023/`, separate from the active checkout's
shared interpreter. Package downloads were hash checked; installation was then
offline with no dependency re-resolution:

```sh
NEW_PYTHON -m pip --isolated install --no-index --no-deps --require-hashes \
  --only-binary=:all: --find-links=REVIEWED_WHEELHOUSE \
  -r requirements/research-macos-cp314.lock
NEW_PYTHON -m pip check
NEW_PYTHON scripts/ci_audit_manifest.py INSTALLED_MANIFEST
AUDITOR_PYTHON -m pip_audit --strict --no-deps --disable-pip \
  --progress-spinner off -r INSTALLED_MANIFEST
```

Uppercase paths above are placeholders for the new environment/artifacts, never
the shared operational environment. The actual installed inventory, including
installer packages, is audited. No vulnerabilities or unidentified distributions
are excluded. The older shared environment's diagnostic audit is explicitly not
a passing gate; its remediation remains CL-e1nr.

Staged validation ran 429 focused tests with network and operational credentials
denied. A temporary pytest tool directory was appended after runtime site-packages;
it did not override the deployed dependencies or get installed into the runtime.
Imports asserted that runtime libraries came from the new virtualenv and source
modules came from the immutable release. The one warning was pytest's inability
to rewrite an already-imported `anyio` module; no timeout or test was disabled.
Exact commands, working directories, all 18 test filenames and artifact hashes
are retained in `data/deployments/2026-10-23_research_release/validation-commands.json`.

## Targeted cutover

The verified owner is `gui/501/com.curlit.paper.event-pipeline`. Preserve its
arguments, working directory, environment and trading restrictions. Change only
the interpreter and absolute script path in its launch manifest. Keeping the
original working directory preserves the existing `.env`, configuration and
state locations; their hashes must match the preflight. The script inserts the
immutable release root before imports.

Wait for a completed idle cycle, verify the old identity, stop through launchd,
and confirm the old writer and its children are gone before bootstrapping the
replacement. Do not start a second writer with `daemons.sh`. Verify the new
singleton, actual interpreter/script paths, preserved input hashes, unaffected
trading PIDs, and a completed cycle. Startup alone is not useful niche output.

Failure means stop and investigate. Keep the prior manifest and environment for
diagnostics/recovery planning; do not automatically restore a known-vulnerable
environment or roll back research evidence gates. No broker orders, database
migration, historical correction, entry unpause or live-money activation belongs
to this research cutover.
