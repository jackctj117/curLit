# Security finding dispositions — CL-u59z

Reviewed 2026-09-08 against Bandit 1.9.4, `bandit -r src/ -ll`.
These were 16 scanner findings, not 16 independently confirmed vulnerabilities.
No baseline, global rule exclusions, or confidence downgrade is used.

| Finding location (function) | Disposition and evidence |
| --- | --- |
| B314 `data/alternative.py` (`EntsoeSource._parse_load_xml`) | Use defusedxml with DTD forbidden; malicious XML raises, existing ENTSO-E quantity/time fixtures preserved. |
| B314 `data/truth_social.py` (`parse_feed`) | Same parser policy; unsafe/malformed feeds produce a logged empty result. Existing RSS post fixtures preserved. |
| B314 `research/ingest.py` (`ArxivFetcher._parse`) | Same policy; fetch wrapper catches security/parser failures. Existing Atom paper fixtures preserved. |
| B314 `research/ingest.py` (`RSSFetcher._parse`) | Same policy and failure handling; existing RSS paper fixtures preserved. |
| B324 `events/idea_ledger.py` (`make_idea_id`) | SHA-1 is explicitly non-security: `usedforsecurity=False`. Changing the digest would break persisted dedup IDs. OpenSSL-generated independent vector verifies compatibility. This is not suitable for authentication or adversarial collision resistance. |
| B615 `nlp/inference.py` (tokenizer load) | Pass validated immutable revision, disable remote code; explicit local paths use local-only loading. |
| B615 `nlp/inference.py` (model load) | Same revision as tokenizer. Pin each existing registry model, retain its identity/label mapping. Invalid or absent revisions for unregistered remote models fail before loading. |
| B614 `nlp/inference.py` (temperature sidecar) | Explicit `weights_only=True`, regardless of Torch's version-dependent default. Tiny real tensor/scalar files round-trip; arbitrary checkpoint classes are rejected. Only the resolved local checkpoint can supply calibration. |
| B608 `data/symbols.py` (`_load_cache`) | Replace column interpolation with two complete literal queries. Existing SEC-column and pre-migration SQLite fixtures cover both. |
| B608 `models/feature_versioning.py` (`store`) | Remove redundant dialect-dependent suffix: both branches already used identical `ON CONFLICT` syntax. Fixed SQL, bound values; existing round-trip/idempotence fixtures. |
| B608 `data/provider.py` (`get_realized_vol`) | Narrow line-specific disposition: interpolated cutoff is empty or fixed `AND ts <= :cutoff`; pair, cutoff and limit are bound. Hostile-symbol regression covers both branches. |
| B608 `data/provider.py` (`get_intraday_values_batch`) | Narrow disposition: floor clause is empty or fixed SQL; symbols use expanding binds, timestamps remain bound. Hostile-symbol regression covers both branches. |
| B608 `execution/alpaca_equity_executor.py` (`fetch_executable_ideas`) | Narrow disposition: every `where` item is a source literal, never configuration text. Confidence is bound. Tests cover all niche/red-team combinations with hostile confidence input. |
| B608 `execution/alpaca_options_executor.py` (`fetch_executable_ideas`) | Same trust boundary; options urgency filtering unchanged. Same hostile-input/filter matrix. |
| B608 `execution/alpaca_equity_exit.py` (`_mark_exit`) | Narrow disposition: `fill_sql` selects two literal assignments. All identifiers/data values are bound. Actual SQLite execution with an injection-shaped idea ID updates only its own row in both modes. |
| B608 `portfolio/attribution.py` (`compute_strategy_pnl`) | Narrow disposition: `where` joins only fixed strategy/time predicates, values bound. SQLite hostile-strategy fixtures cannot read another strategy's inventory, with/without cutoff. |

Focused new evidence is in `tests/unit/test_security_hardening.py`; existing
parser, provider, symbol, execution, attribution and feature-store tests remain
unchanged. Three DTD attack cases failed before the XML fix. No existing test
expectation was weakened.

## Model provenance and remaining boundaries

Registry revisions were resolved from each existing model's official
`https://huggingface.co/api/models/<model-id>` metadata on 2026-09-08.
All eight repositories advertise safetensors artifacts. The Fed revision
`7695c0aebcd1a85ee23ff41df6a57b024e20f82b` also matches the deployment's
pre-change cached `refs/main`. No new model family/provider is introduced.
Future revision updates require review; a commit pin establishes reproducibility,
not an independent audit of training data or model quality. Local directories
and the installed libraries still require a trusted filesystem/environment.

Implementation follows the documented [Transformers revision and remote-code
controls](https://huggingface.co/docs/transformers/main/en/model_doc/auto),
[PyTorch restricted loading](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
and [defusedxml parser controls](https://github.com/tiran/defusedxml).

These changes do not implement pending-order reservations, a fill-based ledger,
or the remaining trading-hardening roadmap. A clean scanner is not operational
or strategy approval.
