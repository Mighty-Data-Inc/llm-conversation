# Changelog

All notable changes to this package will be documented in this file.

## 2.0.0 - 2026-03-09

- **Breaking:** Updated `GptConversation` calling convention.
- Expanded documentation and added extensive new examples to reflect the new API usage patterns.

## 1.4.0 - 2026-03-07

- Rewrote `_gpt_submit_shotgun` to build the reconciliation conversation on a separate deep copy of the original messages rather than mutating the input list.
- Strengthened the adjudication prompt: instructs the model to evaluate minority opinions carefully and avoid repeating the same reasoning errors as incorrect workers.
- The chain-of-thought ponder call now explicitly uses plain-text mode (no JSON constraint), freeing the model to reason without schema restrictions before producing the final structured answer.
- Added `test_shotgun_to_get_reliable_answer_on_unreliable_prompt`: an integration test that (a) verifies the shotgun strategy produces the correct letter-count answer, and (b) validates the baseline failure rate by running un-shotgunned parallel attempts and asserting that at least one fails.

## 1.3.2 - 2026-03-04

- Fixed remaining `patch()` target strings in `test_gpt_submit.py` that still referenced the old `gpt_conversation` module name.

## 1.3.1 - 2026-03-04

- Version bump to trigger a CI/CD pipeline test run.

## 1.3.0 - 2026-03-04

- **Breaking:** Renamed the importable package from `gpt_conversation` to `mightydatainc_gpt_conversation` to match the PyPI distribution name convention. Clients must update imports to `from mightydatainc_gpt_conversation import ...`.

## 1.2.0 - 2026-03-04

- Fixed syntax error in `test_integration.py` where the opening `{` was missing from the `JSONSchemaFormat` dict argument.

## 1.1.1 - 2026-03-04

- Initial release as a standalone package, split out from the mdi-llmkit monorepo.
- Includes `gpt_submit`, `GptConversation`, and `json_schema_format`.

## 1.1.0 - 2026-02-28

- Removed root-level convenience re-exports from `mdi_llmkit` so package usage is subpackage-only.
- Preserved subpackage import paths for public APIs (`mdi_llmkit.gpt_api`, `mdi_llmkit.json_surgery`, `mdi_llmkit.semanticMatch`).
- Added import-surface regression coverage to ensure root exports remain empty while subpackage imports continue to work.

## 1.0.6 - 2026-02-27

- Version bump to align Python/TypeScript package versions and trigger a release pipeline test run.

## 1.0.4 - 2026-02-27

- Fixed an intermittently failing unit test by tightening prompt behavior for deterministic outcomes.

## 1.0.3 - 2026-02-27

- Version bump only, to trigger the release pipeline.

## 1.0.2 - 2026-02-27

- Added `mdi_llmkit.semanticMatch.compare_item_lists` with deterministic pre-processing plus LLM-guided rename/add/remove classification.
- Added Python comparison API types and callback contracts (`SemanticallyComparableListItem`, `ItemComparisonResult`, `OnComparingItemCallback`, `StringListComparison`).
- Added live API test coverage for comparison behavior and callback telemetry in `tests/test_compare_lists.py`.
- Added Python README documentation for semantic list comparison usage and input formats.

## 1.0.1 - 2026-02-27

- Promoted the package to the first stable line with immediate patch-level hardening.
- Normalized test API-key handling with trimming to avoid hidden-whitespace secret issues in CI.
- Retained masked CI secret diagnostics in workflow to make future environment debugging faster.

## 0.1.0 - 2026-02-25

- Added `gpt_submit` helper with retry behavior, datetime system-message injection, JSON mode support, and warning callback support.
- Added `GptConversation` with role helpers, submit wrappers, and last-reply convenience accessors.
- Added `json_surgery` iterative JSON mutation workflow with validation/progress hooks and iteration/time safety limits.
- Added package test coverage for GPT API helpers, schema formatting, JSON surgery, and subpackage imports.
