# Changelog

All notable changes to this package will be documented in this file.

## 2.0.0 - 2026-03-09

- **Breaking:** Updated `GptConversation` calling convention.
- Expanded documentation and added extensive new examples to reflect the new API usage patterns.

## 1.4.0 - 2026-03-07

- Rewrote `gptSubmitShotgun` to build the reconciliation conversation on a separate deep copy of the original messages rather than mutating the input array.
- Strengthened the adjudication prompt: instructs the model to evaluate minority opinions carefully and avoid repeating the same reasoning errors as incorrect workers.
- The chain-of-thought ponder call now explicitly uses plain-text mode (`jsonResponse: undefined`), freeing the model to reason without schema restrictions before producing the final structured answer.
- Added shotgun integration test: verifies the shotgun strategy produces the correct letter-count answer for "strawberry milkshake", and validates the baseline failure rate by running un-shotgunned parallel attempts and asserting that at least one fails.

## 1.3.3 - 2026-03-04

- Version bump to trigger CI/CD and resolve npm trusted publishing (OIDC) issue for scoped packages. Fixed by upgrading to Node 24 and removing `registry-url` from the workflow, which was injecting a conflicting auth token.

## 1.3.2 - 2026-03-04

- Fixed npm publish workflow to remove stale `NPM_TOKEN` that was overriding the OIDC trusted publisher authentication.

## 1.3.1 - 2026-03-04

- Version bump to trigger a CI/CD pipeline test run.

## 1.2.0 - 2026-03-04

- Keep TypeScript version in sync with Python package `1.2.0`.

## 1.1.1 - 2026-03-04

- Initial release as a standalone package, split out from the mdi-llmkit monorepo.
- Includes `gptSubmit`, `GptConversation`, and `JSONSchemaFormat`.

## 1.0.6 - 2026-02-27

- Version bump to align Python/TypeScript package versions and trigger a release pipeline test run.

## 1.0.5 - 2026-02-27

- Added repository metadata in `package.json` to satisfy npm provenance validation.
- Bumped package version for another release attempt.

## 1.0.4 - 2026-02-27

- Updated TypeScript release workflow to use token-based npm authentication (`NPM_TOKEN`).
- Bumped package version for a fresh release attempt.

## 1.0.3 - 2026-02-27

- Version bump only, to trigger the release pipeline.

## 1.0.2 - 2026-02-27

- Patch-level release to keep npm and PyPI package versions aligned.
- No TypeScript API behavior changes in this patch.

## 1.0.1 - 2026-02-27

- Promoted the package to the first stable line with immediate patch-level hardening.
- Added CI diagnostics that log a masked `OPENAI_API_KEY` fingerprint for easier secret troubleshooting.
- Normalized test API-key handling with trimming to avoid hidden-whitespace secret issues in CI.

## 0.1.0 - 2026-02-25

- Added `gptSubmit` helper with retry behavior, datetime system-message injection, JSON mode support, and warning callback support.
- Added `GptConversation` with role helpers, submit wrappers, and last-reply convenience accessors.
- Added `JSONSchemaFormat` with compact DSL support and recursive schema expansion for OpenAI Structured Outputs.
- Added parity-oriented test coverage for submit helpers, conversation helpers, and JSON schema edge cases.
- Added TypeScript package README usage examples and Python-to-TypeScript migration notes.
