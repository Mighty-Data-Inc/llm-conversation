# Changelog

All notable changes to this package will be documented in this file.

## [1.0.2] - 2026-03-12

### Changed

- Made the GPT shotgun reliability integration test more efficient by reducing redundant work across attempts.
- Made the GPT shotgun reliability integration test less flaky by improving handling of stochastic live-model responses.

## [1.0.1] - 2026-03-12

### Added

- Added JSONSchemaFormat enum shorthand support for:
	- `[String, ["alpha", "beta", ...]]`
	- `["string", ["alpha", "beta", ...]]`
	- `[String, "description", ["alpha", "beta", ...]]`
	- `["string", "description", ["alpha", "beta", ...]]`

### Fixed

- Preserved tuple disambiguation behavior so `["string", "description"]` remains a type+description tuple, not an enum.
- Preserved existing tuple validation behavior for non-numeric and numeric tuple branches while adding string enum shorthands.

## [1.0.0] - 2026-03-12

### Added

- Initial public release.

