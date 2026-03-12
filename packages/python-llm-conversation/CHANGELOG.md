# Changelog

All notable changes to this package will be documented in this file.

## [1.0.1] - 2026-03-12

### Added

- Added JSONSchemaFormat enum shorthand support for:
	- `[str, ["alpha", "beta", ...]]`
	- `["string", ["alpha", "beta", ...]]`
	- `[str, "description", ["alpha", "beta", ...]]`
	- `["string", "description", ["alpha", "beta", ...]]`
- Added Python unit test coverage for the new enum shorthand forms.

### Fixed

- Preserved tuple disambiguation behavior so `["string", "description"]` and `("string", "description")` remain type+description tuples, not enums.
- Preserved existing tuple validation behavior for non-numeric and numeric tuple branches while adding string enum shorthands.

## [1.0.0] - 2026-03-12

### Added

- Initial public release.

