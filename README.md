# gpt-conversation

`gpt-conversation` is a cross-language toolkit for building reliable LLM features with OpenAI's Responses API.

The project exists to remove repeated integration work from application teams. Instead of re-implementing the same conversation and structured-output patterns in every codebase, this repository provides a shared, production-oriented foundation for Python and TypeScript.

## Purpose

- Keep LLM integration practical and predictable in real products.
- Preserve behavioral parity across Python and TypeScript implementations.
- Emphasize reusable building blocks rather than framework lock-in.

## Scope

This repository focuses on the parts of LLM development that are easy to get wrong repeatedly:

- Managing multi-message conversations over time.
- Working with structured JSON outputs safely.
- Keeping shared semantics aligned across languages.

This repository is intentionally not an agent framework, orchestration platform, or full application starter.

## Repository Layout

- `packages/python-gpt-conversation`: Python package implementation.
- `packages/typescript-gpt-conversation`: TypeScript package implementation.

Both packages follow the same product intent and are developed together to maintain consistency.

## Design Principles

- Minimal, composable abstractions.
- Explicit behavior and stable contracts.
- Production-first defaults.
- Language parity where it matters most.

## API Documentation

The root README describes intent and project-level context.

For package usage, API reference, and language-specific setup, see:

- `packages/python-gpt-conversation/README.md`
- `packages/typescript-gpt-conversation/README.md`

## Quality and Reliability

The project prioritizes practical correctness over synthetic demos. Package-level test suites are designed to protect real integration behavior, including structured-output workflows.

## Releases

Python and TypeScript packages are versioned and released from this monorepo using repository automation. Release mechanics are kept with each package and workflow configuration.

## License

MIT. See `LICENSE`.
