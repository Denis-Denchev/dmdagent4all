# Contributing

This project is security-sensitive. Contributions must preserve the core rule: the model is not trusted for security.

## Development Setup

```bash
./scripts/install.sh
```

Run tests:

```bash
PYTHONPATH=src python -m unittest
```

Run syntax checks:

```bash
python -m compileall src tests
```

## Contribution Rules

- Do not add plaintext token storage.
- Do not expose raw secrets to model context.
- Do not add direct shell execution.
- Do not enable high-risk tools by default.
- Add or update tests for permission changes.
- Keep install, CLI, API, and UI text in English.
