# Privacy Policy

DMD Agent 4 All is designed as a local-first personal AI control center.

## Defaults

- Local model provider: Ollama
- Cloud models: disabled by default
- Email body to cloud models: disabled by default
- Calendar details to cloud models: disabled by default
- Secret redaction: enabled by default
- Approval required before sending private connector context to a cloud model

## What The Model Must Not See

The model must not receive:

- Gmail refresh tokens
- OpenAI API keys
- Anthropic API keys
- GitHub tokens
- SSH keys
- `.env` files
- raw browser cookies
- system keychains
- OS credential store values

The model receives only sanitized tool results.

## Memory

Local memory is stored as Markdown files under the app data directory:

```text
~/.local/share/dmdagent4all/memory/
  profile.md
  preferences.md
  people/
  projects/
  daily/
  facts/
```

Markdown files are the source of truth. Vector databases, if enabled later, are only derived indexes.

## Remote Chat Interfaces

Telegram and WhatsApp are convenient remote interfaces, but they are not local-private channels. Messages sent through those services pass through external platforms. The product must clearly label this in the UI and settings.

## Cloud Model Approval

If the user enables a cloud model and asks for a private-data action, the product must show an approval prompt before sending sensitive context:

```text
This action may send private connector content to a cloud model.
Allow once / Always allow for this tool / Cancel
```

## Data Location

Default local data path:

```text
~/.local/share/dmdagent4all/
  config.yaml
  memory/
  workspace/
  logs/
  audit.db
  vector_index/
```

Secrets should use the operating system secret store where available:

- macOS: Keychain
- Linux: Secret Service / libsecret
- Windows: Credential Manager

Headless environments may use an encrypted local token store, but tokens must never be stored in plaintext `.env` files.
