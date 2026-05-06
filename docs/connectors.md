# Connectors

Connectors are explicit integrations. They are not general computer access.

## Gmail

Default scope target:

```text
gmail.readonly
```

Additional permissions:

- `gmail.compose`
- `gmail.send`
- `gmail.modify`

Rules:

- Read-only tools can run without approval after permission is granted.
- Draft creation is lower risk but still enforced by backend policy.
- Sending always requires approval.
- Delete is disabled by default.

Important: Some Gmail OAuth scopes can technically allow more than one action. Backend policy must enforce the intended product behavior even when a token has broader capability.

Current build status: Gmail manifests and backend handlers are registered, but
real OAuth is not implemented. If enabled and permitted, Gmail tools return
`not_configured` rather than attempting a fake action.

## Calendar

Preferred minimal scope for availability:

```text
calendar.freebusy
```

Additional permissions:

- `calendar.readonly`
- `calendar.events`

Rules:

- Read/free-busy can run without approval after permission is granted.
- Create/update/delete require approval.
- Delete is high risk.

Current build status: Calendar tools use a local workspace JSONL event store for
create, update, delete, today, week, and free-slot queries. Google Calendar OAuth
sync is still a connector task.

## Browser

Browser automation is disabled by default.

Rules:

- isolated browser profile
- no personal Chrome profile
- no saved passwords
- downloads only to workspace
- approval required for clicks, form fills, submits, logins, and purchases

Current build status: `browser.open`, `browser.extract_text`, and
`browser.scrape_markdown` use guarded HTTP reads. Scraped Markdown is saved
locally under `scrapefiles`. `browser.click`, `browser.fill_form`, and
`browser.submit` use an isolated Playwright profile when optional browser
support is installed:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
```

## Telegram

Telegram is a remote interface, not a private local channel.

Rules:

- disabled by default
- allowlisted user IDs only
- no public bot mode
- approval buttons for risky actions
- audit every remote request

Local setup from the terminal chat:

```text
/telegram setup
```

The same panel also supports:

```text
/telegram token <bot_token>
/telegram once
/telegram allow <telegram_user_id>
/telegram enable
/telegram disable
/telegram run
/telegram status
/back
```

Inside terminal chat, `/telegram`, `/help telegram`, or a plain request such as
`set up telegram bot` opens deterministic Telegram setup before any LLM planning
is used.

Operational notes:

- The bot token is read from `DMDAGENT_TELEGRAM_BOT_TOKEN` by default, or from
  the current API process environment when it has been pasted in the dashboard.
- The token is not stored in `config.yaml`, persisted by the dashboard, or sent
  to the model.
- `/id` returns the Telegram user ID needed for the allowlist.
- `/approvals`, `/approve <id>`, and `/deny <id>` work from Telegram for allowlisted users.
- Risky actions still go through the same approval queue as CLI and web UI.
- The dashboard can enable/disable Telegram, configure the token environment
  variable name, load a token into the current API process, and edit the
  allowlisted user IDs.
- `start web` starts Telegram polling in the API process when Telegram is
  enabled and a token is available. Loading a token from the dashboard also
  starts polling automatically once the user ID is allowlisted.

## WhatsApp

WhatsApp support should follow the same remote-interface rules as Telegram. It is not part of the first backend milestone.
