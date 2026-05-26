# Connectors

Connectors are explicit integrations. They are not general computer access.

## Gmail And Outlook

Email connectors use IMAP/SMTP in the current build. Secrets are read only from
environment variables in the running API process; they are not stored in
`config.yaml` or sent to the model.

The dashboard exposes these settings under Config -> Email Connectors. Use it
to enable Gmail/Outlook, adjust mailbox/env-var names, and load username plus
app password into the current API process. Credentials loaded from the UI are
process-local; after a backend or container restart, load them again or provide
the same variables through the shell/Docker environment.

Loading credentials from the dashboard also enables the selected provider and
its email tools. Sending still requires approval before SMTP delivery.
Use **Test Gmail Connection** or **Test Outlook Connection** after loading
credentials. The test logs into IMAP, opens the configured mailbox, then logs
into SMTP without sending a message.

Gmail env vars:

```text
DMDCORE_GMAIL_USERNAME
DMDCORE_GMAIL_APP_PASSWORD
DMDCORE_GMAIL_FROM optional
```

Outlook env vars:

```text
DMDCORE_OUTLOOK_USERNAME
DMDCORE_OUTLOOK_APP_PASSWORD
DMDCORE_OUTLOOK_FROM optional
```

Enable provider config:

```yaml
email:
  gmail:
    enabled: true
  outlook:
    enabled: true
```

Permissions:

- `gmail.readonly` / `outlook.readonly`
- `gmail.compose` / `outlook.compose`
- `gmail.send` / `outlook.send`
- `gmail.modify` / `outlook.modify`

Tools:

- search messages
- read one message/thread by IMAP uid
- summarize inbox/search results
- create local drafts
- create local reply drafts
- send local drafts
- archive messages
- Gmail label copy

Rules:

- Read-only tools can run without approval after permission is granted.
- Draft creation is lower risk but still enforced by backend policy.
- Sending always requires approval.
- Archive/label/modify actions require approval.
- Delete is not implemented.

Important: Gmail and Outlook account settings may require app passwords or
provider-side IMAP/SMTP enablement. Microsoft 365 tenants may disable basic
SMTP/IMAP auth; in that case this connector must later be replaced with OAuth /
Microsoft Graph for that account.
When Outlook returns `5.7.139 Authentication unsuccessful, basic authentication
is disabled`, the message was not sent. Use Gmail SMTP for the current build or
add/configure Microsoft Graph OAuth for Outlook.

Current build status: Gmail and Outlook have real IMAP/SMTP handlers. If the
provider is disabled or required env vars are missing, tools fail closed with
`connector_not_configured`.

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

- The bot token is read from `DMDCORE_TELEGRAM_BOT_TOKEN` by default, or from
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
