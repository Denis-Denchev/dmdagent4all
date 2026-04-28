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

## Browser

Browser automation is disabled by default.

Rules:

- isolated browser profile
- no personal Chrome profile
- no saved passwords
- downloads only to workspace
- approval required for clicks, form fills, submits, logins, and purchases

## Telegram

Telegram is a remote interface, not a private local channel.

Rules:

- disabled by default
- allowlisted user IDs only
- no public bot mode
- approval buttons for risky actions
- audit every remote request

## WhatsApp

WhatsApp support should follow the same remote-interface rules as Telegram. It is not part of the first backend milestone.
