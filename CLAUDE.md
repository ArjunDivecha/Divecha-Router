## Cross-session messaging

Claude Code sessions can message each other directly. `ListAgents` (or `/list-agents`, `/peers`)
lists reachable sessions; `SendMessage` delivers plain text to one by name. Same-machine delivery
uses a local socket; cross-machine is reply-only via Remote Control. Use it to hand off a finding
to a session working elsewhere instead of relaying it through the user. A message is text only —
never conversation history or files; to share full context, resume the session instead.
