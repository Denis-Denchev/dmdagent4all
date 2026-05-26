"""Casual chat mode (LLM-wiki pattern).

A fast, no-tools companion. The LLM has read-only access to the user's
markdown memory through a wiki-style index + on-demand file fetch.

Flow per user turn:
  1. Build a memory index (relative path + first non-empty line as summary).
  2. Send {system + index + conversation history + user message} to the LLM.
  3. If the LLM reply starts with a JSON block requesting files (READ: [...]),
     fetch them, append as memory excerpts, and call the LLM once more for the
     final answer. Maximum two iterations to keep latency low.
  4. Return the plain-text reply plus the list of memory files consulted.

The endpoint is independent of the agent pipeline: no planner, no permissions,
no tools, no approvals. The provider is the same one configured for the agent.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from dmdcore.llm import LLMMessage, LLMProvider


SYSTEM_PROMPT = """You are the user's casual chat companion inside the DMD Core app.

You are NOT an agent. You have no tools, you take no actions, you never call APIs, files, or commands. You only talk.

You have read-only access to the user's personal memory as a wiki of markdown notes. Below this prompt the user (the system) injects:

- MEMORY_INDEX — every memory file with a short one-line description
- MEMORY_EXCERPTS — optional full text of files you asked to read on the previous turn

How to use memory:
- Skim the index before answering. If a file's description looks relevant to the user's message, request its full content by replying ONLY with a single JSON line:
    {"read": ["relative/path/one.md", "relative/path/two.md"]}
  Nothing else on that line, no prose, no code fences.
- The system will then send you the requested files in MEMORY_EXCERPTS and ask you again. At that point answer the user normally, in plain text, using what you read.
- You may request files at most once per user turn. If you do not need files, answer the user directly.
- Reference memory naturally ("ти ми каза, че…", "имам бележка за…") — do not dump file paths or markdown structure at the user.

Style:
- Match the user's language (Bulgarian or English). The user is Denis.
- Be warm, conversational, concise. Like a friend who happens to remember everything.
- No emoji. No markdown headings. Short paragraphs. Plain prose.
- If the user is just chatting (greeting, small talk, jokes), reply naturally without searching memory.
- If you do not know something and memory does not have it, say so plainly. Do not invent facts.

Refusals:
- This is casual chat. You do not execute, send, install, write code, modify files, or do anything that requires real-world actions. If the user asks for that, briefly point them to Agent mode and stay in chat.
"""

_READ_RE = re.compile(r"^\s*\{\s*\"read\"\s*:\s*\[[^\]]*\]\s*\}\s*$", re.DOTALL)


@dataclass
class CasualReply:
    reply: str
    memory_files_read: list[str] = field(default_factory=list)
    iterations: int = 1


@dataclass
class _Session:
    history: deque[LLMMessage]


class CasualChat:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        memory_root: Path,
        max_history_messages: int = 20,
        max_file_bytes: int = 12_000,
    ) -> None:
        self._provider = provider
        self._memory_root = memory_root
        self._max_history = max_history_messages
        self._max_file_bytes = max_file_bytes
        self._sessions: dict[str, _Session] = {}
        self._lock = RLock()

    def reply(self, *, session_id: str, message: str) -> CasualReply:
        message = message.strip()
        if not message:
            return CasualReply(reply="")
        session = self._get_session(session_id)
        index_text = self._build_index_text()
        system_message = LLMMessage(role="system", content=SYSTEM_PROMPT)

        def call(memory_block: str) -> str:
            messages: list[LLMMessage] = [system_message]
            messages.append(LLMMessage(role="system", content=memory_block))
            messages.extend(session.history)
            messages.append(LLMMessage(role="user", content=message))
            response = self._provider.chat(messages, temperature=0.5)
            return response.content.strip()

        memory_block = self._wrap_memory_block(index_text, excerpts=None)
        first = call(memory_block)
        requested = _parse_read_request(first)
        memory_files_read: list[str] = []
        iterations = 1

        if requested:
            excerpts, read_paths = self._load_excerpts(requested)
            memory_files_read = read_paths
            memory_block = self._wrap_memory_block(index_text, excerpts=excerpts)
            final = call(memory_block)
            iterations = 2
        else:
            final = first

        self._append_history(session, message, final)
        return CasualReply(reply=final, memory_files_read=memory_files_read, iterations=iterations)

    def _get_session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = _Session(history=deque(maxlen=self._max_history))
                self._sessions[session_id] = session
            return session

    def _append_history(self, session: _Session, user_msg: str, assistant_msg: str) -> None:
        with self._lock:
            session.history.append(LLMMessage(role="user", content=user_msg))
            session.history.append(LLMMessage(role="assistant", content=assistant_msg))

    def _build_index_text(self) -> str:
        entries = list(_iter_memory_files(self._memory_root))
        if not entries:
            return "MEMORY_INDEX: (empty — no memory files)"
        lines = ["MEMORY_INDEX:"]
        for relative, summary in entries:
            line = f"- {relative} — {summary}" if summary else f"- {relative}"
            lines.append(line)
        return "\n".join(lines)

    def _wrap_memory_block(self, index_text: str, *, excerpts: list[str] | None) -> str:
        parts = [index_text]
        if excerpts:
            parts.append("\nMEMORY_EXCERPTS:")
            parts.extend(excerpts)
        else:
            parts.append("\nMEMORY_EXCERPTS: (none requested yet)")
        return "\n".join(parts)

    def _load_excerpts(self, relative_paths: list[str]) -> tuple[list[str], list[str]]:
        root_resolved = self._memory_root.resolve()
        excerpts: list[str] = []
        loaded: list[str] = []
        for rel in relative_paths:
            path = self._safe_resolve(rel, root_resolved)
            if path is None:
                excerpts.append(f"\n--- {rel} ---\n(not found)\n")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                excerpts.append(f"\n--- {rel} ---\n(unreadable: {exc})\n")
                continue
            if len(text) > self._max_file_bytes:
                text = text[: self._max_file_bytes] + "\n[truncated]"
            relative = path.relative_to(root_resolved).as_posix()
            excerpts.append(f"\n--- {relative} ---\n{text}\n")
            loaded.append(relative)
        return excerpts, loaded

    def _safe_resolve(self, rel: str, root_resolved: Path) -> Path | None:
        rel = rel.strip().lstrip("/").lstrip("./")
        if not rel:
            return None
        candidate = (self._memory_root / rel).resolve()
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            return None
        if not candidate.is_file() or candidate.suffix.lower() != ".md":
            return None
        return candidate


def _iter_memory_files(root: Path) -> list[tuple[str, str]]:
    if not root.exists():
        return []
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in path.parts):
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        summary = _summary_for(path)
        rows.append((relative, summary))
    return rows


def _summary_for(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = handle.read(2048)
    except OSError:
        return ""
    lines = content.splitlines()
    if not lines:
        return ""
    in_frontmatter = False
    description_from_frontmatter = ""
    for raw in lines:
        line = raw.strip()
        if line == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            lowered = line.lower()
            if lowered.startswith("description:"):
                description_from_frontmatter = line.split(":", 1)[1].strip().strip('"').strip("'")
                if description_from_frontmatter:
                    return description_from_frontmatter[:160]
            continue
        if not line or line.startswith("#"):
            continue
        return line[:160]
    return description_from_frontmatter[:160]


def _parse_read_request(text: str) -> list[str] | None:
    if not text or "{" not in text:
        return None
    first_line = text.strip().splitlines()[0].strip()
    if not _READ_RE.match(first_line):
        return None
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    paths = payload.get("read")
    if not isinstance(paths, list):
        return None
    cleaned: list[str] = []
    for item in paths:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
        if len(cleaned) >= 8:
            break
    return cleaned or None
