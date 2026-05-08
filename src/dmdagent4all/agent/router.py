from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Literal

from dmdagent4all.permissions import ToolRequest


RouteKind = Literal["normal_chat", "tool_request"]


@dataclass(frozen=True)
class RouteResult:
    kind: RouteKind
    request: ToolRequest | None = None


class ConversationRouter:
    def route(self, text: str) -> RouteResult | None:
        stripped = text.strip()
        if not stripped:
            return RouteResult(kind="normal_chat")

        workspace = _workspace_request(stripped)
        if workspace is not None:
            return RouteResult(kind="tool_request", request=workspace)

        file_request = _file_request(stripped)
        if file_request is not None:
            return RouteResult(kind="tool_request", request=file_request)

        terminal = _terminal_request(stripped)
        if terminal is not None:
            return RouteResult(kind="tool_request", request=terminal)

        if _is_obvious_normal_chat(stripped):
            return RouteResult(kind="normal_chat")
        return None


def _workspace_request(text: str) -> ToolRequest | None:
    normalized = _normalize(text)
    if any(marker in normalized for marker in {"remind", "notify", "напомни", "напомниш"}):
        return None
    cd_match = re.search(
        r"^(?:cd|chdir|go\s+into|enter|влез\s+в|иди\s+в)\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if cd_match:
        return _workspace_switch_request(_clean_path(cd_match.group("path")))
    terminal_cd_match = re.search(
        r"^(?:(?:in\s+my|in)\s+terminal\s+)?(?:execute|exeute|run|изпълни|пусни)"
        r"(?:\s+(?:in\s+my|in)\s+terminal|\s+(?:в|във)\s+(?:терминала|terminal))?"
        r"\s+cd\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if terminal_cd_match:
        return _workspace_switch_request(_clean_path(terminal_cd_match.group("path")))
    open_match = re.search(
        r"^(?:open|отвори|отвори\s+папка|switch\s+to|go\s+to|смени\s+към)\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if open_match:
        path = _clean_path(open_match.group("path"))
        if "terminal" in path.casefold() or "терминал" in path.casefold():
            return None
        if _looks_local_folder_target(path):
            return _workspace_switch_request(path)
    match = re.search(
        r"(?:work\s+in|switch\s+workspace\s+to|change\s+workspace\s+to|workspace\s+to|"
        r"работи\s+в|смени\s+workspace\s+(?:на|към)|смени\s+работната\s+папка\s+(?:на|към))"
        r"\s+(?P<path>~?/?[^\n]+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        if normalized in {"workspace", "current workspace", "кой workspace", "текущ workspace"}:
            return ToolRequest(
                tool="workspace.status",
                args={},
                reason="User asked for current workspace status.",
            )
        return None
    path = _clean_path(match.group("path"))
    return _workspace_switch_request(path)


def _workspace_switch_request(path: str) -> ToolRequest | None:
    if not path:
        return None
    return ToolRequest(
        tool="workspace.switch",
        args={"path": path},
        reason="User asked to switch the active workspace.",
    )


def _file_request(text: str) -> ToolRequest | None:
    if _looks_email_text(text):
        return None
    delete_match = re.search(
        r"^(?:delete|remove|rm|изтрий|премахни)\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if delete_match:
        path = _clean_delete_path(delete_match.group("path"))
        return ToolRequest(
            tool="files.delete",
            args={"path": path, "recursive": _looks_directory_delete(text)},
            reason="User asked to delete a local file or directory.",
        )

    read_match = re.search(
        r"^(?:read|show|print|cat|прочети|покажи|принтирай)\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if read_match:
        path = _clean_path(read_match.group("path"))
        if "memory" in path.casefold():
            return None
        if path:
            return ToolRequest(
                tool="files.read",
                args={"path": path},
                reason="User asked to read a local file.",
            )
    return None


def _terminal_request(text: str) -> ToolRequest | None:
    command_text = _extract_terminal_command_text(text)
    if command_text is None:
        return None
    try:
        command = shlex.split(command_text)
    except ValueError:
        return None
    if not command:
        return None
    if command[0] not in {"ls", "pwd", "git", "cat", "mkdir", "docker", "pytest", "npm", "psql", "sqlite3"}:
        return None
    return ToolRequest(
        tool="terminal.run",
        args={"command": command},
        reason="User asked to run a terminal command.",
    )


def _extract_terminal_command_text(text: str) -> str | None:
    normalized = _normalize(text)
    direct_prefix = re.match(
        r"^(?:изпълни|пусни|стартирай|run|execute|exeute)(?:\s+(?:в|in)\s+(?:терминала|terminal))?\s+(?P<command>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if direct_prefix:
        return direct_prefix.group("command").strip()
    if normalized.startswith(("ls", "pwd", "git status", "git diff", "cat ")):
        return text.strip()
    return None


def _is_obvious_normal_chat(text: str) -> bool:
    normalized = _normalize(text)
    if normalized in {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "благодаря",
        "благодаря ти",
        "мерси",
        "здравей",
        "здрасти",
        "джарвис как си",
        "как си",
    }:
        return True
    if any(marker in normalized for marker in {"преведи", "преведеш", "translate"}):
        return True
    return False


def _looks_directory_delete(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in {"directory", "folder", "директория", "папка", "-rf", "-r"})


def _clean_path(value: str) -> str:
    cleaned = value.strip().strip(" ,!?:;\"'")
    for prefix in {
        "this ",
        "the ",
        "този ",
        "тази ",
        "това ",
        "file ",
        "файл ",
        "path ",
        "път ",
        "folder ",
        "directory ",
        "папка ",
        "директория ",
    }:
        if cleaned.casefold().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.strip(" ,!?:;\"'")


def _clean_delete_path(value: str) -> str:
    cleaned = _clean_path(value)
    trailing_folder_match = re.match(
        r"^(?P<name>[A-Za-z0-9_.\-/]+)\s+(?:folder|directory|папка|директория)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if trailing_folder_match:
        return trailing_folder_match.group("name").strip()
    return cleaned


def _looks_local_folder_target(path: str) -> bool:
    normalized = path.casefold().strip()
    if not normalized or normalized in {"google", "гугъл"}:
        return False
    if re.search(r"https?://", normalized):
        return False
    if re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/.*)?", normalized, flags=re.IGNORECASE):
        return False
    if normalized.endswith((".md", ".txt", ".json", ".py", ".js", ".ts", ".tsx", ".html", ".css")):
        return False
    return True


def _looks_email_text(text: str) -> bool:
    normalized = _normalize(text)
    return any(
        marker in normalized
        for marker in {
            "email",
            "e-mail",
            "mail",
            "gmail",
            "outlook",
            "имейл",
            "мейл",
            "мейла",
            "поща",
        }
    )


def _normalize(text: str) -> str:
    return text.casefold().strip().strip(" .,!?:;")
