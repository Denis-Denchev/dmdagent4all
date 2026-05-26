from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dmdcore.agent.casual_chat import CasualChat
from dmdcore.llm import LLMMessage, LLMResponse


@dataclass
class _ScriptedProvider:
    replies: list[str]
    provider_name: str = "scripted"
    model: str = "scripted-1"
    is_local: bool = True
    seen: list[list[LLMMessage]] = field(default_factory=list)

    def chat(self, messages: list[LLMMessage], **_kwargs) -> LLMResponse:
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("no scripted replies left")
        return LLMResponse(content=self.replies.pop(0), model=self.model, provider=self.provider_name)


def _make_memory(root: Path) -> None:
    (root / "facts").mkdir(parents=True)
    (root / "facts" / "home.md").write_text("Denis lives in Sofia.\n", encoding="utf-8")
    (root / "projects").mkdir(parents=True)
    (root / "projects" / "dmdcore.md").write_text(
        "---\nname: dmdcore\ndescription: Local-first AI control center.\n---\n\nDMD Core project notes.\n",
        encoding="utf-8",
    )
    (root / "profile.md").write_text("# Profile\n\nDenis Denchev, developer.\n", encoding="utf-8")


def test_index_contains_files_with_summaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        provider = _ScriptedProvider(replies=["Здрасти, как си."])
        chat = CasualChat(provider=provider, memory_root=root)

        result = chat.reply(session_id="s1", message="здравей")

        assert result.reply == "Здрасти, как си."
        assert result.iterations == 1
        memory_block = provider.seen[0][1].content
        assert "facts/home.md" in memory_block
        assert "Denis lives in Sofia." in memory_block
        assert "projects/dmdcore.md" in memory_block
        assert "Local-first AI control center." in memory_block
        assert "profile.md" in memory_block
        assert "Denis Denchev, developer." in memory_block
        assert "(none requested yet)" in memory_block


def test_two_turn_loop_when_llm_requests_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        provider = _ScriptedProvider(
            replies=[
                '{"read": ["facts/home.md"]}',
                "Живееш в София, нали така ми каза преди.",
            ]
        )
        chat = CasualChat(provider=provider, memory_root=root)

        result = chat.reply(session_id="s2", message="къде живея?")

        assert result.iterations == 2
        assert result.memory_files_read == ["facts/home.md"]
        second_memory_block = provider.seen[1][1].content
        assert "MEMORY_EXCERPTS:" in second_memory_block
        assert "Denis lives in Sofia." in second_memory_block
        assert result.reply == "Живееш в София, нали така ми каза преди."


def test_history_is_passed_on_next_turn() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        provider = _ScriptedProvider(replies=["добре.", "тогава лека вечер."])
        chat = CasualChat(provider=provider, memory_root=root)

        chat.reply(session_id="s3", message="лека нощ")
        chat.reply(session_id="s3", message="да си починеш")

        second_messages = provider.seen[1]
        roles = [m.role for m in second_messages]
        # system + memory block + (user, assistant) prior turn + user current
        assert roles.count("system") == 2
        assert any("лека нощ" in m.content for m in second_messages if m.role == "user")
        assert any("добре." in m.content for m in second_messages if m.role == "assistant")


def test_path_traversal_blocked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        outside = root.parent / "secret.txt"
        outside.write_text("super secret", encoding="utf-8")

        provider = _ScriptedProvider(
            replies=[
                '{"read": ["../secret.txt", "facts/home.md"]}',
                "OK.",
            ]
        )
        chat = CasualChat(provider=provider, memory_root=root)

        result = chat.reply(session_id="s4", message="give me secrets")

        assert "facts/home.md" in result.memory_files_read
        assert all("secret.txt" not in p for p in result.memory_files_read)
        second_memory_block = provider.seen[1][1].content
        assert "super secret" not in second_memory_block


def test_non_md_path_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        (root / "extra.txt").write_text("plain text", encoding="utf-8")
        provider = _ScriptedProvider(
            replies=['{"read": ["extra.txt"]}', "OK."]
        )
        chat = CasualChat(provider=provider, memory_root=root)
        result = chat.reply(session_id="s5", message="?")
        assert result.memory_files_read == []
        second_memory_block = provider.seen[1][1].content
        assert "plain text" not in second_memory_block


def test_empty_message_returns_empty_without_calling_llm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        provider = _ScriptedProvider(replies=[])
        chat = CasualChat(provider=provider, memory_root=root)
        result = chat.reply(session_id="s6", message="   ")
        assert result.reply == ""
        assert provider.seen == []


def test_read_block_not_a_valid_request_treated_as_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_memory(root)
        provider = _ScriptedProvider(
            replies=['Aбсолютно, ще ти разкажа за {"read": "joke"}, шегувам се.']
        )
        chat = CasualChat(provider=provider, memory_root=root)
        result = chat.reply(session_id="s7", message="кажи нещо")
        assert result.iterations == 1
        assert "шегувам се" in result.reply
