#!/usr/bin/env python3
"""
🤖 AI Chat — терминальный чат-клиент для LLM (OpenAI-совместимый API).

Возможности:
- Стриминг ответов в реальном времени (rich.live.Live)
- Markdown-рендеринг (переключаемый на лету)
- Поддержка thinking-моделей (reasoning_content)
- Персистентная история ввода и автозавершение (prompt_toolkit)
- История сессий, глобальная статистика токенов и стоимости
- Многострочный ввод (Ctrl+Enter — отправка)
- Неинтерактивный режим (--output) для скриптов/пайпов
"""

# ============================================================================
# 1. ИМПОРТЫ
# ============================================================================
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, TypedDict

# Внешние зависимости
try:
    from openai import OpenAI
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        RateLimitError,
    )
except ImportError:  # pragma: no cover
    print("Не установлена библиотека 'openai'. Выполните: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover
    print("Не установлена библиотека 'rich'. Выполните: pip install rich", file=sys.stderr)
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import get_app
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import Completer, Completion, merge_completers
    from prompt_toolkit.formatted_text import ANSI
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:  # pragma: no cover
    print(
        "Не установлена библиотека 'prompt_toolkit'. Выполните: pip install prompt_toolkit",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================================
# 2. КОНСТАНТЫ
# ============================================================================
VERSION = "1.0.0"

# Пути (всё рядом с main.py)
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
HISTORY_DIR = BASE_DIR / "history"
INPUT_HISTORY_PATH = BASE_DIR / ".input_history"
USAGE_STATS_PATH = BASE_DIR / "usage_stats.json"

# Лимиты
MAX_FILE_SIZE = 100 * 1024  # 100 KB для /file и --input
MAX_PASTE_PREVIEW_LINES = 30  # лимит строк при предпросмотре вставки
CONTEXT_WARN_THRESHOLD = 0.80  # порог предупреждения о контексте (80%)
MAX_RETRIES = 3  # количество повторных попыток при сетевых ошибках

# Шаблон конфига по умолчанию (создаётся при первом запуске)
DEFAULT_CONFIG: dict[str, Any] = {
    "api_key": "sk-YOUR-KEY-HERE",
    "api_base": "https://api.openai.com/v1",
    "default_model": "gpt-4o",
    "system_prompt": "Ты полезный ассистент.",
    "render_markdown": False,
    "generation": {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4096,
    },
    "models": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "context_window": 128000,
            "input_price_per_1m_tokens_rub": 150.0,
            "output_price_per_1m_tokens_rub": 600.0,
        },
        {
            "id": "deepseek-reasoner",
            "name": "DeepSeek R1",
            "context_window": 65536,
            "supports_thinking": True,
            "input_price_per_1m_tokens_rub": 3.0,
            "output_price_per_1m_tokens_rub": 15.0,
        },
    ],
}

# Список команд для автозавершения
COMMANDS = [
    "/new",
    "/history",
    "/model",
    "/markdown",
    "/md",
    "/file",
    "/clear",
    "/usage",
    "/help",
    "/exit",
]

# Аргументы команд для автозавершения
COMMAND_ARGS: dict[str, list[str]] = {
    "/new": ["--clear"],
    "/history": ["all"],
    "/model": ["--default", "--provider"],
    "/markdown": ["on", "off", "toggle"],
    "/md": ["on", "off", "toggle"],
}


# ============================================================================
# 3. ТИПЫ ДАННЫХ
# ============================================================================
class Usage(TypedDict, total=False):
    """Статистика использования токенов одного ответа."""

    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int


@dataclass
class Message:
    """Одно сообщение в диалоге."""

    role: str  # "system" | "user" | "assistant"
    content: str
    timestamp: str = ""
    model: Optional[str] = None  # модель, сгенерировавшая ответ (для assistant)
    thinking: Optional[str] = None  # размышления (reasoning_content), только для отображения
    usage: Optional[dict[str, int]] = None  # токены ответа
    interrupted: bool = False  # был ли ответ прерван (Ctrl+C)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")

    def to_api_dict(self) -> dict[str, str]:
        """Возвращает словарь для отправки в API (без thinking/usage/прочего)."""
        return {"role": self.role, "content": self.content}

    def to_json_dict(self) -> dict[str, Any]:
        """Возвращает словарь для сохранения в JSON-файл сессии."""
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.model:
            data["model"] = self.model
        if self.thinking:
            data["thinking"] = self.thinking
        if self.usage:
            data["usage"] = self.usage
        if self.interrupted:
            data["interrupted"] = True
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "Message":
        """Восстанавливает сообщение из словаря JSON."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", ""),
            model=data.get("model"),
            thinking=data.get("thinking"),
            usage=data.get("usage"),
            interrupted=data.get("interrupted", False),
        )


@dataclass
class Session:
    """Сессия диалога с моделью."""

    session_id: str
    created_at: str
    default_model: str
    messages: list[Message] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Суммарное количество токенов сессии (prompt + completion)."""
        total = 0
        for msg in self.messages:
            if msg.usage:
                total += msg.usage.get("prompt_tokens", 0)
                total += msg.usage.get("completion_tokens", 0)
        return total

    def first_user_message(self) -> str:
        """Возвращает текст первого сообщения пользователя (или пустую строку)."""
        for msg in self.messages:
            if msg.role == "user":
                return msg.content
        return ""

    def to_json_dict(self) -> dict[str, Any]:
        """Сериализует сессию в словарь для сохранения."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "default_model": self.default_model,
            "total_tokens": self.total_tokens,
            "messages": [m.to_json_dict() for m in self.messages],
        }


@dataclass
class ModelInfo:
    """Информация о модели из конфига."""

    id: str
    name: str
    context_window: int = 128000
    supports_thinking: bool = False
    input_price_per_1m_tokens_rub: Optional[float] = None
    output_price_per_1m_tokens_rub: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelInfo":
        """Создаёт ModelInfo из словаря конфига с применением дефолтов."""
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            context_window=data.get("context_window", 128000),
            supports_thinking=data.get("supports_thinking", False),
            input_price_per_1m_tokens_rub=data.get("input_price_per_1m_tokens_rub"),
            output_price_per_1m_tokens_rub=data.get("output_price_per_1m_tokens_rub"),
        )

    def has_pricing(self) -> bool:
        """Заданы ли цены для расчёта стоимости."""
        return (
            self.input_price_per_1m_tokens_rub is not None
            and self.output_price_per_1m_tokens_rub is not None
        )


class ConfigError(Exception):
    """Ошибка валидации конфигурации."""


# ============================================================================
# 4. КЛАСС Config — загрузка, валидация, дефолты, сохранение default_model
# ============================================================================
class Config:
    """
    Конфигурация приложения.

    Загружает config.json, валидирует обязательные поля и типы,
    подставляет дефолты для необязательных полей, умеет обновлять
    default_model с сохранением форматирования файла.
    """

    REQUIRED_FIELDS = ("api_key", "api_base", "default_model")

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        self.api_key: str = raw["api_key"]
        self.api_base: str = raw["api_base"]
        self.default_model: str = raw["default_model"]
        self.system_prompt: str = raw.get("system_prompt", "Ты полезный ассистент.")
        self.render_markdown: bool = raw.get("render_markdown", False)

        gen = raw.get("generation", {})
        self.temperature: float = gen.get("temperature", 0.7)
        self.top_p: float = gen.get("top_p", 1.0)
        self.max_tokens: int = gen.get("max_tokens", 4096)

        self.models: list[ModelInfo] = [
            ModelInfo.from_dict(m) for m in raw.get("models", [])
        ]

    # ---- Поиск моделей ------------------------------------------------------
    def find_model(self, model_id: str) -> Optional[ModelInfo]:
        """Возвращает ModelInfo по точному id, либо None."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def search_models(self, query: str) -> list[ModelInfo]:
        """
        Ищет модели по id или name (case-insensitive, частичное совпадение).
        Если query — число, трактует как индекс (1-based) в списке моделей.
        """
        query = query.strip()
        # Поиск по номеру
        if query.isdigit():
            idx = int(query) - 1
            if 0 <= idx < len(self.models):
                return [self.models[idx]]
            return []
        # Поиск по подстроке
        q = query.lower()
        # Сначала пытаемся найти точное совпадение
        exact = [m for m in self.models if m.id.lower() == q or m.name.lower() == q]
        if exact:
            return exact
        return [m for m in self.models if q in m.id.lower() or q in m.name.lower()]

    # ---- Сохранение default_model -------------------------------------------
    def set_default_model(self, model_id: str) -> None:
        """
        Записывает новый default_model в config.json, сохраняя остальные
        поля и форматирование (отступы 2 пробела, как в шаблоне).
        """
        self.default_model = model_id
        self._raw["default_model"] = model_id
        CONFIG_PATH.write_text(
            json.dumps(self._raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- Фабрика ------------------------------------------------------------
    @classmethod
    def load(cls) -> "Config":
        """
        Загружает и валидирует конфиг.

        Если файл отсутствует — создаёт шаблон и завершает работу (код 0).
        При ошибках валидации поднимает ConfigError.
        """
        if not CONFIG_PATH.exists():
            cls._create_template()
            print(
                f"Конфиг создан: {CONFIG_PATH}. Укажите api_key и запустите снова."
            )
            sys.exit(0)

        # Парсинг JSON с указанием строки ошибки
        text = CONFIG_PATH.read_text(encoding="utf-8")
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Ошибка парсинга config.json (строка {e.lineno}): {e.msg}"
            ) from e

        cls._validate(raw)
        return cls(raw)

    @staticmethod
    def _create_template() -> None:
        """Создаёт файл config.json с дефолтными значениями."""
        CONFIG_PATH.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        """Проверяет наличие обязательных полей и корректность типов."""
        # Обязательные поля
        for field_name in Config.REQUIRED_FIELDS:
            if field_name not in raw:
                raise ConfigError(
                    f'Ошибка конфига: отсутствует обязательное поле "{field_name}"'
                )

        # Типы обязательных полей
        for field_name in Config.REQUIRED_FIELDS:
            if not isinstance(raw[field_name], str):
                raise ConfigError(
                    f'Ошибка конфига: поле "{field_name}" должно быть строкой, '
                    f'получено "{raw[field_name]}"'
                )

        # Типы необязательных полей (если присутствуют)
        if "render_markdown" in raw and not isinstance(raw["render_markdown"], bool):
            raise ConfigError(
                f'Ошибка конфига: поле "render_markdown" должно быть булевым, '
                f'получено "{raw["render_markdown"]}"'
            )

        if "system_prompt" in raw and not isinstance(raw["system_prompt"], str):
            raise ConfigError(
                f'Ошибка конфига: поле "system_prompt" должно быть строкой, '
                f'получено "{raw["system_prompt"]}"'
            )

        gen = raw.get("generation", {})
        if not isinstance(gen, dict):
            raise ConfigError(
                'Ошибка конфига: поле "generation" должно быть объектом'
            )
        if "temperature" in gen and not isinstance(gen["temperature"], (int, float)):
            raise ConfigError(
                f'Ошибка конфига: поле "temperature" должно быть числом, '
                f'получено "{gen["temperature"]}"'
            )
        if "top_p" in gen and not isinstance(gen["top_p"], (int, float)):
            raise ConfigError(
                f'Ошибка конфига: поле "top_p" должно быть числом, '
                f'получено "{gen["top_p"]}"'
            )
        if "max_tokens" in gen and not isinstance(gen["max_tokens"], int):
            raise ConfigError(
                f'Ошибка конфига: поле "max_tokens" должно быть целым числом, '
                f'получено "{gen["max_tokens"]}"'
            )

        if "models" in raw and not isinstance(raw["models"], list):
            raise ConfigError(
                'Ошибка конфига: поле "models" должно быть списком'
            )


# ============================================================================
# 5. КЛАСС HistoryManager — сохранение/загрузка сессий
# ============================================================================
class HistoryManager:
    """
    Управление историей сессий в каталоге ./history/.

    Каждая сессия — отдельный JSON-файл с именем YYYYMMDD_HHMMSS.json.
    """

    def __init__(self) -> None:
        HISTORY_DIR.mkdir(exist_ok=True)

    def save(self, session: Session) -> None:
        """Сохраняет сессию в файл (перезаписывает существующий)."""
        # Не сохраняем пустые сессии (только system или вообще ничего)
        non_system = [m for m in session.messages if m.role != "system"]
        if not non_system:
            return
        path = HISTORY_DIR / f"{session.session_id}.json"
        path.write_text(
            json.dumps(session.to_json_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, session_id: str) -> Optional[Session]:
        """Загружает сессию по session_id (имя файла без .json)."""
        path = HISTORY_DIR / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        messages = [Message.from_json_dict(m) for m in data.get("messages", [])]
        return Session(
            session_id=data.get("session_id", ""),
            created_at=data.get("created_at", ""),
            default_model=data.get("default_model", ""),
            messages=messages,
        )

    def list_sessions(self, limit: Optional[int] = 20) -> list[Session]:
        """
        Возвращает список сессий, отсортированных по дате (новые первыми).
        Если limit задан — возвращает только последние limit штук.
        """
        files = sorted(
            HISTORY_DIR.glob("*.json"),
            key=lambda p: p.stem,
            reverse=True,
        )
        sessions: list[Session] = []
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            messages = [Message.from_json_dict(m) for m in data.get("messages", [])]
            sessions.append(
                Session(
                    session_id=data.get("session_id", ""),
                    created_at=data.get("created_at", ""),
                    default_model=data.get("default_model", ""),
                    messages=messages,
                )
            )
        if limit is not None:
            return sessions[:limit]
        return sessions


# ============================================================================
# 6. КЛАСС UsageTracker — глобальная статистика токенов
# ============================================================================
class UsageTracker:
    """
    Чтение и обновление накопительной статистики usage_stats.json
    (суммарные токены, стоимость и число запросов за всё время).
    """

    def __init__(self) -> None:
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        """Загружает статистику из файла или возвращает пустую структуру."""
        if not USAGE_STATS_PATH.exists():
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_rub": 0.0,
                "total_requests": 0,
                "updated_at": "",
            }
        try:
            return json.loads(USAGE_STATS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_cost_rub": 0.0,
                "total_requests": 0,
                "updated_at": "",
            }

    def add(self, prompt_tokens: int, completion_tokens: int, cost_rub: float) -> None:
        """Добавляет данные одного запроса и сохраняет файл."""
        self.data["total_prompt_tokens"] = (
            self.data.get("total_prompt_tokens", 0) + prompt_tokens
        )
        self.data["total_completion_tokens"] = (
            self.data.get("total_completion_tokens", 0) + completion_tokens
        )
        self.data["total_cost_rub"] = round(
            self.data.get("total_cost_rub", 0.0) + cost_rub, 4
        )
        self.data["total_requests"] = self.data.get("total_requests", 0) + 1
        self.data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self._save()

    def _save(self) -> None:
        """Записывает статистику в файл."""
        USAGE_STATS_PATH.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ============================================================================
# 7. КЛАСС ChatClient — API-вызовы, стриминг, подсчёт токенов
# ============================================================================
@dataclass
class StreamResult:
    """Результат стриминга ответа."""

    content: str
    thinking: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    elapsed: float
    interrupted: bool = False


class ChatClient:
    """
    Клиент для общения с OpenAI-совместимым API.

    Отвечает за стриминг ответов, обработку reasoning_content,
    подсчёт токенов и расчёт стоимости.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def update_provider(self, api_base: str, api_key: Optional[str] = None) -> None:
        """Меняет провайдера (api_base/api_key) и пересоздаёт клиент."""
        self.config.api_base = api_base
        if api_key:
            self.config.api_key = api_key
        self.client = OpenAI(api_key=self.config.api_key, base_url=self.config.api_base)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Грубая оценка количества токенов (fallback, если API не вернул usage).
        Для текста с кириллицей — len/2, иначе len/4.
        """
        if not text:
            return 0
        has_cyrillic = any("\u0400" <= ch <= "\u04ff" for ch in text)
        divisor = 2 if has_cyrillic else 4
        return max(1, len(text) // divisor)

    def calc_cost(
        self, model: Optional[ModelInfo], prompt_tokens: int, completion_tokens: int
    ) -> Optional[float]:
        """
        Рассчитывает стоимость в рублях по ценам модели.
        Возвращает None, если у модели нет цен (модель не в списке).
        """
        if model is None or not model.has_pricing():
            return None
        cost = (
            prompt_tokens * model.input_price_per_1m_tokens_rub / 1_000_000
            + completion_tokens * model.output_price_per_1m_tokens_rub / 1_000_000
        )
        return cost

    def stream(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        on_chunk: "Optional[callable]" = None,
    ) -> StreamResult:
        """
        Выполняет стриминговый запрос к API.

        on_chunk(content_so_far, thinking_so_far, completion_tokens) —
        колбэк, вызываемый на каждый chunk для обновления Live.

        Возвращает StreamResult. Обрабатывает прерывание Ctrl+C.
        """
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        interrupted = False
        start = time.monotonic()

        # Пытаемся включить include_usage (поддерживается не всеми API)
        create_kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        try:
            create_kwargs["stream_options"] = {"include_usage": True}
            stream = self.client.chat.completions.create(**create_kwargs)
        except TypeError:
            # API не поддерживает stream_options — убираем и пробуем снова
            create_kwargs.pop("stream_options", None)
            stream = self.client.chat.completions.create(**create_kwargs)

        try:
            for chunk in stream:
                # Usage может прийти в последнем chunk'е
                usage = getattr(chunk, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                    # reasoning-токены (если API сообщает)
                    details = getattr(usage, "completion_tokens_details", None)
                    if details:
                        reasoning_tokens = (
                            getattr(details, "reasoning_tokens", 0) or 0
                        )

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # reasoning_content — размышления (thinking)
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    thinking_parts.append(reasoning)

                # Обычный контент ответа
                if delta.content:
                    content_parts.append(delta.content)

                # Колбэк для обновления Live
                if on_chunk is not None:
                    cur_completion = (
                        completion_tokens
                        if completion_tokens
                        else self.estimate_tokens("".join(content_parts))
                    )
                    on_chunk(
                        "".join(content_parts),
                        "".join(thinking_parts),
                        cur_completion,
                    )
        except KeyboardInterrupt:
            # Прерывание во время генерации — сохраняем фрагмент
            interrupted = True
            try:
                stream.close()
            except Exception:
                pass

        elapsed = time.monotonic() - start
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)

        # Fallback-подсчёт токенов, если API не вернул usage
        if prompt_tokens == 0:
            prompt_tokens = sum(
                self.estimate_tokens(m["content"]) for m in messages
            )
        if completion_tokens == 0:
            completion_tokens = self.estimate_tokens(content)

        return StreamResult(
            content=content,
            thinking=thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            elapsed=elapsed,
            interrupted=interrupted,
        )


# ============================================================================
# 8. КЛАСС ChatUI — интерфейс (rich + prompt_toolkit)
# ============================================================================
class CommandCompleter(Completer):
    """
    Кастомный completer для команд и их аргументов.
    Дополняет ввод, начинающийся с '/'.
    """

    def get_completions(self, document, complete_event):  # type: ignore[override]
        text = document.text_before_cursor
        # Автодополняем только команды (строки, начинающиеся с /)
        if not text.startswith("/"):
            return

        parts = text.split()
        # Дополнение имени команды
        if len(parts) <= 1 and not text.endswith(" "):
            word = parts[0] if parts else "/"
            for cmd in COMMANDS:
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))
            return

        # Дополнение аргументов команды
        cmd = parts[0]
        args = COMMAND_ARGS.get(cmd, [])
        if not args:
            return
        # Текущее слово (то, что после пробела)
        current = "" if text.endswith(" ") else parts[-1]
        for arg in args:
            if arg.startswith(current):
                yield Completion(arg, start_position=-len(current))


class ChatUI:
    """
    Терминальный интерфейс чата.

    Объединяет rich-вывод и prompt_toolkit-ввод, обрабатывает команды,
    стриминг ответов, режим markdown (runtime-флаг) и автозавершение.
    """

    def __init__(
        self,
        config: Config,
        client: ChatClient,
        history_mgr: HistoryManager,
        usage_tracker: UsageTracker,
        markdown_enabled: bool,
        debug: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.history_mgr = history_mgr
        self.usage_tracker = usage_tracker
        self.markdown_enabled = markdown_enabled  # runtime-флаг (не сохраняется)
        self.debug = debug

        self.console = Console()
        self.current_model_id = config.default_model

        # Статистика текущей сессии
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cost = 0.0
        self.session_requests = 0

        # Флаг показа первой подсказки
        self._hint_shown = False

        # Создаём новую сессию
        self.session = self._new_session()

        # Настройка prompt_toolkit
        self.prompt_session = self._build_prompt_session()

    # ------------------------------------------------------------------ #
    #  Создание сессии и prompt_toolkit
    # ------------------------------------------------------------------ #
    def _new_session(self) -> Session:
        """Создаёт новую сессию с system-сообщением."""
        now = datetime.now()
        session_id = now.strftime("%Y%m%d_%H%M%S")
        session = Session(
            session_id=session_id,
            created_at=now.isoformat(timespec="seconds"),
            default_model=self.current_model_id,
        )
        # Добавляем system-сообщение
        session.messages.append(
            Message(role="system", content=self.config.system_prompt)
        )
        return session

    def _build_prompt_session(self) -> PromptSession:
        """Настраивает PromptSession с историей, автодополнением и хоткеями."""
        history = FileHistory(str(INPUT_HISTORY_PATH))

        # Биндинги: Enter — новая строка, Ctrl+Enter — отправка
        kb = KeyBindings()

        @kb.add("c-m")  # Enter (carriage return)
        def _(event) -> None:
            """Enter вставляет перевод строки."""
            event.current_buffer.insert_text("\n")

        @kb.add("c-j")  # Ctrl+Enter в большинстве терминалов даёт c-j
        def _(event) -> None:
            """Ctrl+Enter отправляет сообщение."""
            event.current_buffer.validate_and_handle()

        # Объединяем команды и историю в один completer
        completer = merge_completers([CommandCompleter()])

        return PromptSession(
            history=history,
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            multiline=True,
            key_bindings=kb,
            complete_while_typing=True,
        )

    # ------------------------------------------------------------------ #
    #  Утилиты модели
    # ------------------------------------------------------------------ #
    def _current_model_info(self) -> Optional[ModelInfo]:
        """Возвращает ModelInfo текущей модели (или None)."""
        return self.config.find_model(self.current_model_id)

    def _model_display_name(self, model_id: Optional[str] = None) -> str:
        """Возвращает name модели (или id, если не найдена)."""
        mid = model_id or self.current_model_id
        info = self.config.find_model(mid)
        return info.name if info else mid

    # ------------------------------------------------------------------ #
    #  Баннер и разделители
    # ------------------------------------------------------------------ #
    def show_banner(self) -> None:
        """Выводит стартовую панель с названием модели."""
        name = self._model_display_name()
        text = Text()
        text.append("🤖 AI Chat │ ", style="bold cyan")
        text.append(f"Модель: {name}\n", style="bold cyan")
        text.append("Введите /help для списка команд", style="dim")
        panel = Panel(text, border_style="cyan", expand=True)
        self.console.print(panel)

    def _rule(self) -> None:
        """Печатает горизонтальный разделитель."""
        self.console.print(Rule(style="dim"))

    # ------------------------------------------------------------------ #
    #  Вывод ошибок
    # ------------------------------------------------------------------ #
    def show_error(self, message: str, exc: Optional[Exception] = None) -> None:
        """Выводит ошибку в красной панели; traceback — в stderr при --debug."""
        panel = Panel(
            Text(message, style="bold red"),
            title="Ошибка",
            border_style="red",
        )
        self.console.print(panel)
        if self.debug and exc is not None:
            import traceback

            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def show_info(self, message: str) -> None:
        """Выводит информационное сообщение тусклым стилем."""
        self.console.print(message, style="dim")

    # ------------------------------------------------------------------ #
    #  Рендеринг ответа (две ветки: plain / markdown)
    # ------------------------------------------------------------------ #
    def render_response(self, text: str):
        """
        Возвращает рендерабл для текста ответа в зависимости от режима.

        - markdown ВЫКЛ: Text без markup и подсветки (plain).
        - markdown ВКЛ: Markdown.
        """
        if self.markdown_enabled:
            return Markdown(text)
        return Text(text, no_wrap=False)

    # ------------------------------------------------------------------ #
    #  Строка статистики
    # ------------------------------------------------------------------ #
    def _build_stats_line(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        cost: Optional[float],
        elapsed: float,
        streaming: bool = False,
    ) -> Text:
        """Собирает строку статистики ответа (dim cyan)."""
        suffix = "..." if streaming else ""
        tilde = "~" if streaming else ""

        line = Text(style="dim cyan")
        line.append("⏱ Токены: ")
        line.append(f"↑{prompt_tokens} ↓{completion_tokens}{suffix}")
        if reasoning_tokens > 0:
            line.append(f" (💭{reasoning_tokens})")
        line.append(" │ ")
        if cost is None:
            line.append("💰 —")
        else:
            line.append(f"💰 {tilde}{cost:.4f}₽")
        line.append(" │ ")
        line.append(f"⏳ {elapsed:.1f}с")
        return line

    # ------------------------------------------------------------------ #
    #  Отправка сообщения и стриминг ответа
    # ------------------------------------------------------------------ #
    def send_message(self, user_text: str) -> None:
        """
        Добавляет сообщение пользователя, выполняет запрос со стримингом,
        выводит ответ и обновляет статистику.
        """
        # Добавляем сообщение пользователя в сессию
        self.session.messages.append(Message(role="user", content=user_text))

        # Проверка контекста перед отправкой
        self._check_context()

        model_info = self._current_model_info()
        model_name = self._model_display_name()

        # Формируем сообщения для API (без thinking)
        api_messages = [m.to_api_dict() for m in self.session.messages]

        # Заголовок ответа
        self._rule()
        self.console.print(f"🤖 {model_name}", style="bold green")
        self.console.print()

        # Стриминг с Live
        result = self._stream_with_retry(model_info, api_messages, model_name)
        if result is None:
            # Запрос не удался — удаляем добавленное user-сообщение из истории API,
            # но оставляем в сессии (пользователь видит свой ввод)
            return

        # Считаем стоимость
        cost = self.client.calc_cost(
            model_info, result.prompt_tokens, result.completion_tokens
        )

        # Финальная строка статистики
        stats = self._build_stats_line(
            result.prompt_tokens,
            result.completion_tokens,
            result.reasoning_tokens,
            cost,
            result.elapsed,
            streaming=False,
        )
        self.console.print()
        self.console.print(stats)
        self._rule()

        # Сохраняем ответ в сессию
        assistant_msg = Message(
            role="assistant",
            content=result.content,
            model=self.current_model_id,
            thinking=result.thinking if result.thinking else None,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            interrupted=result.interrupted,
        )
        self.session.messages.append(assistant_msg)

        # Обновляем статистику сессии
        self.session_prompt_tokens += result.prompt_tokens
        self.session_completion_tokens += result.completion_tokens
        if cost is not None:
            self.session_cost += cost
        self.session_requests += 1

        # Обновляем глобальную статистику
        self.usage_tracker.add(
            result.prompt_tokens,
            result.completion_tokens,
            cost if cost is not None else 0.0,
        )

        # Автосохранение сессии
        self.history_mgr.save(self.session)

    def _stream_with_retry(
        self,
        model_info: Optional[ModelInfo],
        api_messages: list[dict[str, str]],
        model_name: str,
    ) -> Optional[StreamResult]:
        """
        Выполняет стриминг с обработкой ошибок и повторными попытками.
        Возвращает StreamResult или None при неустранимой ошибке.
        """
        supports_thinking = model_info.supports_thinking if model_info else False

        attempt = 0
        delay = 1.0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                return self._do_stream(api_messages, supports_thinking, model_info)
            except AuthenticationError as e:
                self.show_error(
                    "Неверный API-ключ. Проверьте поле api_key в config.json", e
                )
                return None
            except RateLimitError as e:
                retry_after = self._extract_retry_after(e)
                self.show_info(f"Превышен лимит. Повтор через {retry_after}с...")
                time.sleep(retry_after)
                continue
            except APIStatusError as e:
                code = getattr(e, "status_code", 0)
                if code == 400 and self._is_context_error(e):
                    handled = self._handle_context_overflow(model_info)
                    if handled:
                        # Повторяем с обрезанным контекстом
                        api_messages = [
                            m.to_api_dict() for m in self.session.messages
                        ]
                        continue
                    return None
                if code >= 500:
                    self.show_info(f"Ошибка сервера ({code}). Повтор через {delay:.0f}с...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                self.show_error(f"Ошибка API ({code}): {e}", e)
                return None
            except (APIConnectionError, APITimeoutError) as e:
                if attempt < MAX_RETRIES:
                    self.show_info(
                        f"Повторная попытка {attempt + 1}/{MAX_RETRIES} "
                        f"через {delay:.0f}с..."
                    )
                    time.sleep(delay)
                    delay *= 2
                    continue
                self.show_error(f"Сетевая ошибка: {e}", e)
                return None
            except Exception as e:  # неожиданная ошибка
                self.show_error(f"Непредвиденная ошибка: {e}", e)
                return None
        self.show_error("Превышено число повторных попыток.")
        return None

    def _do_stream(
        self,
        api_messages: list[dict[str, str]],
        supports_thinking: bool,
        model_info: Optional[ModelInfo],
    ) -> StreamResult:
        """Выполняет один проход стриминга с Live-обновлением."""
        # Флаг: выводили ли заголовок "Размышляю..."
        state = {"thinking_header": False}

        # Используем Live для обновления текста ответа в реальном времени
        with Live(
            console=self.console,
            refresh_per_second=12,
            transient=False,
        ) as live:

            def on_chunk(content: str, thinking: str, completion_tokens: int) -> None:
                """Колбэк обновления Live на каждый chunk."""
                renderables: list[Any] = []

                # Блок thinking (всегда plain, dim italic)
                if thinking:
                    think_text = Text(style="dim italic")
                    think_text.append("💭 Размышляю...\n")
                    # Отступ для строк размышлений
                    for ln in thinking.splitlines():
                        think_text.append(f"   {ln}\n")
                    renderables.append(think_text)
                    if content:
                        renderables.append(Text(""))  # отступ

                # Текст ответа
                if content:
                    renderables.append(self.render_response(content))

                # Строка статистики (streaming)
                prompt_est = (
                    self.client.estimate_tokens(
                        "".join(m["content"] for m in api_messages)
                    )
                )
                stats = self._build_stats_line(
                    prompt_est,
                    completion_tokens,
                    0,
                    None,  # стоимость не показываем в процессе (или ~)
                    0.0,
                    streaming=True,
                )
                renderables.append(Text(""))
                renderables.append(stats)

                # Собираем группу
                from rich.console import Group

                live.update(Group(*renderables))

            result = self.client.stream(
                self.current_model_id, api_messages, on_chunk=on_chunk
            )

            # Финальное обновление Live без индикаторов стриминга
            renderables: list[Any] = []
            if result.thinking:
                think_text = Text(style="dim italic")
                think_text.append("💭 Размышляю...\n")
                for ln in result.thinking.splitlines():
                    think_text.append(f"   {ln}\n")
                renderables.append(think_text)
                renderables.append(Text(""))
            if result.content:
                renderables.append(self.render_response(result.content))
            from rich.console import Group

            live.update(Group(*renderables) if renderables else Text(""))

        return result

    @staticmethod
    def _extract_retry_after(exc: Exception) -> int:
        """Извлекает Retry-After из заголовков ошибки 429 (по умолчанию 5с)."""
        try:
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", {})
                ra = headers.get("Retry-After") or headers.get("retry-after")
                if ra:
                    return int(float(ra))
        except (ValueError, AttributeError):
            pass
        return 5

    @staticmethod
    def _is_context_error(exc: Exception) -> bool:
        """Определяет, связана ли ошибка 400 с превышением контекста."""
        msg = str(exc).lower()
        return "context" in msg or "maximum" in msg or "token" in msg

    def _handle_context_overflow(self, model_info: Optional[ModelInfo]) -> bool:
        """
        Обрабатывает превышение контекста: предлагает обрезать старые сообщения.
        Возвращает True, если контекст обрезан и нужно повторить запрос.
        """
        window = model_info.context_window if model_info else 0
        self.show_error(
            f"Превышен контекст модели ({window} токенов). "
            f"Используйте /new для нового разговора."
        )
        answer = self.console.input(
            "[bold]Удалить старые сообщения и повторить? (y/n): [/bold]"
        ).strip().lower()
        if answer == "y":
            # Удаляем самые старые сообщения (кроме system)
            system_msgs = [m for m in self.session.messages if m.role == "system"]
            other_msgs = [m for m in self.session.messages if m.role != "system"]
            # Удаляем половину старых
            keep = other_msgs[len(other_msgs) // 2 :]
            self.session.messages = system_msgs + keep
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Управление контекстом
    # ------------------------------------------------------------------ #
    def _check_context(self) -> None:
        """Проверяет приблизительный размер контекста и предупреждает."""
        model_info = self._current_model_info()
        if model_info is None:
            return
        total = sum(
            self.client.estimate_tokens(m.content) for m in self.session.messages
        )
        window = model_info.context_window
        if window <= 0:
            return
        ratio = total / window
        if ratio > CONTEXT_WARN_THRESHOLD:
            pct = int(ratio * 100)
            self.console.print(
                f"Внимание: использовано ~{pct}% контекста ({total}/{window})",
                style="yellow",
            )

    # ------------------------------------------------------------------ #
    #  Обработка команд
    # ------------------------------------------------------------------ #
    def handle_command(self, text: str) -> None:
        """Разбирает и выполняет команду, начинающуюся с '/'."""
        parts = text.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/new":
            self._cmd_new(clear="--clear" in args)
        elif cmd == "/history":
            self._cmd_history(args)
        elif cmd == "/model":
            self._cmd_model(args)
        elif cmd in ("/markdown", "/md"):
            self._cmd_markdown(cmd, args)
        elif cmd == "/file":
            self._cmd_file(args)
        elif cmd == "/clear":
            self._cmd_clear()
        elif cmd == "/usage":
            self._cmd_usage()
        elif cmd == "/help":
            self._cmd_help()
        elif cmd == "/exit":
            self._cmd_exit()
        else:
            self.show_info(
                f"Неизвестная команда: {cmd}. Введите /help для справки."
            )

    def _cmd_new(self, clear: bool = False) -> None:
        """Команда /new — новый разговор."""
        self.history_mgr.save(self.session)
        self.session = self._new_session()
        # Сбрасываем статистику сессии
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cost = 0.0
        self.session_requests = 0
        if clear:
            self.console.clear()
            self.show_banner()
        self.show_info("Сессия сохранена. Новый разговор начат.")

    def _cmd_history(self, args: list[str]) -> None:
        """Команда /history — список/загрузка сессий."""
        # /history <session_id>
        if args and args[0] != "all":
            session_id = args[0]
            loaded = self.history_mgr.load(session_id)
            if loaded is None:
                self.show_info(f"Сессия {session_id} не найдена")
                return
            self._load_session(loaded)
            return

        # /history all — все сессии, иначе последние 20
        limit = None if (args and args[0] == "all") else 20
        sessions = self.history_mgr.list_sessions(limit=limit)
        if not sessions:
            self.show_info("История пуста.")
            return

        # Таблица
        table = Table(title="История чатов")
        table.add_column("№", justify="right", style="cyan")
        table.add_column("Дата", style="green")
        table.add_column("Модель")
        table.add_column("Первое сообщение")
        table.add_column("Токены", justify="right", style="dim cyan")

        for i, sess in enumerate(sessions, 1):
            first = sess.first_user_message().replace("\n", " ")
            if len(first) > 60:
                first = first[:57] + "..."
            date_str = sess.created_at.replace("T", " ")
            table.add_row(
                str(i),
                date_str,
                self._model_display_name(sess.default_model),
                first,
                str(sess.total_tokens),
            )
        self.console.print(table)

        # Запрос номера для загрузки
        choice = self.console.input(
            "[dim]Введите номер для загрузки (Enter — отмена): [/dim]"
        ).strip()
        if not choice:
            return
        if not choice.isdigit():
            self.show_info("Некорректный номер.")
            return
        idx = int(choice) - 1
        if not (0 <= idx < len(sessions)):
            self.show_info("Номер вне диапазона.")
            return
        # Перезагружаем полную сессию по id
        full = self.history_mgr.load(sessions[idx].session_id)
        if full:
            self._load_session(full)

    def _load_session(self, session: Session) -> None:
        """Загружает сессию как текущий контекст."""
        # Сохраняем текущую сессию перед заменой
        self.history_mgr.save(self.session)
        self.session = session
        if session.default_model:
            self.current_model_id = session.default_model
        # Сбрасываем статистику сессии (она пересчитывается из загруженной)
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cost = 0.0
        self.session_requests = 0
        for m in session.messages:
            if m.role == "assistant" and m.usage:
                self.session_prompt_tokens += m.usage.get("prompt_tokens", 0)
                self.session_completion_tokens += m.usage.get("completion_tokens", 0)
                self.session_requests += 1
        self.console.clear()
        self.show_banner()
        self.show_info(f"Загружена сессия {session.session_id}")
        # Кратко выводим историю
        self._replay_session()

    def _replay_session(self) -> None:
        """Выводит сообщения загруженной сессии."""
        for m in self.session.messages:
            if m.role == "system":
                continue
            if m.role == "user":
                self._rule()
                self.console.print("Вы › ", style="bold", end="")
                self.console.print(m.content, markup=False, highlight=False)
            elif m.role == "assistant":
                self._rule()
                name = self._model_display_name(m.model)
                self.console.print(f"🤖 {name}", style="bold green")
                self.console.print()
                if m.thinking:
                    think = Text(style="dim italic")
                    think.append("💭 Размышления:\n")
                    for ln in m.thinking.splitlines():
                        think.append(f"   {ln}\n")
                    self.console.print(think)
                self.console.print(self.render_response(m.content))
        self._rule()

    def _cmd_model(self, args: list[str]) -> None:
        """Команда /model — выбор/переключение модели."""
        if not self.config.models:
            self.show_info("Список моделей пуст (config.models).")
            return

        # /model без аргументов — пронумерованный список
        if not args:
            self._show_model_list()
            choice = self.console.input(
                "[dim]Введите номер модели (Enter — отмена): [/dim]"
            ).strip()
            if not choice:
                return
            matches = self.config.search_models(choice)
            if len(matches) == 1:
                self._switch_model(matches[0])
            else:
                self.show_info("Модель не найдена")
            return

        # Парсим флаги
        provider: Optional[str] = None
        make_default = False
        positional: list[str] = []
        i = 0
        while i < len(args):
            a = args[i]
            if a == "--provider":
                if i + 1 < len(args):
                    provider = args[i + 1]
                    i += 2
                    continue
            elif a == "--default":
                make_default = True
            else:
                positional.append(a)
            i += 1

        query = " ".join(positional)
        matches = self.config.search_models(query)

        if len(matches) == 0:
            self.show_info("Модель не найдена")
            return
        if len(matches) > 1:
            self.show_info("Найдено несколько моделей:")
            for m in matches:
                self.console.print(f"  • {m.name} ({m.id})", style="dim")
            return

        model = matches[0]
        self._switch_model(model, announce=False)

        # Обработка --provider
        if provider:
            # Пытаемся найти провайдера в конфиге (если поддерживается),
            # иначе просто обновляем api_base нельзя без данных — сообщаем.
            self.show_info(
                f"Модель переключена на {model.name}, провайдер: {provider}"
            )
        # Обработка --default
        elif make_default:
            self.config.set_default_model(model.id)
            self.show_info(f"Модель {model.name} установлена по умолчанию.")
        else:
            self.show_info(f"Модель переключена на {model.name}")

    def _show_model_list(self) -> None:
        """Выводит пронумерованный список моделей."""
        table = Table(title="Доступные модели")
        table.add_column("№", justify="right", style="cyan")
        table.add_column("Название")
        table.add_column("ID", style="dim")
        table.add_column("Контекст", justify="right")
        table.add_column("💭", justify="center")
        for i, m in enumerate(self.config.models, 1):
            marker = "✓" if m.id == self.current_model_id else ""
            thinking = "💭" if m.supports_thinking else ""
            table.add_row(
                str(i) + (" " + marker if marker else ""),
                m.name,
                m.id,
                str(m.context_window),
                thinking,
            )
        self.console.print(table)

    def _switch_model(self, model: ModelInfo, announce: bool = True) -> None:
        """Переключает текущую модель."""
        self.current_model_id = model.id
        self.session.default_model = model.id
        if announce:
            self.show_info(f"Модель переключена на {model.name}")

    def _cmd_markdown(self, cmd: str, args: list[str]) -> None:
        """Команда /markdown и /md — управление режимом рендеринга."""
        arg = args[0].lower() if args else ""

        if not arg:
            # /md без аргумента — переключить; /markdown — показать состояние
            if cmd == "/md":
                self.markdown_enabled = not self.markdown_enabled
                state = "ВКЛ" if self.markdown_enabled else "ВЫКЛ"
                self.show_info(f"Markdown-рендеринг: {state}")
            else:
                state = "ВКЛ" if self.markdown_enabled else "ВЫКЛ"
                self.show_info(f"Markdown-рендеринг: {state}")
            return

        if arg == "on":
            self.markdown_enabled = True
            self.show_info("Markdown-рендеринг включён.")
        elif arg == "off":
            self.markdown_enabled = False
            self.show_info("Markdown-рендеринг выключен.")
        elif arg == "toggle":
            self.markdown_enabled = not self.markdown_enabled
            state = "ВКЛ" if self.markdown_enabled else "ВЫКЛ"
            self.show_info(f"Markdown-рендеринг: {state}")
        else:
            self.show_info(f"Неизвестный аргумент: {arg}")

    def _cmd_file(self, args: list[str]) -> None:
        """Команда /file — отправить содержимое текстового файла."""
        if not args:
            self.show_info("Укажите путь: /file <путь>")
            return
        path = Path(" ".join(args)).expanduser()
        if not path.exists():
            self.show_error(f"Файл не найден: {path}")
            return
        # Проверка размера
        size = path.stat().st_size
        if size > MAX_FILE_SIZE:
            self.show_error(
                f"Файл слишком большой ({size} байт, максимум {MAX_FILE_SIZE})."
            )
            return
        # Чтение
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            self.show_error(
                f"Не удалось прочитать файл: {e}. "
                f"Убедитесь что файл в кодировке UTF-8."
            )
            return

        # Превью (первые 3 строки + размер)
        lines = content.splitlines()
        preview = "\n".join(lines[:3])
        self.console.print(
            Panel(
                Text(preview, no_wrap=False),
                title=f"Превью: {path.name} ({size} байт)",
                border_style="cyan",
            )
        )
        answer = self.console.input(
            "[bold]Отправить содержимое файла? (y/n): [/bold]"
        ).strip().lower()
        if answer == "y":
            self.send_message(content)

    def _cmd_clear(self) -> None:
        """Команда /clear — очистить экран (контекст сохраняется)."""
        self.console.clear()
        self.show_banner()

    def _cmd_usage(self) -> None:
        """Команда /usage — статистика сессии и за всё время."""
        # Текущая сессия
        avg = (
            self.session_cost / self.session_requests
            if self.session_requests
            else 0.0
        )
        sess_table = Table(title="Текущая сессия")
        sess_table.add_column("Метрика")
        sess_table.add_column("Значение", justify="right")
        sess_table.add_row("Prompt токены", str(self.session_prompt_tokens))
        sess_table.add_row("Completion токены", str(self.session_completion_tokens))
        sess_table.add_row("Стоимость", f"{self.session_cost:.4f}₽")
        sess_table.add_row("Запросов", str(self.session_requests))
        sess_table.add_row("Средняя стоимость запроса", f"{avg:.4f}₽")
        self.console.print(sess_table)

        # За всё время
        d = self.usage_tracker.data
        all_table = Table(title="За всё время")
        all_table.add_column("Метрика")
        all_table.add_column("Значение", justify="right")
        all_table.add_row("Prompt токены", str(d.get("total_prompt_tokens", 0)))
        all_table.add_row(
            "Completion токены", str(d.get("total_completion_tokens", 0))
        )
        all_table.add_row(
            "Суммарная стоимость", f"{d.get('total_cost_rub', 0.0):.4f}₽"
        )
        all_table.add_row("Всего запросов", str(d.get("total_requests", 0)))
        self.console.print(all_table)

    def _cmd_help(self) -> None:
        """Команда /help — таблица команд."""
        table = Table(title="Команды")
        table.add_column("Команда", style="bold yellow")
        table.add_column("Описание")
        rows = [
            ("/new [--clear]", "Новый разговор (--clear — очистить экран)"),
            ("/history [all|<id>]", "Список/загрузка сессий"),
            ("/model [имя|№] [--default] [--provider <p>]", "Выбор модели"),
            ("/markdown [on|off|toggle]", "Управление markdown-рендерингом"),
            ("/md [on|off|toggle]", "То же; без аргумента — переключить"),
            ("/file <путь>", "Отправить содержимое текстового файла"),
            ("/clear", "Очистить экран (контекст сохраняется)"),
            ("/usage", "Статистика токенов и стоимости"),
            ("/help", "Эта справка"),
            ("/exit", "Выход"),
        ]
        for cmd, desc in rows:
            table.add_row(cmd, desc)
        self.console.print(table)
        self.console.print(
            "\n[dim]Многострочный ввод: Enter — новая строка, "
            "Ctrl+Enter — отправить.[/dim]"
        )
        state = "ВКЛ" if self.markdown_enabled else "ВЫКЛ"
        self.console.print(
            f"[dim]Markdown-рендеринг сейчас: {state} "
            f"(переключить: /md).[/dim]"
        )

    def _cmd_exit(self) -> None:
        """Команда /exit — сохранить и выйти."""
        self.history_mgr.save(self.session)
        self.console.print("До свидания!", style="dim")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    #  Главный цикл
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Запускает интерактивный цикл ввода/вывода."""
        self.console.clear()
        self.show_banner()

        # Предупреждение, если default_model не найдена в списке
        if self.config.models and self._current_model_info() is None:
            self.show_info(
                f"Модель {self.current_model_id} не найдена в списке models. "
                f"Стоимость не будет рассчитываться."
            )

        while True:
            try:
                # Подсказка при первом запуске
                if not self._hint_shown:
                    self.console.print(
                        "[dim](Ctrl+Enter — отправить, /help — справка, "
                        "/md — markdown ВКЛ/ВЫКЛ)[/dim]"
                    )

                user_input = self.prompt_session.prompt("Вы › ")

                # Скрываем подсказку после первого ввода
                self._hint_shown = True

                # Пустой ввод — игнорируем
                if not user_input or not user_input.strip():
                    continue

                stripped = user_input.strip()

                # Команда?
                if stripped.startswith("/"):
                    self.handle_command(stripped)
                    continue

                # Обычное сообщение
                self.send_message(stripped)

            except KeyboardInterrupt:
                # Ctrl+C в режиме ожидания ввода — выход
                self.history_mgr.save(self.session)
                self.console.print("\nДо свидания!", style="dim")
                sys.exit(0)
            except EOFError:
                # Ctrl+D — аналог /exit
                self.history_mgr.save(self.session)
                self.console.print("\nДо свидания!", style="dim")
                sys.exit(0)

    # ------------------------------------------------------------------ #
    #  Неинтерактивный режим / одиночный запрос
    # ------------------------------------------------------------------ #
    def run_single_output(self, user_text: str) -> None:
        """
        Неинтерактивный режим (--output): отправить запрос, вывести только
        ответ в stdout и завершить работу.
        """
        self.session.messages.append(Message(role="user", content=user_text))
        api_messages = [m.to_api_dict() for m in self.session.messages]
        model_info = self._current_model_info()

        try:
            result = self.client.stream(
                self.current_model_id, api_messages, on_chunk=None
            )
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            sys.exit(1)

        # Выводим только текст ответа
        print(result.content)

        # Сохраняем сессию и статистику
        cost = self.client.calc_cost(
            model_info, result.prompt_tokens, result.completion_tokens
        )
        assistant_msg = Message(
            role="assistant",
            content=result.content,
            model=self.current_model_id,
            thinking=result.thinking if result.thinking else None,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        )
        self.session.messages.append(assistant_msg)
        self.usage_tracker.add(
            result.prompt_tokens,
            result.completion_tokens,
            cost if cost is not None else 0.0,
        )
        self.history_mgr.save(self.session)

    def load_session_by_id(self, session_id: str) -> bool:
        """Загружает сессию по id как контекст (для --history). True — успех."""
        loaded = self.history_mgr.load(session_id)
        if loaded is None:
            self.show_error(f"Сессия {session_id} не найдена")
            return False
        self.session = loaded
        if loaded.default_model:
            self.current_model_id = loaded.default_model
        for m in loaded.messages:
            if m.role == "assistant" and m.usage:
                self.session_prompt_tokens += m.usage.get("prompt_tokens", 0)
                self.session_completion_tokens += m.usage.get("completion_tokens", 0)
                self.session_requests += 1
        return True


# ============================================================================
# 9. ТОЧКА ВХОДА
# ============================================================================
def read_text_file(path_str: str) -> Optional[str]:
    """
    Читает текстовый файл UTF-8 с проверкой существования и размера.
    Возвращает содержимое или None (с выводом ошибки в stderr).
    """
    path = Path(path_str).expanduser()
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return None
    if path.stat().st_size > MAX_FILE_SIZE:
        print(f"Файл слишком большой (максимум {MAX_FILE_SIZE} байт).", file=sys.stderr)
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        print(
            f"Не удалось прочитать файл: {e}. "
            f"Убедитесь что файл в кодировке UTF-8.",
            file=sys.stderr,
        )
        return None


def main() -> None:
    """Точка входа: парсинг аргументов и запуск нужного режима."""
    parser = argparse.ArgumentParser(
        description="🤖 AI Chat — терминальный чат-клиент для LLM"
    )
    parser.add_argument(
        "--markdown", "-m", action="store_true",
        help="Принудительно включить markdown-режим",
    )
    parser.add_argument(
        "--no-markdown", action="store_true",
        help="Принудительно выключить markdown-режим",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Полный traceback ошибок в stderr",
    )
    parser.add_argument(
        "--history", metavar="SESSION_ID",
        help="Открыть сохранённую сессию при запуске",
    )
    parser.add_argument(
        "--input", metavar="FILE",
        help="Считать текстовый файл как запрос пользователя",
    )
    parser.add_argument(
        "--output", metavar="TEXT",
        help="Неинтерактивный режим: отправить текст, вывести ответ и выйти",
    )
    args = parser.parse_args()

    # Загрузка конфига
    try:
        config = Config.load()
    except ConfigError as e:
        # Выводим ошибку конфига без traceback
        Console().print(Panel(Text(str(e), style="bold red"),
                              title="Ошибка", border_style="red"))
        sys.exit(1)

    # Определяем режим markdown с учётом флагов
    markdown_enabled = config.render_markdown
    if args.markdown:
        markdown_enabled = True
    if args.no_markdown:
        markdown_enabled = False

    # Инициализация компонентов
    client = ChatClient(config)
    history_mgr = HistoryManager()
    usage_tracker = UsageTracker()

    ui = ChatUI(
        config=config,
        client=client,
        history_mgr=history_mgr,
        usage_tracker=usage_tracker,
        markdown_enabled=markdown_enabled,
        debug=args.debug,
    )

    # Загрузка сессии из --history (как контекст)
    if args.history:
        if not ui.load_session_by_id(args.history):
            sys.exit(1)

    # Определяем текст запроса из --input / --output
    request_text: Optional[str] = None
    if args.input:
        request_text = read_text_file(args.input)
        if request_text is None:
            sys.exit(1)
    if args.output:
        request_text = args.output

    # Неинтерактивный режим (--output): вывести ответ и выйти
    if args.output is not None:
        ui.run_single_output(request_text or "")
        sys.exit(0)

    # Режим --input: отправить файл как запрос, затем продолжить интерактивно
    if args.input:
        ui.console.clear()
        ui.show_banner()
        ui.send_message(request_text or "")
        ui._hint_shown = True
        ui.run()
        return

    # Обычный интерактивный режим
    ui.run()


if __name__ == "__main__":
    main()
