"""
Терминальный чат-клиент для LLM с OpenAI-совместимым API.

Поддерживает стриминг, thinking-модели, историю сессий,
многострочный ввод и расчёт стоимости запросов.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Импорты сторонних библиотек с понятной ошибкой при отсутствии
# ──────────────────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError, RateLimitError
except ImportError:
    print("Ошибка: не установлена библиотека openai. Выполните: pip install openai", file=sys.stderr)
    sys.exit(1)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.console import Group
except ImportError:
    print("Ошибка: не установлена библиотека rich. Выполните: pip install rich", file=sys.stderr)
    sys.exit(1)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style as PTStyle
except ImportError:
    print("Ошибка: не установлена библиотека prompt_toolkit. Выполните: pip install prompt_toolkit", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────────────────────

VERSION = "1.0.0"

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"
HISTORY_DIR = BASE_DIR / "history"

MAX_FILE_SIZE = 100 * 1024  # 100 КБ — лимит для команды /file
MAX_RETRIES = 3              # Количество попыток при сетевых ошибках
RETRY_DELAYS = [1, 2, 4]     # Задержки между попытками (секунды)

# Шаблон конфига при первом запуске
DEFAULT_CONFIG = {
    "api_key": "sk-YOUR-KEY-HERE",
    "api_base": "https://api.openai.com/v1",
    "default_model": "gpt-4o",
    "system_prompt": "Ты полезный ассистент.",
    "generation": {
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens": 4096
    },
    "models": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "context_window": 128000,
            "input_price_per_1m_tokens_rub": 150.0,
            "output_price_per_1m_tokens_rub": 600.0
        },
        {
            "id": "deepseek-reasoner",
            "name": "DeepSeek R1",
            "context_window": 65536,
            "supports_thinking": True,
            "input_price_per_1m_tokens_rub": 3.0,
            "output_price_per_1m_tokens_rub": 15.0
        }
    ]
}


# ──────────────────────────────────────────────────────────────────────────────
# Типы данных
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Информация о модели из конфига."""
    id: str
    name: str
    context_window: int = 128000
    supports_thinking: bool = False
    input_price_per_1m_tokens_rub: float = 0.0
    output_price_per_1m_tokens_rub: float = 0.0


@dataclass
class Message:
    """Одно сообщение в истории чата."""
    role: str                              # system | user | assistant
    content: str
    timestamp: str = ""
    model: Optional[str] = None            # Только для assistant
    thinking: Optional[str] = None         # Reasoning-контент (если был)
    usage: Optional[dict[str, int]] = None # {"prompt_tokens": N, "completion_tokens": M}
    interrupted: bool = False              # Прервано ли Ctrl+C

    def to_api_dict(self) -> dict[str, str]:
        """Возвращает dict для отправки в API (без служебных полей)."""
        return {"role": self.role, "content": self.content}

    def to_json(self) -> dict[str, Any]:
        """Сериализация в JSON для сохранения."""
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }
        if self.model is not None:
            d["model"] = self.model
        if self.thinking:
            d["thinking"] = self.thinking
        if self.usage is not None:
            d["usage"] = self.usage
        if self.interrupted:
            d["interrupted"] = True
        return d

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Message":
        """Десериализация из JSON."""
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
    """Сессия чата — набор сообщений + метаданные."""
    session_id: str
    created_at: str
    default_model: str
    messages: list[Message] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "default_model": self.default_model,
            "messages": [m.to_json() for m in self.messages],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            default_model=data.get("default_model", ""),
            messages=[Message.from_json(m) for m in data.get("messages", [])],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────────────────────────────────

class ConfigError(Exception):
    """Ошибка валидации/парсинга конфига."""


class Config:
    """Загрузка и валидация конфигурации приложения."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.api_key: str = ""
        self.api_base: str = ""
        self.default_model: str = ""
        self.system_prompt: str = ""
        self.temperature: float = 0.7
        self.top_p: float = 1.0
        self.max_tokens: int = 4096
        self.models: list[ModelInfo] = []

    @classmethod
    def ensure_exists(cls, path: Path) -> bool:
        """Создаёт шаблон конфига, если файл не существует. Возвращает True если создан."""
        if path.exists():
            return False
        path.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return True

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Загружает и валидирует конфиг."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"Не удалось прочитать config.json: {e}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"Ошибка парсинга config.json (строка {e.lineno}, колонка {e.colno}): {e.msg}"
            )

        if not isinstance(data, dict):
            raise ConfigError("Корень config.json должен быть объектом")

        cfg = cls(path)

        # Обязательные поля
        for field_name in ("api_key", "api_base", "default_model"):
            if field_name not in data:
                raise ConfigError(f'отсутствует обязательное поле "{field_name}"')
            if not isinstance(data[field_name], str):
                raise ConfigError(
                    f'поле "{field_name}" должно быть строкой, '
                    f'получено "{type(data[field_name]).__name__}"'
                )

        cfg.api_key = data["api_key"]
        cfg.api_base = data["api_base"]
        cfg.default_model = data["default_model"]

        if cfg.api_key.strip() in ("", "sk-YOUR-KEY-HERE"):
            raise ConfigError("укажите реальный api_key в config.json")

        # Опциональные верхнего уровня
        cfg.system_prompt = data.get("system_prompt", "Ты полезный ассистент.")
        if not isinstance(cfg.system_prompt, str):
            raise ConfigError(
                f'поле "system_prompt" должно быть строкой, '
                f'получено "{type(cfg.system_prompt).__name__}"'
            )

        # Блок generation
        gen = data.get("generation", {})
        if not isinstance(gen, dict):
            raise ConfigError('поле "generation" должно быть объектом')

        def _num(field_name: str, default: float, typ: type) -> float:
            val = gen.get(field_name, default)
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise ConfigError(
                    f'поле "{field_name}" должно быть числом, получено "{val}"'
                )
            return typ(val)

        cfg.temperature = float(_num("temperature", 0.7, float))
        cfg.top_p = float(_num("top_p", 1.0, float))
        cfg.max_tokens = int(_num("max_tokens", 4096, int))

        # Список моделей
        raw_models = data.get("models", [])
        if not isinstance(raw_models, list):
            raise ConfigError('поле "models" должно быть массивом')

        for idx, m in enumerate(raw_models):
            if not isinstance(m, dict):
                raise ConfigError(f"models[{idx}] должно быть объектом")
            if "id" not in m:
                raise ConfigError(f'models[{idx}]: отсутствует поле "id"')
            cfg.models.append(ModelInfo(
                id=str(m["id"]),
                name=str(m.get("name", m["id"])),
                context_window=int(m.get("context_window", 128000)),
                supports_thinking=bool(m.get("supports_thinking", False)),
                input_price_per_1m_tokens_rub=float(m.get("input_price_per_1m_tokens_rub", 0.0)),
                output_price_per_1m_tokens_rub=float(m.get("output_price_per_1m_tokens_rub", 0.0)),
            ))

        return cfg

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Ищет модель по id (точное совпадение)."""
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def find_models(self, query: str) -> list[ModelInfo]:
        """Поиск моделей по id или name (частичное, case-insensitive)."""
        q = query.lower().strip()
        results = []
        for m in self.models:
            if q == m.id.lower() or q == m.name.lower():
                return [m]  # Точное совпадение — возвращаем сразу
            if q in m.id.lower() or q in m.name.lower():
                results.append(m)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# Управление историей сессий
# ──────────────────────────────────────────────────────────────────────────────

class HistoryManager:
    """Сохранение/загрузка сессий чата в виде JSON-файлов."""

    def __init__(self, history_dir: Path) -> None:
        self.dir = history_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, session: Session) -> Path:
        """Сохраняет сессию в файл {session_id}.json."""
        path = self.dir / f"{session.session_id}.json"
        path.write_text(
            json.dumps(session.to_json(), ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return path

    def list_sessions(self) -> list[Session]:
        """Возвращает список всех сессий, отсортированных по дате (новые первыми)."""
        sessions = []
        for path in sorted(self.dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(Session.from_json(data))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return sessions

    def load(self, session_id: str) -> Optional[Session]:
        """Загружает сессию по id."""
        path = self.dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Session.from_json(data)
        except (json.JSONDecodeError, KeyError, OSError):
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Подсчёт токенов и стоимости
# ──────────────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Грубая оценка количества токенов (если API не вернул usage)."""
    if not text:
        return 0
    # Если есть кириллица — делим на 2, иначе на 4
    has_cyrillic = any('\u0400' <= ch <= '\u04FF' for ch in text)
    return max(1, len(text) // (2 if has_cyrillic else 4))


def calculate_cost(prompt_tokens: int, completion_tokens: int, model: Optional[ModelInfo]) -> Optional[float]:
    """Рассчитывает стоимость запроса в рублях. None если модель не известна."""
    if model is None:
        return None
    cost = (
        prompt_tokens * model.input_price_per_1m_tokens_rub / 1_000_000
        + completion_tokens * model.output_price_per_1m_tokens_rub / 1_000_000
    )
    return cost


# ──────────────────────────────────────────────────────────────────────────────
# Клиент чата — API + стриминг
# ──────────────────────────────────────────────────────────────────────────────

class ChatInterrupted(Exception):
    """Стрим был прерван пользователем (Ctrl+C)."""


@dataclass
class StreamResult:
    """Результат стриминга ответа модели."""
    content: str
    thinking: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    elapsed: float
    interrupted: bool = False


class ChatClient:
    """Обёртка над OpenAI-клиентом со стримингом и обработкой ошибок."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def stream_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        on_thinking: Optional[callable] = None,
        on_content: Optional[callable] = None,
    ) -> StreamResult:
        """
        Делает стримящий запрос к API.

        on_thinking(text) — вызывается при получении нового куска reasoning
        on_content(text)  — вызывается при получении нового куска ответа
        """
        # Подсчёт prompt-токенов как fallback (если API не вернёт usage)
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        fallback_prompt_tokens = estimate_tokens(prompt_text)

        # Параметры запроса
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

        # Пробуем включить usage в стриме (поддерживается не всеми API)
        try:
            kwargs_with_usage = dict(kwargs)
            kwargs_with_usage["stream_options"] = {"include_usage": True}
            stream = self.client.chat.completions.create(**kwargs_with_usage)
        except TypeError:
            # SDK не знает stream_options — повторяем без него
            stream = self.client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        reasoning_tokens = 0
        interrupted = False
        start = time.monotonic()

        try:
            for chunk in stream:
                # Извлекаем usage если он есть
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or prompt_tokens
                    completion_tokens = getattr(usage, "completion_tokens", 0) or completion_tokens
                    # reasoning_tokens может быть вложен в completion_tokens_details
                    details = getattr(usage, "completion_tokens_details", None)
                    if details is not None:
                        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or reasoning_tokens

                # Если нет choices — это финальный chunk только с usage
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Thinking / reasoning content
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    thinking_parts.append(reasoning)
                    if on_thinking:
                        on_thinking(reasoning)

                # Обычный контент
                if delta.content:
                    content_parts.append(delta.content)
                    if on_content:
                        on_content(delta.content)

        except KeyboardInterrupt:
            interrupted = True

        elapsed = time.monotonic() - start
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)

        # Fallback на оценку если usage не пришёл
        if prompt_tokens == 0:
            prompt_tokens = fallback_prompt_tokens
        if completion_tokens == 0:
            completion_tokens = estimate_tokens(content + thinking)

        return StreamResult(
            content=content,
            thinking=thinking,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            elapsed=elapsed,
            interrupted=interrupted,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Пользовательский интерфейс
# ──────────────────────────────────────────────────────────────────────────────

class ChatUI:
    """Основной класс UI: ввод, команды, рендеринг ответов."""

    def __init__(self, config: Config, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.console = Console()
        self.client = ChatClient(config)
        self.history_mgr = HistoryManager(HISTORY_DIR)
        self.current_model_id: str = config.default_model
        self.session: Session = self._new_session()
        self.input_session = self._build_prompt_session()
        self.first_input = True  # Показывать ли подсказку под промптом

    # ── Создание новой сессии ─────────────────────────────────────────────
    def _new_session(self) -> Session:
        now = datetime.now()
        sid = now.strftime("%Y%m%d_%H%M%S")
        sess = Session(
            session_id=sid,
            created_at=now.isoformat(timespec="seconds"),
            default_model=self.current_model_id,
            messages=[Message(
                role="system",
                content=self.config.system_prompt,
                timestamp=now.isoformat(timespec="seconds"),
            )],
        )
        return sess

    # ── prompt_toolkit конфигурация ───────────────────────────────────────
    def _build_prompt_session(self) -> PromptSession:
        """Создаёт сессию ввода с биндингами Enter / Alt+Enter."""
        kb = KeyBindings()

        @kb.add("enter")
        def _(event):
            # Enter = новая строка
            event.current_buffer.insert_text("\n")

        @kb.add("escape", "enter")
        def _(event):
            # Alt+Enter (Esc → Enter) = отправка
            event.current_buffer.validate_and_handle()

        # На некоторых терминалах Alt+Enter присылается как этот код
        @kb.add("c-j")  # Ctrl+J — fallback (как на некоторых системах Alt+Enter)
        def _(event):
            event.current_buffer.validate_and_handle()

        style = PTStyle.from_dict({
            "prompt": "bold cyan",
        })

        return PromptSession(
            multiline=True,
            key_bindings=kb,
            history=InMemoryHistory(),
            style=style,
        )

    # ── Утилиты вывода ────────────────────────────────────────────────────
    def _current_model_info(self) -> Optional[ModelInfo]:
        return self.config.get_model(self.current_model_id)

    def _current_model_label(self) -> str:
        m = self._current_model_info()
        return m.name if m else self.current_model_id

    def show_banner(self) -> None:
        """Стартовый баннер."""
        label = self._current_model_label()
        text = Text()
        text.append("🤖 AI Chat", style="bold cyan")
        text.append(f"  │  Модель: ", style="dim")
        text.append(label, style="bold green")
        text.append("\n")
        text.append("Введите /help для списка команд", style="dim")
        self.console.print(Panel(text, border_style="cyan"))

    def show_error(self, message: str, title: str = "Ошибка") -> None:
        """Показывает ошибку в красной панели."""
        self.console.print(Panel(
            Text(message, style="bold red"),
            title=title,
            border_style="red"
        ))

    def show_info(self, message: str) -> None:
        """Информационное сообщение dim-стилем."""
        self.console.print(Text(message, style="dim"))

    # ── Сохранение сессии ─────────────────────────────────────────────────
    def save_session(self) -> None:
        """Сохраняет текущую сессию, если в ней есть содержательные сообщения."""
        # Не сохраняем сессии без юзерских/ассистентских сообщений
        non_system = [m for m in self.session.messages if m.role != "system"]
        if not non_system:
            return
        try:
            self.history_mgr.save(self.session)
        except OSError as e:
            self.show_error(f"Не удалось сохранить сессию: {e}")

    # ── Главный цикл ──────────────────────────────────────────────────────
    def run(self) -> None:
        """Главный цикл REPL."""
        self.show_banner()

        # Предупреждение если модель не в списке
        if self._current_model_info() is None and self.config.models:
            self.show_info(
                f"Предупреждение: модель {self.current_model_id} не найдена в списке models. "
                "Стоимость не будет рассчитываться."
            )

        while True:
            try:
                user_input = self._read_input()
            except (KeyboardInterrupt, EOFError):
                self.console.print()
                self._exit_gracefully()
                return

            if user_input is None:
                continue

            stripped = user_input.strip()
            if not stripped:
                continue  # Пустой ввод — игнорируем

            # Команды
            if stripped.startswith("/"):
                should_continue = self._handle_command(stripped)
                if not should_continue:
                    return
                continue

            # Обычное сообщение
            self._send_message(user_input)

    def _read_input(self) -> Optional[str]:
        """Читает многострочный ввод от пользователя."""
        label = self._current_model_label()
        prompt_html = HTML(f'\n<prompt>[{label}] Вы ›</prompt> ')

        # Подсказка только при первом вводе
        if self.first_input:
            self.console.print(
                Text("(Alt+Enter — отправить, /help — справка)", style="dim italic")
            )

        try:
            text = self.input_session.prompt(prompt_html)
        except KeyboardInterrupt:
            raise
        except EOFError:
            raise

        self.first_input = False
        return text

    # ── Команды ───────────────────────────────────────────────────────────
    def _handle_command(self, cmd_line: str) -> bool:
        """
        Обработка команд. Возвращает False если нужно выйти из main loop.
        """
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/exit":
            self._exit_gracefully()
            return False
        elif cmd == "/help":
            self._cmd_help()
        elif cmd == "/new":
            self._cmd_new()
        elif cmd == "/history":
            self._cmd_history(arg)
        elif cmd == "/model":
            self._cmd_model(arg)
        elif cmd == "/file":
            self._cmd_file(arg)
        elif cmd == "/clear":
            self.console.clear()
            self.show_banner()
        elif cmd == "/usage":
            self._cmd_usage()
        else:
            self.show_info(f"Неизвестная команда: {cmd}. Введите /help для справки.")

        return True

    def _cmd_help(self) -> None:
        """Вывод справки по командам."""
        table = Table(title="Команды", border_style="cyan", show_lines=False)
        table.add_column("Команда", style="bold yellow", no_wrap=True)
        table.add_column("Описание", style="white")
        table.add_row("/new", "Начать новый разговор (текущий сохранится)")
        table.add_row("/history [all]", "Список сохранённых сессий (последние 20 / все)")
        table.add_row("/model", "Выбор модели из списка")
        table.add_row("/model <имя|№>", "Быстрое переключение модели")
        table.add_row("/file <путь>", "Отправить содержимое текстового файла")
        table.add_row("/clear", "Очистить экран (контекст сохранится)")
        table.add_row("/usage", "Статистика по текущей сессии")
        table.add_row("/help", "Эта справка")
        table.add_row("/exit", "Выход с сохранением сессии")
        self.console.print(table)
        self.console.print(
            Text(
                "\nВвод: Enter — новая строка, Alt+Enter — отправить сообщение.\n"
                "Прерывание генерации: Ctrl+C.",
                style="dim"
            )
        )

    def _cmd_new(self) -> None:
        """Сохранить текущую сессию и начать новую."""
        self.save_session()
        self.session = self._new_session()
        self.console.print(Text("Сессия сохранена. Новый разговор начат.", style="dim green"))

    def _cmd_history(self, arg: str) -> None:
        """Список сессий и загрузка выбранной."""
        sessions = self.history_mgr.list_sessions()
        if not sessions:
            self.show_info("История пуста.")
            return

        show_all = arg.strip().lower() == "all"
        display = sessions if show_all else sessions[:20]

        table = Table(title=f"Сохранённые сессии ({len(display)} из {len(sessions)})", border_style="cyan")
        table.add_column("№", style="bold yellow", justify="right")
        table.add_column("Дата", style="cyan")
        table.add_column("Модель", style="green")
        table.add_column("Первое сообщение", style="white")

        for idx, s in enumerate(display, 1):
            first_user = next((m.content for m in s.messages if m.role == "user"), "")
            preview = first_user.replace("\n", " ").strip()
            if len(preview) > 60:
                preview = preview[:57] + "..."
            try:
                dt = datetime.fromisoformat(s.created_at).strftime("%Y-%m-%d %H:%M")
            except ValueError:
                dt = s.created_at
            table.add_row(str(idx), dt, s.default_model or "—", preview or "(пусто)")

        self.console.print(table)

        try:
            choice = self.console.input("\n[dim]Номер для загрузки (Enter — отмена): [/dim]").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return

        if not choice:
            return

        try:
            idx = int(choice)
            if not 1 <= idx <= len(display):
                self.show_info("Неверный номер.")
                return
        except ValueError:
            self.show_info("Нужно ввести число.")
            return

        # Сохраняем текущую, загружаем выбранную
        self.save_session()
        self.session = display[idx - 1]
        if self.session.default_model:
            self.current_model_id = self.session.default_model
        self.console.print(Text(f"Загружена сессия от {self.session.created_at}", style="dim green"))
        self._render_loaded_session()

    def _render_loaded_session(self) -> None:
        """Краткий вывод загруженной сессии."""
        for m in self.session.messages:
            if m.role == "system":
                continue
            self.console.print(Rule(style="dim"))
            if m.role == "user":
                self.console.print(Text(f"Вы:", style="bold cyan"))
                self.console.print(m.content)
            elif m.role == "assistant":
                model_label = m.model or "Ассистент"
                self.console.print(Text(f"🤖 {model_label}", style="bold green"))
                self.console.print(Markdown(m.content))
        self.console.print(Rule(style="dim"))

    def _cmd_model(self, arg: str) -> None:
        """Переключение модели."""
        if not self.config.models:
            self.show_info("Список моделей пуст. Добавьте модели в config.json.")
            return

        if arg.strip():
            # Быстрое переключение
            query = arg.strip()
            # Сначала пытаемся как номер
            try:
                idx = int(query)
                if 1 <= idx <= len(self.config.models):
                    self._switch_model(self.config.models[idx - 1])
                    return
            except ValueError:
                pass

            matches = self.config.find_models(query)
            if len(matches) == 0:
                self.show_info(f"Модель не найдена: {query}")
            elif len(matches) == 1:
                self._switch_model(matches[0])
            else:
                self.show_info("Найдено несколько моделей, уточните:")
                for m in matches:
                    self.console.print(f"  • {m.name} ({m.id})")
            return

        # Интерактивный выбор
        table = Table(title="Доступные модели", border_style="cyan")
        table.add_column("№", style="bold yellow", justify="right")
        table.add_column("Имя", style="green")
        table.add_column("ID", style="dim")
        table.add_column("Контекст", justify="right")
        table.add_column("Thinking", justify="center")

        for i, m in enumerate(self.config.models, 1):
            marker = " ←" if m.id == self.current_model_id else ""
            table.add_row(
                str(i),
                m.name + marker,
                m.id,
                f"{m.context_window:,}".replace(",", " "),
                "✓" if m.supports_thinking else "—"
            )

        self.console.print(table)
        try:
            choice = self.console.input("\n[dim]Номер модели (Enter — отмена): [/dim]").strip()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return

        if not choice:
            return

        try:
            idx = int(choice)
            if not 1 <= idx <= len(self.config.models):
                self.show_info("Неверный номер.")
                return
        except ValueError:
            self.show_info("Нужно ввести число.")
            return

        self._switch_model(self.config.models[idx - 1])

    def _switch_model(self, model: ModelInfo) -> None:
        """Переключиться на указанную модель."""
        self.current_model_id = model.id
        self.console.print(Text(f"Модель переключена на {model.name}", style="dim green"))

    def _cmd_file(self, arg: str) -> None:
        """Отправка содержимого файла как сообщения."""
        if not arg.strip():
            self.show_info("Использование: /file <путь>")
            return

        path = Path(arg.strip().strip('"').strip("'"))
        if not path.exists():
            self.show_error(f"Файл не найден: {path}")
            return
        if not path.is_file():
            self.show_error(f"Не является файлом: {path}")
            return

        try:
            size = path.stat().st_size
        except OSError as e:
            self.show_error(f"Не удалось получить размер файла: {e}")
            return

        if size > MAX_FILE_SIZE:
            self.show_error(f"Файл слишком большой ({size} байт). Лимит: {MAX_FILE_SIZE} байт.")
            return

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            self.show_error(
                f"Не удалось прочитать файл: {e}. Убедитесь что файл в кодировке UTF-8."
            )
            return
        except OSError as e:
            self.show_error(f"Не удалось прочитать файл: {e}")
            return

        # Превью
        preview_lines = content.splitlines()[:3]
        preview = "\n".join(preview_lines)
        if len(content.splitlines()) > 3:
            preview += "\n..."

        self.console.print(Panel(
            preview,
            title=f"Превью: {path.name} ({size} байт)",
            border_style="cyan"
        ))

        try:
            answer = self.console.input("[dim]Отправить содержимое в чат? (y/n): [/dim]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return

        if answer not in ("y", "yes", "д", "да"):
            self.show_info("Отменено.")
            return

        message = f"Содержимое файла `{path.name}`:\n\n```\n{content}\n```"
        self._send_message(message)

    def _cmd_usage(self) -> None:
        """Статистика по сессии."""
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        cost_known = True
        requests = 0

        for m in self.session.messages:
            if m.role == "assistant" and m.usage:
                requests += 1
                total_prompt += m.usage.get("prompt_tokens", 0)
                total_completion += m.usage.get("completion_tokens", 0)
                model_info = self.config.get_model(m.model or "")
                c = calculate_cost(
                    m.usage.get("prompt_tokens", 0),
                    m.usage.get("completion_tokens", 0),
                    model_info
                )
                if c is None:
                    cost_known = False
                else:
                    total_cost += c

        table = Table(title="Статистика сессии", border_style="cyan")
        table.add_column("Метрика", style="dim")
        table.add_column("Значение", style="bold green", justify="right")
        table.add_row("Запросов", str(requests))
        table.add_row("Prompt токенов", f"{total_prompt:,}".replace(",", " "))
        table.add_row("Completion токенов", f"{total_completion:,}".replace(",", " "))
        table.add_row("Всего токенов", f"{total_prompt + total_completion:,}".replace(",", " "))
        if cost_known and requests > 0:
            table.add_row("Общая стоимость", f"{total_cost:.4f}₽")
            table.add_row("Средняя/запрос", f"{total_cost / requests:.4f}₽")
        else:
            table.add_row("Общая стоимость", "—")
        self.console.print(table)

    # ── Отправка сообщения и стриминг ─────────────────────────────────────
    def _send_message(self, content: str) -> None:
        """Отправка сообщения пользователя и получение ответа."""
        # Добавляем сообщение пользователя
        user_msg = Message(
            role="user",
            content=content,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        self.session.messages.append(user_msg)

        # Проверка контекста
        self._check_context_size()

        # Подготовка истории для API (без thinking, без служебных полей)
        api_messages = [m.to_api_dict() for m in self.session.messages]

        model_info = self._current_model_info()

        # Стриминг с retry
        result = self._stream_with_retry(api_messages, model_info)

        if result is None:
            # Окончательная неудача — откатываем добавленное сообщение пользователя
            # (но если это был /file — пользователь это понимает; оставим как есть)
            return

        # Сохраняем ответ ассистента
        assistant_msg = Message(
            role="assistant",
            content=result.content,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            model=self.current_model_id,
            thinking=result.thinking if result.thinking else None,
            usage={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            interrupted=result.interrupted,
        )
        self.session.messages.append(assistant_msg)

        # Автосохранение
        self.save_session()

    def _stream_with_retry(
        self,
        api_messages: list[dict[str, str]],
        model_info: Optional[ModelInfo],
    ) -> Optional[StreamResult]:
        """Выполняет стриминг с retry при сетевых ошибках."""
        last_error: Optional[str] = None

        for attempt in range(MAX_RETRIES):
            try:
                return self._do_stream(api_messages, model_info)
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                if status == 401:
                    self.show_error("Неверный API-ключ. Проверьте поле api_key в config.json")
                    return None
                elif status == 429:
                    # Rate limit — ждём
                    retry_after = self._parse_retry_after(e)
                    self.show_info(f"Превышен лимит. Повтор через {retry_after}с...")
                    try:
                        time.sleep(retry_after)
                    except KeyboardInterrupt:
                        return None
                    continue
                elif status == 400:
                    # Возможно — превышен контекст
                    msg = str(e).lower()
                    if "context" in msg or "length" in msg or "too long" in msg or "maximum" in msg:
                        return self._handle_context_overflow(api_messages, model_info)
                    self.show_error(f"Ошибка запроса (400): {e}")
                    return None
                elif status and status >= 500:
                    last_error = f"Ошибка сервера ({status})"
                    self.show_info(f"{last_error}. Повтор...")
                else:
                    self.show_error(f"Ошибка API ({status}): {e}")
                    return None
            except (APIConnectionError, APITimeoutError) as e:
                last_error = f"Сетевая ошибка: {e}"
            except RateLimitError as e:
                self.show_info(f"Превышен лимит запросов. Повтор...")
                last_error = str(e)
            except Exception as e:
                if self.debug:
                    traceback.print_exc(file=sys.stderr)
                self.show_error(f"Непредвиденная ошибка: {e}")
                return None

            # Пауза перед следующей попыткой
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                self.show_info(f"Повторная попытка {attempt + 2}/{MAX_RETRIES} через {delay}с...")
                try:
                    time.sleep(delay)
                except KeyboardInterrupt:
                    return None

        self.show_error(f"Не удалось выполнить запрос после {MAX_RETRIES} попыток. {last_error or ''}")
        return None

    def _parse_retry_after(self, error: APIStatusError) -> int:
        """Извлекает Retry-After из заголовков ответа."""
        try:
            response = getattr(error, "response", None)
            if response is not None:
                ra = response.headers.get("retry-after") or response.headers.get("Retry-After")
                if ra:
                    return int(float(ra))
        except (AttributeError, ValueError, TypeError):
            pass
        return 5  # Дефолт

    def _handle_context_overflow(
        self,
        api_messages: list[dict[str, str]],
        model_info: Optional[ModelInfo],
    ) -> Optional[StreamResult]:
        """Обработка ошибки превышения контекста."""
        window = model_info.context_window if model_info else 128000
        self.show_error(
            f"Превышен контекст модели ({window} токенов). Используйте /new для нового разговора."
        )
        try:
            answer = self.console.input(
                "[dim]Удалить старые сообщения и повторить? (y/n): [/dim]"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None

        if answer not in ("y", "yes", "д", "да"):
            return None

        # Удаляем самые старые non-system сообщения (по парам)
        trimmed = False
        while len(self.session.messages) > 3:  # Оставляем хотя бы system + последний user
            # Найти первый non-system и удалить его
            for i, m in enumerate(self.session.messages):
                if m.role != "system":
                    del self.session.messages[i]
                    trimmed = True
                    break
            # Удалим ещё одно сообщение (по парам user/assistant)
            for i, m in enumerate(self.session.messages):
                if m.role != "system":
                    del self.session.messages[i]
                    break

            # Пересобираем и пробуем снова
            new_api_messages = [m.to_api_dict() for m in self.session.messages]
            if estimate_tokens("\n".join(x.get("content", "") for x in new_api_messages)) < window * 0.7:
                api_messages = new_api_messages
                break

        if not trimmed:
            return None

        self.show_info("Старые сообщения удалены. Повторный запрос...")
        try:
            return self._do_stream(api_messages, model_info)
        except Exception as e:
            self.show_error(f"Повторный запрос не удался: {e}")
            return None

    def _do_stream(
        self,
        api_messages: list[dict[str, str]],
        model_info: Optional[ModelInfo],
    ) -> StreamResult:
        """Запускает стриминг с Live-обновлением UI."""
        self.console.print(Rule(style="dim"))

        # Состояние для рендеринга
        state = {
            "content": "",
            "thinking": "",
            "completion_tokens": 0,
            "prompt_tokens": 0,
        }
        start_ts = time.monotonic()
        model_label = self._current_model_label()
        supports_thinking = model_info.supports_thinking if model_info else False

        def render() -> Group:
            """Собирает текущее представление ответа."""
            blocks = []

            # Блок thinking
            if state["thinking"]:
                blocks.append(Text("💭 Размышляю...", style="bold dim"))
                blocks.append(Text(state["thinking"], style="dim italic"))
                blocks.append(Text(""))  # пустая строка

            # Блок ответа
            if state["content"] or not state["thinking"]:
                blocks.append(Text(f"🤖 {model_label}", style="bold green"))
                blocks.append(Text(""))
                if state["content"]:
                    blocks.append(Markdown(state["content"]))

            # Статистика
            elapsed = time.monotonic() - start_ts
            up = state["prompt_tokens"]
            down = state["completion_tokens"]
            cost = calculate_cost(up, down, model_info)
            up_str = f"↑{up}" if up else "↑?"
            down_str = f"↓{down}..."

            if cost is not None:
                cost_str = f"~{cost:.4f}₽"
            else:
                cost_str = "—"

            stats = Text(
                f"\n ⏱ Токены: {up_str} {down_str}  │  💰 {cost_str}  │  ⏳ {elapsed:.1f}с",
                style="dim cyan"
            )
            blocks.append(stats)

            return Group(*blocks)

        completion_chunks = 0
        interrupted = False
        result: Optional[StreamResult] = None

        try:
            with Live(render(), console=self.console, refresh_per_second=12, transient=False) as live:
                def on_thinking(text: str) -> None:
                    state["thinking"] += text
                    live.update(render())

                def on_content(text: str) -> None:
                    nonlocal completion_chunks
                    state["content"] += text
                    completion_chunks += 1
                    # Грубая оценка во время стриминга
                    state["completion_tokens"] = estimate_tokens(state["content"] + state["thinking"])
                    live.update(render())

                # Запускаем стрим
                try:
                    result = self.client.stream_completion(
                        model=self.current_model_id,
                        messages=api_messages,
                        on_thinking=on_thinking,
                        on_content=on_content,
                    )
                    # Обновляем точные токены если пришли в usage
                    state["prompt_tokens"] = result.prompt_tokens
                    state["completion_tokens"] = result.completion_tokens
                    interrupted = result.interrupted
                except KeyboardInterrupt:
                    interrupted = True
                    result = StreamResult(
                        content=state["content"],
                        thinking=state["thinking"],
                        prompt_tokens=estimate_tokens("\n".join(m.get("content", "") for m in api_messages)),
                        completion_tokens=state["completion_tokens"],
                        reasoning_tokens=0,
                        elapsed=time.monotonic() - start_ts,
                        interrupted=True,
                    )

                # Финальный рендер
                live.update(self._render_final(result, model_info, model_label))

        finally:
            self.console.print(Rule(style="dim"))

        if interrupted:
            self.show_info("Генерация прервана.")

        return result  # type: ignore

    def _render_final(
        self,
        result: StreamResult,
        model_info: Optional[ModelInfo],
        model_label: str,
    ) -> Group:
        """Финальный рендер ответа (без ~ и ...)."""
        blocks = []

        if result.thinking:
            blocks.append(Text("💭 Размышления", style="bold dim"))
            blocks.append(Text(result.thinking, style="dim italic"))
            blocks.append(Text(""))

        blocks.append(Text(f"🤖 {model_label}", style="bold green"))
        blocks.append(Text(""))
        if result.content:
            blocks.append(Markdown(result.content))
        else:
            blocks.append(Text("(пустой ответ)", style="dim"))

        cost = calculate_cost(result.prompt_tokens, result.completion_tokens, model_info)
        if cost is not None:
            cost_str = f"{cost:.4f}₽"
        else:
            cost_str = "—"

        reasoning_part = f" (💭{result.reasoning_tokens})" if result.reasoning_tokens else ""
        interrupted_part = " [прервано]" if result.interrupted else ""

        stats = Text(
            f"\n ⏱ Токены: ↑{result.prompt_tokens} ↓{result.completion_tokens}{reasoning_part}"
            f"  │  💰 {cost_str}  │  ⏳ {result.elapsed:.1f}с{interrupted_part}",
            style="dim cyan"
        )
        blocks.append(stats)
        return Group(*blocks)

    # ── Проверка контекста ────────────────────────────────────────────────
    def _check_context_size(self) -> None:
        """Предупреждает если контекст близок к лимиту."""
        model_info = self._current_model_info()
        if model_info is None:
            return
        total_text = "\n".join(m.content for m in self.session.messages)
        tokens = estimate_tokens(total_text)
        window = model_info.context_window
        ratio = tokens / window if window > 0 else 0

        if ratio > 0.8:
            self.console.print(Text(
                f"Внимание: использовано ~{int(ratio * 100)}% контекста ({tokens}/{window})",
                style="bold yellow"
            ))

    # ── Выход ─────────────────────────────────────────────────────────────
    def _exit_gracefully(self) -> None:
        """Сохранение сессии и выход."""
        self.save_session()
        self.console.print(Text("До свидания!", style="bold cyan"))


# ──────────────────────────────────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Точка входа в приложение."""
    debug = "--debug" in sys.argv

    console = Console()

    # Создание конфига при первом запуске
    if Config.ensure_exists(CONFIG_PATH):
        console.print(Panel(
            Text(
                f"Конфиг создан: {CONFIG_PATH}\n"
                "Укажите api_key и запустите снова.",
                style="bold yellow"
            ),
            title="Первый запуск",
            border_style="yellow"
        ))
        sys.exit(0)

    # Загрузка и валидация конфига
    try:
        config = Config.load(CONFIG_PATH)
    except ConfigError as e:
        console.print(Panel(
            Text(f"Ошибка конфига: {e}", style="bold red"),
            title="Ошибка",
            border_style="red"
        ))
        sys.exit(1)

    # Запуск UI
    try:
        ui = ChatUI(config, debug=debug)
        ui.run()
    except Exception as e:
        if debug:
            traceback.print_exc(file=sys.stderr)
        console.print(Panel(
            Text(f"Фатальная ошибка: {e}", style="bold red"),
            title="Ошибка",
            border_style="red"
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
