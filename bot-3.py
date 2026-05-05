import asyncio
import logging
import os
from typing import Any, Callable, Awaitable

from dotenv import load_dotenv
import asyncpg
from openai import AsyncOpenAI

from aiogram import Bot, Dispatcher, F, Router, BaseMiddleware
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
    TelegramObject,
)

# ── загрузка .env ─────────────────────────────────────────────────────────────

load_dotenv()

BOT_TOKEN    = os.environ["BOT_TOKEN"]
DEEPSEEK_KEY = os.environ["DEEPSEEK_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
ADMIN_IDS: list[int] = [
    int(x.strip()) for x in os.environ["ADMIN_IDS"].split(",") if x.strip()
]

# ── константы ─────────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "deepseek-chat":     "DeepSeek V3",
    "deepseek-reasoner": "DeepSeek R1",
}
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "Ты умный и полезный ассистент. Отвечай чётко, структурированно и по делу.\n"
    "Если вопрос требует рассуждений — думай пошагово, но не усложняй без необходимости.\n"
    "Общайся на том языке, на котором к тебе обращаются."
)

# ── логирование ───────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── DeepSeek клиент ───────────────────────────────────────────────────────────

ai_client = AsyncOpenAI(
    api_key=DEEPSEEK_KEY,
    base_url="https://api.deepseek.com",
)

# ── FSM ───────────────────────────────────────────────────────────────────────

class BroadcastState(StatesGroup):
    waiting_text = State()

# ── база данных ───────────────────────────────────────────────────────────────

class Database:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def create(cls, dsn: str) -> "Database":
        pool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)
        db = cls(pool)
        await db._init_db()
        return db

    async def _init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id    BIGINT PRIMARY KEY,
                    username   TEXT      DEFAULT '',
                    full_name  TEXT      DEFAULT '',
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id        BIGINT PRIMARY KEY REFERENCES users(user_id),
                    memory_enabled BOOLEAN DEFAULT TRUE,
                    model          TEXT    DEFAULT 'deepseek-chat'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id         BIGSERIAL PRIMARY KEY,
                    user_id    BIGINT    NOT NULL REFERENCES users(user_id),
                    role       TEXT      NOT NULL,
                    content    TEXT      NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)

    async def upsert_user(self, user_id: int, username: str, full_name: str):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (user_id, username, full_name)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO UPDATE SET
                    username  = EXCLUDED.username,
                    full_name = EXCLUDED.full_name
            """, user_id, username, full_name)
            await conn.execute("""
                INSERT INTO user_settings (user_id) VALUES ($1)
                ON CONFLICT (user_id) DO NOTHING
            """, user_id)

    async def get_all_users(self) -> list[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT user_id, username, full_name FROM users ORDER BY created_at DESC"
            )

    async def get_user_settings(self, user_id: int) -> dict:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT memory_enabled, model FROM user_settings WHERE user_id = $1",
                user_id,
            )
        if row is None:
            return {"memory_enabled": True, "model": DEFAULT_MODEL}
        return {
            "memory_enabled": bool(row["memory_enabled"]),
            "model": row["model"] or DEFAULT_MODEL,
        }

    async def set_user_setting(self, user_id: int, key: str, value: Any):
        if key not in {"memory_enabled", "model"}:
            raise ValueError(f"Unknown key: {key}")
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"UPDATE user_settings SET {key} = $1 WHERE user_id = $2",
                value, user_id,
            )

    async def get_history(self, user_id: int, limit: int = 40) -> list[asyncpg.Record]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content FROM history
                WHERE user_id = $1
                ORDER BY id DESC LIMIT $2
            """, user_id, limit)
        return list(reversed(rows))

    async def add_message(self, user_id: int, role: str, content: str):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO history (user_id, role, content) VALUES ($1, $2, $3)",
                user_id, role, content,
            )

    async def clear_history(self, user_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM history WHERE user_id = $1", user_id)

    async def get_stats(self) -> dict:
        async with self.pool.acquire() as conn:
            users_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            msgs_count  = await conn.fetchval("SELECT COUNT(*) FROM history")
        return {"users": users_count, "messages": msgs_count}


# ── middleware ────────────────────────────────────────────────────────────────

class DbMiddleware(BaseMiddleware):
    def __init__(self, db: Database):
        self.db = db

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict], Awaitable[Any]],
        event: TelegramObject,
        data: dict,
    ) -> Any:
        data["db"] = self.db
        return await handler(event, data)


# ── клавиатуры ────────────────────────────────────────────────────────────────

def kb_settings(memory_enabled: bool) -> InlineKeyboardMarkup:
    mem_label = "🟢 Память: ВКЛ" if memory_enabled else "🔴 Память: ВЫКЛ"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=mem_label,             callback_data="settings:toggle_memory")],
        [InlineKeyboardButton(text="🤖 Выбор модели",     callback_data="settings:model")],
        [InlineKeyboardButton(text="🗑 Очистить историю", callback_data="settings:clear")],
        [InlineKeyboardButton(text="❌ Закрыть",          callback_data="settings:close")],
    ])

def kb_models(current_model: str) -> InlineKeyboardMarkup:
    rows = []
    for mid, mname in AVAILABLE_MODELS.items():
        mark = "✅ " if mid == current_model else ""
        rows.append([InlineKeyboardButton(text=f"{mark}{mname}", callback_data=f"model:{mid}")])
    rows.append([InlineKeyboardButton(text="◀️ Назад", callback_data="settings:back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👥 Список пользователей", callback_data="admin:users")],
        [InlineKeyboardButton(text="📊 Статистика",           callback_data="admin:stats")],
        [InlineKeyboardButton(text="📢 Рассылка",             callback_data="admin:broadcast")],
        [InlineKeyboardButton(text="❌ Закрыть",              callback_data="admin:close")],
    ])

def kb_back_admin() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="◀️ Назад", callback_data="admin:back")]
    ])


# ── утилиты ───────────────────────────────────────────────────────────────────

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

async def send_long(message: Message, text: str, **kwargs):
    for i in range(0, len(text), 4000):
        await message.answer(text[i:i + 4000], **kwargs)


# ── роутер ────────────────────────────────────────────────────────────────────

router = Router()

# ── команды ───────────────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message, db: Database):
    u = message.from_user
    await db.upsert_user(u.id, u.username or "", u.full_name or "")
    await message.answer(
        f"👋 Привет, *{u.first_name}*!\n\n"
        "Я AI-ассистент на базе DeepSeek. Просто напиши мне что-нибудь и я отвечу.\n\n"
        "⚙️ /settings — настройки\n"
        "🗑 /clear — очистить историю диалога\n"
        "ℹ️ /help — помощь",
        parse_mode=ParseMode.MARKDOWN,
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📖 *Справка*\n\n"
        "• Просто пишите сообщение — бот ответит через DeepSeek.\n"
        "• /settings — включить/выключить память, сменить модель.\n"
        "• /clear — сбросить контекст разговора.\n"
        "• /model — быстро сменить модель.\n\n"
        "🧠 *Память* — когда включена, бот помнит историю диалога.\n"
        "🤖 *Модели*:\n"
        "  • DeepSeek V3 — быстрый, для общих задач\n"
        "  • DeepSeek R1 — аналитический",
        parse_mode=ParseMode.MARKDOWN,
    )

@router.message(Command("settings"))
async def cmd_settings(message: Message, db: Database):
    s = await db.get_user_settings(message.from_user.id)
    await message.answer(
        "⚙️ *Настройки*\n\nВыберите параметр для изменения:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_settings(s["memory_enabled"]),
    )

@router.message(Command("model"))
async def cmd_model(message: Message, db: Database):
    s = await db.get_user_settings(message.from_user.id)
    await message.answer(
        "🤖 *Выберите модель:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_models(s.get("model", DEFAULT_MODEL)),
    )

@router.message(Command("clear"))
async def cmd_clear(message: Message, db: Database):
    await db.clear_history(message.from_user.id)
    await message.answer("🗑 История диалога очищена!")

@router.message(Command("admin"))
async def cmd_admin(message: Message):
    if not is_admin(message.from_user.id):
        await message.answer("⛔ У вас нет доступа к этой команде.")
        return
    await message.answer(
        "🛠 *Панель администратора*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_admin(),
    )

@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer("❌ Действие отменено.")


# ── колбэки: настройки ────────────────────────────────────────────────────────

@router.callback_query(F.data == "settings:toggle_memory")
async def cb_toggle_memory(call: CallbackQuery, db: Database):
    uid = call.from_user.id
    s = await db.get_user_settings(uid)
    new_val = not s["memory_enabled"]
    await db.set_user_setting(uid, "memory_enabled", new_val)
    label = "включена 🟢" if new_val else "выключена 🔴"
    await call.message.edit_text(
        f"⚙️ *Настройки*\n\nПамять {label}.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_settings(new_val),
    )
    await call.answer()

@router.callback_query(F.data == "settings:clear")
async def cb_settings_clear(call: CallbackQuery, db: Database):
    uid = call.from_user.id
    await db.clear_history(uid)
    s = await db.get_user_settings(uid)
    await call.message.edit_text(
        "⚙️ *Настройки*\n\n✅ История очищена!",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_settings(s["memory_enabled"]),
    )
    await call.answer()

@router.callback_query(F.data == "settings:model")
async def cb_settings_model(call: CallbackQuery, db: Database):
    s = await db.get_user_settings(call.from_user.id)
    await call.message.edit_text(
        "🤖 *Выберите модель:*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_models(s.get("model", DEFAULT_MODEL)),
    )
    await call.answer()

@router.callback_query(F.data == "settings:back")
async def cb_settings_back(call: CallbackQuery, db: Database):
    s = await db.get_user_settings(call.from_user.id)
    await call.message.edit_text(
        "⚙️ *Настройки*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_settings(s["memory_enabled"]),
    )
    await call.answer()

@router.callback_query(F.data == "settings:close")
async def cb_settings_close(call: CallbackQuery):
    await call.message.delete()
    await call.answer()

@router.callback_query(F.data.startswith("model:"))
async def cb_model_select(call: CallbackQuery, db: Database):
    model_id = call.data.split(":", 1)[1]
    if model_id not in AVAILABLE_MODELS:
        await call.answer("Неизвестная модель.")
        return
    await db.set_user_setting(call.from_user.id, "model", model_id)
    model_name = AVAILABLE_MODELS[model_id]
    await call.message.edit_text(
        f"✅ Модель изменена на *{model_name}*",
        parse_mode=ParseMode.MARKDOWN,
    )
    await call.answer()


# ── колбэки: администратор ────────────────────────────────────────────────────

@router.callback_query(F.data == "admin:users")
async def cb_admin_users(call: CallbackQuery, db: Database):
    if not is_admin(call.from_user.id):
        await call.answer("⛔ Нет доступа.")
        return
    users = await db.get_all_users()
    lines = [f"👥 *Пользователи ({len(users)}):*\n"]
    for u in users[:50]:
        uname = f"@{u['username']}" if u["username"] else "—"
        lines.append(f"• {u['full_name']} ({uname}) — ID: `{u['user_id']}`")
    if len(users) > 50:
        lines.append(f"\n_...и ещё {len(users) - 50}_")
    await call.message.edit_text(
        "\n".join(lines),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_back_admin(),
    )
    await call.answer()

@router.callback_query(F.data == "admin:stats")
async def cb_admin_stats(call: CallbackQuery, db: Database):
    if not is_admin(call.from_user.id):
        await call.answer("⛔ Нет доступа.")
        return
    stats = await db.get_stats()
    await call.message.edit_text(
        f"📊 *Статистика*\n\n"
        f"👥 Пользователей: *{stats['users']}*\n"
        f"💬 Сообщений всего: *{stats['messages']}*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_back_admin(),
    )
    await call.answer()

@router.callback_query(F.data == "admin:broadcast")
async def cb_admin_broadcast(call: CallbackQuery, state: FSMContext):
    if not is_admin(call.from_user.id):
        await call.answer("⛔ Нет доступа.")
        return
    await state.set_state(BroadcastState.waiting_text)
    await call.message.edit_text(
        "📢 Введите сообщение для рассылки всем пользователям:\n\n_(отправьте /cancel для отмены)_",
        parse_mode=ParseMode.MARKDOWN,
    )
    await call.answer()

@router.callback_query(F.data == "admin:back")
async def cb_admin_back(call: CallbackQuery):
    if not is_admin(call.from_user.id):
        await call.answer("⛔ Нет доступа.")
        return
    await call.message.edit_text(
        "🛠 *Панель администратора*",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_admin(),
    )
    await call.answer()

@router.callback_query(F.data == "admin:close")
async def cb_admin_close(call: CallbackQuery):
    await call.message.delete()
    await call.answer()


# ── FSM: рассылка ─────────────────────────────────────────────────────────────

@router.message(BroadcastState.waiting_text)
async def fsm_broadcast(message: Message, state: FSMContext, bot: Bot, db: Database):
    await state.clear()
    if not is_admin(message.from_user.id):
        return
    users = await db.get_all_users()
    sent, failed = 0, 0
    status = await message.answer("📢 Начинаю рассылку...")
    for u in users:
        try:
            await bot.send_message(u["user_id"], message.text)
            sent += 1
            await asyncio.sleep(0.05)
        except Exception:
            failed += 1
    await status.edit_text(
        f"✅ Рассылка завершена!\n\nОтправлено: {sent}\nОшибок: {failed}"
    )


# ── обработчик сообщений ──────────────────────────────────────────────────────

@router.message(F.text)
async def handle_message(message: Message, bot: Bot, db: Database):
    u = message.from_user
    user_text = message.text or ""

    await db.upsert_user(u.id, u.username or "", u.full_name or "")

    s = await db.get_user_settings(u.id)
    memory_enabled = s["memory_enabled"]
    model_id = s.get("model", DEFAULT_MODEL)

    history = await db.get_history(u.id) if memory_enabled else []
    is_first = len(history) == 0

    await db.add_message(u.id, "user", user_text)
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    thinking_msg: Message | None = None
    if is_first:
        thinking_msg = await message.answer("⏳ Инициализирую контекст, подождите немного...")

    try:
        messages_for_api: list[dict] = []

        if is_first:
            messages_for_api.append({"role": "system", "content": SYSTEM_PROMPT})

        for msg in history[-20:]:
            messages_for_api.append({"role": msg["role"], "content": msg["content"]})

        messages_for_api.append({"role": "user", "content": user_text})

        response = await ai_client.chat.completions.create(
            model=model_id,
            messages=messages_for_api,
            max_tokens=4096,
        )

        response_text = (response.choices[0].message.content or "").strip()
        if not response_text:
            response_text = "⚠️ Пустой ответ от модели. Попробуйте ещё раз."

        await db.add_message(u.id, "assistant", response_text)

        if thinking_msg:
            await thinking_msg.edit_text("✅ Готов! Вот мой ответ:")

        model_label = AVAILABLE_MODELS.get(model_id, model_id)
        footer = f"\n\n_🤖 {model_label}_"
        full_text = response_text + footer

        if len(full_text) > 4096:
            await send_long(message, response_text, parse_mode=ParseMode.MARKDOWN)
        else:
            await message.answer(full_text, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения: {e}")
        err_msg = f"⚠️ Произошла ошибка: {e}"
        if thinking_msg:
            await thinking_msg.edit_text(err_msg)
        else:
            await message.answer(err_msg)


# ── точка входа ───────────────────────────────────────────────────────────────

async def main():
    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )

    db = await Database.create(DATABASE_URL)

    dp = Dispatcher(storage=MemoryStorage())
    dp.message.middleware(DbMiddleware(db))
    dp.callback_query.middleware(DbMiddleware(db))
    dp.include_router(router)

    await bot.set_my_commands([
        BotCommand(command="start",    description="Начать"),
        BotCommand(command="help",     description="Помощь"),
        BotCommand(command="settings", description="Настройки"),
        BotCommand(command="model",    description="Сменить модель"),
        BotCommand(command="clear",    description="Очистить историю"),
        BotCommand(command="admin",    description="Панель администратора"),
        BotCommand(command="cancel",   description="Отмена"),
    ])

    logger.info("Бот запущен!")
    await dp.start_polling(bot, drop_pending_updates=True)


if __name__ == "__main__":
    asyncio.run(main())
