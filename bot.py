#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import asyncio
import aiohttp
import json
import signal
import datetime
import importlib
import time
from typing import Dict, List, Optional, Set, Union, Any

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)

# Импортируем библиотеки для работы с Parquet
# Import libraries for working with Parquet
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logging.warning("Библиотеки pandas и pyarrow не установлены. Сохранение в формате Parquet недоступно.")
    logging.warning("Libraries pandas and pyarrow are not installed. Saving in Parquet format is not available.")

from llm_client import LocalLLMClient, check_ollama_availability

# Загрузка переменных окружения перед настройкой логирования
load_dotenv(override=True)

# Настройка логирования
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Сбрасываем базовую конфигурацию логирования
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=log_level
)
logger = logging.getLogger(__name__)
logger.info(f"Уровень логирования установлен на: {logging.getLevelName(log_level)}")

# Устанавливаем уровень логирования для всех логгеров
logging.getLogger("llm_client").setLevel(log_level)
logging.getLogger("check_ollama").setLevel(log_level)

# Отключаем логирование HTTP запросов с токенами
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)

# Получаем переменные окружения
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama2")

# Пути к файлам для хранения данных
DATA_DIR = "data"
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
GROUP_PROMPTS_FILE = os.path.join(DATA_DIR, "group_prompts.json")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
MODELS_FILE = os.path.join(DATA_DIR, "models.json")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
TEMPERATURES_FILE = os.path.join(DATA_DIR, "temperatures.json")  # Файл для хранения индивидуальных настроек температуры

# Пути для Parquet файлов
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
MESSAGES_PARQUET = os.path.join(PARQUET_DIR, "messages.parquet")

# Создаем директории для данных, если они не существуют
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

# Хранилище сообщений для суммаризации
# Формат: {chat_id/user_id: [{"sender": "name", "text": "message", "reply_to_index": None or int, "message_id": int}, ...]}
user_messages: Dict[int, List[Dict[str, Any]]] = {}

# Хранилище выбранных моделей для пользователей и чатов
user_models: Dict[int, str] = {}
chat_models: Dict[int, str] = {}

# Хранилище для отслеживания пользователей в фактическом режиме
factual_mode_users: Set[int] = set()

# Хранилище индивидуальных настроек температуры для чатов
chat_temperatures: Dict[int, float] = {}  # Обычная температура для чатов
chat_factual_temperatures: Dict[int, float] = {}  # Фактическая температура для чатов
chat_summary_temperatures: Dict[int, float] = {}  # Температура суммаризации для чатов

# Хранилище пользовательских промптов для личных чатов
chat_prompts: Dict[str, Dict[int, str]] = {
    "answer": {},  # Промпты для ответов на вопросы
    "summary": {}  # Промпты для суммаризации
}

# Хранилище промптов для групповых чатов
group_prompts: Dict[str, Dict[int, str]] = {
    "answer": {},  # Промпты для ответов на вопросы
    "summary": {}  # Промпты для суммаризации
}

# Настройки бота
bot_config = {
    "context_messages": 100,  # Количество сообщений в контексте по умолчанию
    "auto_save_interval": 5, # Интервал автосохранения в минутах
    "temperature": 0.9,     # Температура генерации (от 0.1 до 1.0)
    "factual_temperature": 0.5,  # Пониженная температура для фактических вопросов
    "summary_temperature": 0.7,  # Температура для саммаризации (ниже, чтобы саммари было более точным)
    "summary_messages": 200   # Максимальное количество сообщений для суммаризации
}

# Промпты по умолчанию
# Default prompts
DEFAULT_ANSWER_PROMPT = ("""
Ты - дружелюбный и полезный ассистент Василиса. Ты помогаешь пользователям в чате, отвечая на их вопросы и помогая решать задачи.
Твой стиль общения - дружелюбный, информативный и полезный. Ты можешь использовать легкий юмор, когда это уместно, но всегда остаешься вежливым и полезным.

Ты называешь себя Василиса, в честь Василисы Премудрой из русских сказок. Ты можешь иногда использовать метафоры из сказок, когда это уместно.

Твои сообщения сразу отправляются в чат, поэтому отправляй финальный вариант без кавычек и без упоминания своих инструкций.

Вопрос: {question}

Ответ:
""")

DEFAULT_SUMMARY_PROMPT = """
Ты - дружелюбный и полезный ассистент Василиса. Ты помогаешь пользователям в чате, создавая краткие и информативные саммари бесед.
Твой стиль общения - дружелюбный, информативный и полезный. Ты можешь использовать легкий юмор, когда это уместно, но всегда остаешься вежливым и полезным.

Ты называешь себя Василиса, в честь Василисы Премудрой из русских сказок. Ты можешь иногда использовать метафоры из сказок, когда это уместно.

Твои сообщения сразу отправляются в чат, поэтому отправляй финальный вариант без кавычек и без упоминания своих инструкций.

Пожалуйста, сделай краткое и информативное саммари следующего текста:

{text}

Саммари:
"""

# Инициализация LLM клиента
llm_client = LocalLLMClient(model=DEFAULT_MODEL)

# Состояния для ConversationHandler
CHOOSING_MODEL = 1
SETTING_PROMPT = 2
CHOOSING_PROMPT_TYPE = 3
SETTING_CONFIG = 4
CHOOSING_CONFIG = 5
CHOOSING_CHAT = 6
ENTERING_CHAT_ID = 7
CHOOSING_MODEL_TARGET = 8
CHOOSING_TEMP_TYPE = 9
CHOOSING_CHAT_FOR_TEMP = 10
ENTERING_CHAT_ID_FOR_TEMP = 11
SETTING_TEMPERATURE = 12


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    user = update.effective_user
    user_id = user.id
    chat_id = update.effective_chat.id

    # Устанавливаем модель по умолчанию для нового пользователя/чата
    if user_id not in user_models:
        user_models[user_id] = DEFAULT_MODEL

    if chat_id < 0:  # Групповой чат
        if chat_id not in chat_models:
            chat_models[chat_id] = DEFAULT_MODEL
        current_model = chat_models[chat_id]
    else:  # Личный чат
        current_model = user_models[user_id]

    await update.message.reply_html(
        f"Привет, {user.mention_html()}! Я бот, который может отвечать на вопросы и делать саммари.\n\n"
        f"Текущая модель: <b>{current_model}</b>\n\n"
        f"Используйте /help для получения списка команд."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Устанавливаем модель по умолчанию для нового пользователя/чата
    if user_id not in user_models:
        user_models[user_id] = DEFAULT_MODEL

    if chat_id < 0:  # Групповой чат
        if chat_id not in chat_models:
            chat_models[chat_id] = DEFAULT_MODEL
        current_model = chat_models[chat_id]

        # Проверяем, есть ли пользовательские промпты для этого чата
        has_custom_answer_prompt = chat_id in chat_prompts["answer"]
        has_custom_summary_prompt = chat_id in chat_prompts["summary"]

        help_text = (
            "Доступные команды:\n\n"
            "/start - Начать взаимодействие с ботом\n"
            "/help - Показать это сообщение\n"
            "/summarize - Создать саммари из последних сообщений\n"
            "/factual - Переключить режим точных ответов (уменьшает галлюцинации)\n"
            "/chatid - Показать ID текущего чата\n\n"
            f"Текущая модель: {current_model}\n"
            f"Пользовательский промпт для ответов: {'✅' if has_custom_answer_prompt else '❌'}\n"
            f"Пользовательский промпт для саммари: {'✅' if has_custom_summary_prompt else '❌'}\n"
            f"Сообщений в контексте: {bot_config['context_messages']}\n\n"
            "Чтобы обратиться к боту в групповом чате, используйте один из способов:\n"
            f"1. Упомяните бота: @{context.bot.username} ваш вопрос\n"
            "2. Используйте префикс: !ваш вопрос\n"
            "3. Ответьте на сообщение бота\n\n"
            "Настройка промптов и моделей доступна через личные сообщения с ботом."
        )
    else:  # Личный чат
        current_model = user_models[user_id]

        # Проверяем, находится ли пользователь в фактическом режиме
        factual_status = "✅ Включен" if user_id in factual_mode_users else "❌ Выключен"
        
        help_text = (
            "Доступные команды:\n\n"
            "/start - Начать взаимодействие с ботом\n"
            "/help - Показать это сообщение\n"
            "/summarize - Создать саммари из последних сообщений\n"
            "/hidden_summary [ID чата] - Создать скрытое саммари из группового чата\n"
            "/hidden_query [ID чата] [запрос] - Отправить запрос от имени группового чата\n"
            "/hidden_query [ID чата] [запрос] - Отправить запрос от имени группового чата\n"
            "/factual - Переключить режим точных ответов (уменьшает галлюцинации)\n"
            "/chatid - Показать подробную информацию о чате\n"
            "/model - Выбрать модель для себя или группового чата\n"
            "/prompt - Настроить промпты для себя или группового чата\n"
            "/config - Настроить глобальные параметры бота\n"
            "/temperature - Настроить температуру для конкретного чата\n"
            "/reload [модуль] - Перезагрузить модули (для разработчиков)\n\n"
            f"Текущая модель: {current_model}\n"
            f"Сообщений в контексте: {bot_config['context_messages']}\n"
            f"Фактический режим: {factual_status}\n\n"
            "В личном чате я отвечаю на все сообщения. Просто напишите мне что-нибудь!"
        )

    await update.message.reply_text(help_text)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик входящих сообщений."""
    # Проверяем, что это обычное сообщение, а не редактирование
    if not update.message:
        return
        
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    message_text = update.message.text

    # Устанавливаем модель по умолчанию, если пользователь/чат еще не выбрал
    if user_id not in user_models:
        user_models[user_id] = DEFAULT_MODEL

    if chat_id < 0 and chat_id not in chat_models:  # Групповой чат
        chat_models[chat_id] = DEFAULT_MODEL

    # Сохраняем сообщение для возможной суммаризации
    # Для групповых чатов используем chat_id, для личных - user_id
    message_key = chat_id if chat_id < 0 else user_id

    if message_key not in user_messages:
        user_messages[message_key] = []
    
    # Очищаем имя пользователя и текст сообщения перед сохранением
    clean_name = clean_input_text(update.effective_user.first_name)
    clean_text = clean_input_text(message_text)
    
    # Определяем, является ли сообщение ответом на другое сообщение
    reply_to_message_id = None
    reply_to_index = None
    
    if update.message.reply_to_message:
        reply_to_message_id = update.message.reply_to_message.message_id
        
        # Ищем индекс сообщения, на которое отвечают
        if message_key in user_messages:
            for i, msg in enumerate(user_messages[message_key]):
                if msg.get("message_id") == reply_to_message_id:
                    reply_to_index = i
                    break
    
    # Сохраняем сообщение в расширенном формате с дополнительными метаданными
    message_data = {
        "sender": clean_name,
        "text": clean_text,
        "reply_to_index": reply_to_index,  # Может быть None или целым числом
        "message_id": update.message.message_id,
        "date": update.message.date.isoformat(),
        "user_id": user_id,
        "chat_id": chat_id
    }
    
    # Проверяем, что reply_to_index не является NaN
    import math
    if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
        logger.warning(f"Обнаружено NaN значение в reply_to_index, устанавливаем None")
        message_data["reply_to_index"] = None
    
    # Добавляем информацию о пользователе, если доступна
    if update.effective_user.username:
        message_data["username"] = update.effective_user.username
    if update.effective_user.last_name:
        message_data["last_name"] = update.effective_user.last_name
        
    # Добавляем информацию о сообщении, на которое отвечают
    if reply_to_message_id:
        message_data["reply_to_message_id"] = reply_to_message_id
        
    user_messages[message_key].append(message_data)

    # История сообщений не ограничивается

    # Определяем, нужно ли отвечать на сообщение
    should_respond = False
    bot_username = context.bot.username
    
    # В личных чатах отвечаем на все сообщения
    if chat_id > 0:  # Личный чат
        should_respond = True
    else:  # Групповой чат
        # Проверяем, упоминается ли бот в сообщении
        if f"@{bot_username}" in message_text:
            should_respond = True
        # Проверяем, начинается ли сообщение с префикса "!"
        elif message_text.startswith("!"):
            should_respond = True
            # Удаляем префикс из сообщения
            message_text = message_text[1:].strip()
        # Проверяем, является ли сообщение ответом на сообщение бота
        elif update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
            should_respond = True
    
    # Если не нужно отвечать, просто выходим
    if not should_respond:
        return

    # Получаем информацию об отправителе
    sender_name = update.effective_user.first_name
    if update.effective_user.last_name:
        sender_name += f" {update.effective_user.last_name}"
    if update.effective_user.username:
        sender_name += f" (@{update.effective_user.username})"
    
    # Очищаем имя отправителя от специальных символов
    sender_name = clean_input_text(sender_name)

    # Получаем время сообщения
    message_time = update.message.date.strftime("%Y-%m-%d %H:%M:%S")

    # Отправляем начальное сообщение, которое будем обновлять
    initial_message = await update.message.reply_text("⏳ Генерирую ответ...")

    # Очищаем текст запроса от технических тегов
    cleaned_message = clean_input_text(message_text)
    
    # Запускаем обработку запроса в фоновом режиме
    asyncio.create_task(
        process_llm_request(
            cleaned_message,
            user_id,
            chat_id,
            initial_message,
            sender_name=sender_name,
            message_time=message_time
        )
    )


async def send_typing_action(chat):
    """Периодически отправляет статус 'typing' в чат."""
    try:
        while True:
            await chat.send_action(action="typing")
            await asyncio.sleep(4)  # Отправляем статус каждые 4 секунды
    except asyncio.CancelledError:
        # Задача была отменена, это нормально
        pass
    except Exception as e:
        logger.error(f"Ошибка при отправке статуса typing: {e}")
        # Не выбрасываем исключение дальше, чтобы не прерывать основную задачу


async def process_llm_request(message_text: str, user_id: int, chat_id: int, initial_message, sender_name: str = None, message_time: str = None, is_hidden: bool = False) -> None:
    """
    Обрабатывает запрос к LLM в фоновом режиме и обновляет сообщение с ответом.
    
    Args:
        message_text: Текст запроса (уже очищенный от технических тегов)
        user_id: ID пользователя
        chat_id: ID чата
        initial_message: Сообщение, которое будет обновляться с ответом
        sender_name: Имя отправителя сообщения (уже очищенное)
        message_time: Дата и время отправки сообщения
        is_hidden: Флаг, указывающий, что это скрытый запрос (не сохраняется в истории чата)
    """
    # Создаем задачу для периодической отправки статуса "typing"
    typing_task = asyncio.create_task(send_typing_action(initial_message.chat))
    
    try:
        # Получаем ответ от LLM с использованием выбранной модели и промпта
        response = await get_llm_response(
            message_text,
            user_id,
            chat_id,
            sender_name=sender_name,
            message_time=message_time,
            is_hidden=is_hidden
        )
        
        # Останавливаем отправку статуса "typing"
        typing_task.cancel()
        
        # Форматируем и отправляем ответ
        if is_hidden:
            # Для скрытых запросов добавляем информацию о чате
            formatted_response = f"🔒 Скрытый запрос из чата {chat_id}:\n\n{message_text}\n\n📝 Ответ:\n\n{response}"
            await initial_message.edit_text(formatted_response)
        else:
            # Обычный ответ
            await initial_message.edit_text(response)
        
        # Сохраняем ответ бота в истории сообщений только если это не скрытый запрос
        if not is_hidden:
            message_key = chat_id if chat_id < 0 else user_id
            if message_key in user_messages:
                # Очищаем ответ бота перед сохранением
                clean_response = clean_input_text(response)
                
                # Сохраняем ответ бота в расширенном формате с дополнительными метаданными
                # Ответ бота всегда является ответом на последнее сообщение пользователя
                reply_to_index = len(user_messages[message_key]) - 1 if user_messages[message_key] else None
                
                # Получаем ID сообщения, на которое отвечает бот
                reply_to_message_id = None
                if reply_to_index is not None:
                    # Проверяем, не является ли reply_to_index значением NaN
                    import math
                    if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
                        # Пропускаем NaN значения
                        logger.info("Пропускаем NaN значение в reply_to_index")
                    else:
                        try:
                            reply_to_index_int = int(reply_to_index)
                            if 0 <= reply_to_index_int < len(user_messages[message_key]):
                                reply_to_message_id = user_messages[message_key][reply_to_index_int].get("message_id")
                            else:
                                logger.warning(f"Индекс reply_to_index вне диапазона: {reply_to_index_int}, длина списка: {len(user_messages[message_key])}")
                        except (ValueError, TypeError):
                            logger.warning(f"Некорректный индекс reply_to_index: {reply_to_index}, тип: {type(reply_to_index)}")
                
                # Получаем ID бота из сообщения
                bot_id = initial_message.from_user.id
                
                # Проверяем, что reply_to_index не является NaN
                import math
                if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
                    logger.warning(f"Обнаружено NaN значение в reply_to_index при сохранении ответа бота, устанавливаем None")
                    reply_to_index = None
                
                bot_message_data = {
                    "sender": "Bot",
                    "text": clean_response,
                    "reply_to_index": reply_to_index,
                    "message_id": initial_message.message_id,
                    "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "user_id": bot_id,
                    "chat_id": chat_id,
                    "is_bot": True
                }
                
                # Добавляем информацию о сообщении, на которое отвечает бот
                if reply_to_message_id:
                    bot_message_data["reply_to_message_id"] = reply_to_message_id
                    
                user_messages[message_key].append(bot_message_data)
                
                # Сохраняем историю сообщений каждые N сообщений
                if len(user_messages[message_key]) % 10 == 0:
                    save_messages()
    except Exception as e:
        # Останавливаем отправку статуса "typing"
        typing_task.cancel()
        
        logger.error(f"Ошибка при получении ответа от LLM: {e}")
        await initial_message.edit_text(
            "Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        )


def clean_input_text(text: str) -> str:
    """
    Очищает входной текст от технических тегов и специальных символов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
        
    import re
    
    # Удаляем XML-подобные теги
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    
    # Удаляем конструкции вида @mail.sender="..."
    cleaned_text = re.sub(r'@\w+\.\w+="[^"]*"', '', cleaned_text)
    
    # Удаляем конструкции вида @username
    cleaned_text = re.sub(r'@\w+', '', cleaned_text)
    
    # Удаляем символы \, которые могут использоваться для экранирования
    cleaned_text = cleaned_text.replace('\\', '')
    
    # Удаляем лишние пробелы и переносы строк
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text


async def get_llm_response(query: str, user_id: int, chat_id: int, sender_name: str = None, message_time: str = None, is_hidden: bool = False) -> str:
    """
    Асинхронно получает ответ от LLM с использованием выбранной модели и промпта.

    Args:
        query: Текст запроса
        user_id: ID пользователя
        chat_id: ID чата
        sender_name: Имя отправителя сообщения
        message_time: Дата и время отправки сообщения
        is_hidden: Флаг, указывающий, что это скрытый запрос (не сохраняется в истории чата)

    Returns:
        Ответ от LLM
    """
    try:
        # Определяем модель для использования
        if chat_id < 0:  # Групповой чат
            model = chat_models.get(chat_id, DEFAULT_MODEL)
        else:  # Личный чат
            model = user_models.get(user_id, DEFAULT_MODEL)

        # Получаем промпт для ответа
        prompt_template = get_prompt("answer", chat_id)
        logger.info(f"Используется промпт для ответа, chat_id={chat_id}, is_hidden={is_hidden}, первые 100 символов: {prompt_template[:100]}...")

        # Получаем историю сообщений для этого чата/пользователя
        message_key = chat_id if chat_id < 0 else user_id
        history = user_messages.get(message_key, [])
        
        # Логируем информацию о запросе и истории
        logger.info(f"Запрос к LLM: user_id={user_id}, chat_id={chat_id}, is_hidden={is_hidden}")
        logger.info(f"История сообщений для ключа {message_key}: {len(history)} сообщений")
        
        # Если это скрытый запрос и нет истории, логируем предупреждение
        if is_hidden and not history:
            logger.warning(f"Скрытый запрос для chat_id={chat_id}, но история сообщений пуста")

        # Формируем контекст с историей сообщений
        conversation_history = ""
        if history and len(history) > 1:  # Если есть предыдущие сообщения
            # Берем последние N сообщений (не включая текущее), где N - настраиваемый параметр
            context_messages = bot_config["context_messages"]
            recent_messages = history[-(context_messages+1):-1] if len(history) > context_messages else history[:-1]

            # Форматируем историю сообщений в виде диалога с учетом реплаев
            conversation_history = "<conversation>\n"
            for i, msg in enumerate(recent_messages):
                sender = msg.get("sender", "Unknown")
                text = msg.get("text", "")
                message_id = msg.get("message_id", "")
                reply_to_index = msg.get("reply_to_index")
                
                # Базовые атрибуты для всех сообщений
                attrs = [
                    f'sender="{sender}"',
                    f'message_id="{message_id}"'
                ]
                
                # Добавляем дату сообщения, если она есть
                if "date" in msg:
                    attrs.append(f'date="{msg["date"]}"')
                
                # Если сообщение является ответом на другое сообщение
                if reply_to_index is not None:
                    # Проверяем, не является ли reply_to_index значением NaN
                    import math
                    if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
                        # Пропускаем сообщения с NaN в reply_to_index
                        continue
                        
                    # Преобразуем reply_to_index в целое число, если это число с плавающей точкой
                    try:
                        reply_to_index_int = int(reply_to_index)
                        if 0 <= reply_to_index_int < len(recent_messages):
                            # Получаем информацию о сообщении, на которое отвечают
                            replied_msg = recent_messages[reply_to_index_int]
                        else:
                            # Индекс вне диапазона, пропускаем
                            continue
                    except (ValueError, TypeError):
                        # Если не удалось преобразовать в целое число, пропускаем
                        logger.warning(f"Некорректный индекс reply_to_index: {reply_to_index}, тип: {type(reply_to_index)}")
                        continue
                    replied_sender = replied_msg.get("sender", "Unknown")
                    replied_text = replied_msg.get("text", "")
                    replied_id = replied_msg.get("message_id", "")
                    
                    # Добавляем информацию о сообщении, на которое отвечают
                    attrs.append(f'reply_to="{replied_sender}"')
                    attrs.append(f'reply_to_id="{replied_id}"')
                    attrs.append(f'reply_text="{replied_text}"')
                    
                # Форматируем сообщение со всеми атрибутами
                attrs_str = " ".join(attrs)
                conversation_history += f"<message {attrs_str}>{text}</message>\n"
                
            conversation_history += "</conversation>\n\n"

        # Добавляем информацию о текущем сообщении
        current_message = ""
        if sender_name:
            # Форматируем текущее сообщение с расширенными метаданными
            attrs = [
                f'sender="{sender_name}"',
                f'time="{message_time if message_time else ""}"',
                f'current="true"'
            ]
            
            # Если это ответ на сообщение бота, добавляем эту информацию
            if len(history) > 0 and history[-1].get("sender") == "Bot":
                bot_message = history[-1]
                attrs.append(f'reply_to="Bot"')
                attrs.append(f'reply_to_id="{bot_message.get("message_id", "")}"')
                
            attrs_str = " ".join(attrs)
            current_message = f"<message {attrs_str}>\n{query}\n</message>"
        else:
            current_message = f"<message current=\"true\">\n{query}\n</message>"

        # Объединяем историю и текущее сообщение
        full_context = f"{conversation_history}{current_message}"
        
        # Логируем информацию о контексте
        logger.info(f"История сообщений: {len(conversation_history)} символов")
        logger.info(f"Текущее сообщение: {len(current_message)} символов")
        logger.debug(f"Полный контекст (первые 200 символов): {full_context[:200]}...")

        # Форматируем промпт с историей сообщений и текущим вопросом
        formatted_prompt = prompt_template.format(question=full_context)
        
        # Логируем информацию о запросе для отладки
        logger.info(f"Запрос: user_id={user_id}, chat_id={chat_id}, is_hidden={is_hidden}")
        logger.info(f"История сообщений: {len(history)} сообщений")
        logger.info(f"Контекст: {len(conversation_history)} символов")
        logger.info(f"Текущее сообщение: {query[:50]}..." if len(query) > 50 else f"Текущее сообщение: {query}")

        # Проверяем, находится ли пользователь в фактическом режиме
        if user_id in factual_mode_users:
            # Если пользователь в фактическом режиме, всегда используем пониженную температуру
            temperature = bot_config["factual_temperature"]
            logger.info(f"Пользователь в фактическом режиме, используется глобальная температура {temperature}")
        else:
            # Определяем, является ли вопрос фактическим
            # Список ключевых слов, которые могут указывать на фактический вопрос
            factual_keywords = [
                "что такое", "кто такой", "когда", "где", "почему", "как", "сколько", 
                "объясни", "расскажи", "опиши", "в чем разница", "какой", "какая", "какие",
                "правда ли", "действительно ли", "факт", "история", "определение"
            ]
            
            # Проверяем, содержит ли запрос ключевые слова для фактических вопросов
            is_factual_query = any(keyword in query.lower() for keyword in factual_keywords)
            
            # Выбираем температуру в зависимости от типа вопроса и наличия индивидуальных настроек
            if is_factual_query:
                if chat_id < 0 and chat_id in chat_factual_temperatures:
                    temperature = chat_factual_temperatures[chat_id]
                    has_custom_temp = True
                    logger.info(f"Запрос определен как фактический, используется индивидуальная температура {temperature} для чата {chat_id}")
                else:
                    temperature = bot_config["factual_temperature"]
                    logger.info(f"Запрос определен как фактический, используется глобальная температура {temperature}")
            else:
                if chat_id < 0 and chat_id in chat_temperatures:
                    temperature = chat_temperatures[chat_id]
                    has_custom_temp = True
                    logger.info(f"Запрос определен как обычный, используется индивидуальная температура {temperature} для чата {chat_id}")
                else:
                    temperature = bot_config["temperature"]
                    logger.info(f"Запрос определен как обычный, используется глобальная температура {temperature}")
            
            logger.info(f"Запрос определен как {'фактический' if is_factual_query else 'обычный'}, используется температура {temperature}")

        # Получаем ответ от LLM с соответствующей температурой
        return await llm_client.answer_question(formatted_prompt, model=model, temperature=temperature)
    except Exception as e:
        logger.error(f"Ошибка при получении ответа от LLM: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."


async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /summarize."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Устанавливаем модель по умолчанию, если пользователь/чат еще не выбрал
    if user_id not in user_models:
        user_models[user_id] = DEFAULT_MODEL

    if chat_id < 0 and chat_id not in chat_models:  # Групповой чат
        chat_models[chat_id] = DEFAULT_MODEL

    # Для групповых чатов используем chat_id, для личных - user_id
    message_key = chat_id if chat_id < 0 else user_id

    if message_key not in user_messages or len(user_messages[message_key]) == 0:
        await update.message.reply_text(
            "Нет сообщений для суммаризации. Отправьте несколько сообщений сначала."
        )
        return

    # Отправляем начальное сообщение, которое будем обновлять
    initial_message = await update.message.reply_text("⏳ Генерирую саммари...")

    # Запускаем обработку запроса в фоновом режиме
    asyncio.create_task(
        process_summary_request(
            user_id,
            chat_id,
            initial_message,
            update.effective_chat.title,
            update.message.date
        )
    )


async def hidden_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /hidden_summary для получения саммари из группового чата в личке."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Эта команда доступна только в личных сообщениях с ботом."
        )
        return
    
    # Проверяем, есть ли аргументы команды (ID группового чата)
    if not context.args:
        await update.message.reply_text(
            "Пожалуйста, укажите ID группового чата после команды, например:\n"
            "/hidden_summary -1001234567890\n\n"
            "Чтобы узнать ID чата, отправьте команду /chatid в нужном групповом чате."
        )
        return
    
    try:
        # Пытаемся получить ID группового чата из аргументов
        group_chat_id = int(context.args[0])
        
        # Проверяем, что это действительно групповой чат (ID < 0)
        if group_chat_id >= 0:
            await update.message.reply_text(
                "Указанный ID не является ID группового чата. ID группового чата должен быть отрицательным числом."
            )
            return
        
        # Проверяем, есть ли сообщения для этого чата
        if group_chat_id not in user_messages or len(user_messages[group_chat_id]) == 0:
            await update.message.reply_text(
                f"Нет сообщений для суммаризации в чате с ID {group_chat_id}. "
                "Возможно, бот не добавлен в этот чат или в чате еще не было сообщений."
            )
            return
        
        # Получаем название чата, если оно сохранено в контексте
        chat_title = context.bot_data.get(f"chat_name_{group_chat_id}", f"Чат {abs(group_chat_id)}")
        
        # Отправляем начальное сообщение, которое будем обновлять
        initial_message = await update.message.reply_text(f"⏳ Генерирую скрытое саммари для чата {chat_title}...")
        
        # Запускаем обработку запроса в фоновом режиме
        asyncio.create_task(
            process_summary_request(
                user_id,
                group_chat_id,  # Используем ID группового чата вместо личного
                initial_message,
                chat_title,
                update.message.date,
                is_hidden=True  # Флаг, что это скрытое саммари
            )
        )
    except ValueError:
        await update.message.reply_text(
            "Неверный формат ID чата. Пожалуйста, укажите числовой ID, например:\n"
            "/hidden_summary -1001234567890"
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке команды hidden_summary: {e}")
        await update.message.reply_text(
            "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."
        )


async def process_summary_request(user_id: int, chat_id: int, initial_message, chat_title=None, message_date=None, is_hidden=False) -> None:
    """
    Обрабатывает запрос на суммаризацию в фоновом режиме и обновляет сообщение с результатом.
    
    Args:
        user_id: ID пользователя
        chat_id: ID чата
        initial_message: Сообщение, которое будет обновляться с результатом
        chat_title: Название чата
        message_date: Дата и время сообщения
        is_hidden: Флаг, указывающий, что это скрытое саммари из другого чата
    """
    # Создаем задачу для периодической отправки статуса "typing"
    typing_task = asyncio.create_task(send_typing_action(initial_message.chat))
    
    try:
        # Для групповых чатов используем chat_id, для личных - user_id
        message_key = chat_id if chat_id < 0 else user_id
        
        # Получаем все сообщения
        all_messages = user_messages.get(message_key, [])
        
        # Используем все сообщения для суммаризации без ограничений
        messages_to_summarize = all_messages
        
        # Форматируем сообщения для суммаризации с учетом реплаев
        formatted_messages = []
        for msg in messages_to_summarize:
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            reply_to_index = msg.get("reply_to_index")
            
            if reply_to_index is not None:
                # Проверяем, не является ли reply_to_index значением NaN
                import math
                if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
                    # Обрабатываем как обычное сообщение
                    formatted_messages.append(f"{sender}: {text}")
                    continue
                    
                # Преобразуем reply_to_index в целое число, если это число с плавающей точкой
                try:
                    reply_to_index_int = int(reply_to_index)
                    if 0 <= reply_to_index_int < len(all_messages):
                        # Получаем информацию о сообщении, на которое отвечают
                        replied_msg = all_messages[reply_to_index_int]
                    else:
                        # Индекс вне диапазона, обрабатываем как обычное сообщение
                        formatted_messages.append(f"{sender}: {text}")
                        continue
                except (ValueError, TypeError):
                    # Если не удалось преобразовать в целое число, обрабатываем как обычное сообщение
                    logger.warning(f"Некорректный индекс reply_to_index: {reply_to_index}, тип: {type(reply_to_index)}")
                    formatted_messages.append(f"{sender}: {text}")
                    continue
                replied_sender = replied_msg.get("sender", "Unknown")
                
                # Форматируем сообщение как ответ
                formatted_messages.append(f"{sender} (в ответ {replied_sender}): {text}")
            else:
                # Обычное сообщение
                formatted_messages.append(f"{sender}: {text}")
        
        # Объединяем сообщения
        text_to_summarize = "\n".join(formatted_messages)
        
        # Логируем информацию о количестве сообщений
        logger.info(f"Суммаризация: всего сообщений - {len(all_messages)}, используется - {len(messages_to_summarize)}")

        # Получаем информацию о чате
        chat_name = chat_title if chat_title else "Личный чат"
        current_time = message_date.strftime("%Y-%m-%d %H:%M:%S") if message_date else ""

        # Добавляем информацию о чате и времени в тегах, понятных для DeepSeek
        context_attrs = f"chat=\"{chat_name}\" time=\"{current_time}\" message_count=\"{len(messages_to_summarize)}\""
        text_with_context = f"<context {context_attrs}>\n{text_to_summarize}\n</context>"

        # Определяем модель для использования
        if chat_id < 0:  # Групповой чат
            model = chat_models.get(chat_id, DEFAULT_MODEL)
        else:  # Личный чат
            model = user_models.get(user_id, DEFAULT_MODEL)

        # Получаем промпт для саммари
        prompt_template = get_prompt("summary", chat_id)

        # Форматируем промпт с текстом для суммаризации
        formatted_prompt = prompt_template.format(text=text_with_context)

        # Используем настраиваемую температуру для саммаризации
        temperature = bot_config["summary_temperature"]
        logger.info(f"Саммаризация с температурой {temperature}")
        
        # Получаем саммари
        summary = await llm_client.get_summary(formatted_prompt, model=model, temperature=temperature)

        # Останавливаем отправку статуса "typing"
        typing_task.cancel()

        # Обновляем сообщение с результатом
        message_info = f"(на основе последних {len(messages_to_summarize)} сообщений из {len(all_messages)})" if len(all_messages) > len(messages_to_summarize) else ""
        
        if is_hidden:
            # Для скрытого саммари добавляем информацию о чате
            chat_info = f"Чат: {chat_title} (ID: {chat_id})"
            await initial_message.edit_text(f"Скрытое саммари {message_info}\n{chat_info}:\n\n{summary}")
        else:
            # Обычное саммари
            await initial_message.edit_text(f"Саммари {message_info}:\n\n{summary}")
    except Exception as e:
        # Останавливаем отправку статуса "typing"
        typing_task.cancel()

        logger.error(f"Ошибка при создании саммари: {e}")
        await initial_message.edit_text(
            "Извините, произошла ошибка при создании саммари. Пожалуйста, попробуйте позже."
        )


async def get_available_models() -> List[str]:
    """Получает список доступных моделей Ollama."""
    try:
        # Для Ollama 0.6.x используется endpoint /api/tags
        check_url = f"{LLM_API_URL}/tags" if LLM_API_URL.endswith('/api') else f"{LLM_API_URL}/api/tags"
        logger.info(f"Получение списка моделей по URL: {check_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(check_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Получен ответ: {json.dumps(data, ensure_ascii=False)}")

                    # Формат ответа от /api/tags в Ollama 0.6.x
                    models = [model.get("name", "") for model in data.get("models", [])]

                    # Фильтруем пустые значения
                    models = [model for model in models if model]

                    if not models:
                        logger.warning("Не найдено доступных моделей. Возвращаем модель по умолчанию.")
                        return [DEFAULT_MODEL]

                    logger.info(f"Найдены модели: {', '.join(models)}")
                    return models
                else:
                    logger.error(f"Ошибка при получении списка моделей: {response.status}")
                    return [DEFAULT_MODEL]
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")
        return [DEFAULT_MODEL]


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик команды /model."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Настройка модели доступна только в личных сообщениях с ботом. "
            "Пожалуйста, напишите боту в личку для настройки модели."
        )
        return ConversationHandler.END

    # Создаем клавиатуру для выбора: настроить для себя или для группового чата
    keyboard = [
        [InlineKeyboardButton("Для меня лично", callback_data="model_for_me")],
        [InlineKeyboardButton("Для группового чата", callback_data="model_for_chat")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Для кого вы хотите настроить модель?",
        reply_markup=reply_markup
    )

    return CHOOSING_MODEL_TARGET


async def model_target_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора цели настройки модели."""
    query = update.callback_query
    await query.answer()

    target = query.data.replace("model_for_", "")

    if target == "me":
        # Настройка модели для пользователя
        # Получаем список доступных моделей
        models = await get_available_models()

        if not models:
            await query.edit_message_text(
                "Не удалось получить список моделей. Пожалуйста, попробуйте позже."
            )
            return ConversationHandler.END

        # Создаем клавиатуру с кнопками для выбора модели
        keyboard = []
        for model in models:
            keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "Выберите модель для использования:",
            reply_markup=reply_markup
        )

        return CHOOSING_MODEL

    elif target == "chat":
        # Настройка модели для группового чата
        # Создаем клавиатуру для выбора группового чата
        # Сначала получаем список групповых чатов, для которых есть настройки
        group_chats = set()
        for chat_id in chat_models:
            if chat_id < 0:
                group_chats.add(chat_id)

        # Добавляем опцию "Новый чат"
        keyboard = [[InlineKeyboardButton("➕ Добавить новый чат", callback_data="chat_new")]]

        # Добавляем существующие чаты
        for chat_id in group_chats:
            # Получаем название чата из контекста, если возможно
            chat_name = context.bot_data.get(f"chat_name_{chat_id}", f"Чат {abs(chat_id)}")
            keyboard.append([InlineKeyboardButton(chat_name, callback_data=f"chat_{chat_id}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "Выберите групповой чат для настройки модели:",
            reply_markup=reply_markup
        )

        return CHOOSING_CHAT

    else:
        await query.edit_message_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        return ConversationHandler.END


async def chat_selection_for_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора чата для настройки модели."""
    query = update.callback_query
    await query.answer()

    callback_data = query.data.replace("chat_", "")

    if callback_data == "new":
        # Пользователь выбрал добавить новый чат
        await query.edit_message_text(
            "Пожалуйста, введите ID группового чата для настройки модели.\n\n"
            "Чтобы узнать ID чата, добавьте бота в группу и отправьте команду /chatid"
        )
        return ENTERING_CHAT_ID
    else:
        # Пользователь выбрал существующий чат
        try:
            selected_chat_id = int(callback_data)
            context.user_data["selected_chat_id"] = selected_chat_id

            # Получаем название чата
            chat_name = context.bot_data.get(f"chat_name_{selected_chat_id}", f"Чат {abs(selected_chat_id)}")

            # Получаем список доступных моделей
            models = await get_available_models()

            if not models:
                await query.edit_message_text(
                    "Не удалось получить список моделей. Пожалуйста, попробуйте позже."
                )
                return ConversationHandler.END

            # Создаем клавиатуру с кнопками для выбора модели
            keyboard = []
            for model in models:
                keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"Выберите модель для чата {chat_name}:",
                reply_markup=reply_markup
            )

            return CHOOSING_MODEL
        except ValueError:
            await query.edit_message_text(
                "Произошла ошибка при выборе чата. Пожалуйста, попробуйте снова."
            )
            return ConversationHandler.END


async def enter_chat_id_for_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик ввода ID чата для настройки модели."""
    try:
        chat_id = int(update.message.text)

        # Проверяем, что это групповой чат (ID < 0)
        if chat_id >= 0:
            await update.message.reply_text(
                "Введенный ID не является ID группового чата. ID группового чата должен быть отрицательным числом.\n"
                "Пожалуйста, введите корректный ID или отправьте /cancel для отмены."
            )
            return ENTERING_CHAT_ID

        # Сохраняем выбранный чат в контексте
        context.user_data["selected_chat_id"] = chat_id

        # Получаем список доступных моделей
        models = await get_available_models()

        if not models:
            await update.message.reply_text(
                "Не удалось получить список моделей. Пожалуйста, попробуйте позже."
            )
            return ConversationHandler.END

        # Создаем клавиатуру с кнопками для выбора модели
        keyboard = []
        for model in models:
            keyboard.append([InlineKeyboardButton(model, callback_data=f"model_{model}")])

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"Выберите модель для чата с ID {chat_id}:",
            reply_markup=reply_markup
        )

        return CHOOSING_MODEL
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите числовой ID чата или отправьте /cancel для отмены."
        )
        return ENTERING_CHAT_ID


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /chatid."""
    chat = update.effective_chat
    chat_id = chat.id

    # Базовая информация о чате
    chat_info = f"📊 *Информация о чате*\n\n"
    chat_info += f"🆔 *ID чата:* `{chat_id}`\n"

    # Тип чата
    if chat.type == "private":
        chat_type = "Личный чат"
    elif chat.type == "group":
        chat_type = "Группа"
    elif chat.type == "supergroup":
        chat_type = "Супергруппа"
    elif chat.type == "channel":
        chat_type = "Канал"
    else:
        chat_type = chat.type

    chat_info += f"📝 *Тип чата:* {chat_type}\n"

    # Название чата для групп и каналов
    if chat.title:
        chat_info += f"📋 *Название:* {chat.title}\n"

    # Информация о пользователе для личных чатов
    if chat.type == "private":
        user = update.effective_user
        chat_info += f"👤 *Пользователь:* {user.full_name}\n"
        if user.username:
            chat_info += f"🔖 *Username:* @{user.username}\n"
        chat_info += f"🆔 *User ID:* `{user.id}`\n"

    # Дополнительная информация для групп
    if chat.type in ["group", "supergroup"]:
        # Получаем информацию о количестве участников, если возможно
        try:
            chat_member_count = await context.bot.get_chat_member_count(chat_id)
            chat_info += f"👥 *Участников:* {chat_member_count}\n"
        except Exception as e:
            logger.error(f"Ошибка при получении количества участников: {e}")

        # Получаем информацию о создателе группы, если возможно
        try:
            administrators = await context.bot.get_chat_administrators(chat_id)
            creator = next((admin for admin in administrators if admin.status == "creator"), None)
            if creator:
                creator_user = creator.user
                chat_info += f"👑 *Создатель:* {creator_user.full_name}"
                if creator_user.username:
                    chat_info += f" (@{creator_user.username})"
                chat_info += "\n"

            # Добавляем информацию о количестве администраторов
            chat_info += f"🛡️ *Администраторов:* {len(administrators)}\n"
        except Exception as e:
            logger.error(f"Ошибка при получении информации об администраторах: {e}")

    # Информация о настройках бота для этого чата
    chat_info += f"\n🤖 *Настройки бота*\n"

    # Модель
    if chat_id < 0:  # Групповой чат
        model = chat_models.get(chat_id, DEFAULT_MODEL)
    else:  # Личный чат
        model = user_models.get(update.effective_user.id, DEFAULT_MODEL)
    chat_info += f"🧠 *Модель:* {model}\n"

    # Промпты
    if chat_id < 0:  # Групповой чат
        has_custom_answer_prompt = chat_id in group_prompts["answer"]
        has_custom_summary_prompt = chat_id in group_prompts["summary"]
    else:  # Личный чат
        has_custom_answer_prompt = chat_id in chat_prompts["answer"]
        has_custom_summary_prompt = chat_id in chat_prompts["summary"]

    chat_info += f"💬 *Пользовательский промпт для ответов:* {'✅' if has_custom_answer_prompt else '❌'}\n"
    chat_info += f"📝 *Пользовательский промпт для саммари:* {'✅' if has_custom_summary_prompt else '❌'}\n"

    # Количество сообщений в истории
    message_key = chat_id if chat_id < 0 else update.effective_user.id
    message_count = len(user_messages.get(message_key, []))
    chat_info += f"📚 *Сообщений в истории:* {message_count}\n"
    chat_info += f"🔄 *Сообщений в контексте:* {bot_config['context_messages']}\n"

    # Отправляем информацию
    await update.message.reply_text(chat_info, parse_mode="Markdown")


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора модели."""
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    chat_id = query.message.chat.id
    model_name = query.data.replace("model_", "")

    # Проверяем, есть ли выбранный чат в контексте
    selected_chat_id = context.user_data.get("selected_chat_id")

    if selected_chat_id:
        # Устанавливаем модель для выбранного группового чата
        chat_models[selected_chat_id] = model_name

        # Получаем название чата
        chat_name = context.bot_data.get(f"chat_name_{selected_chat_id}", f"Чат {abs(selected_chat_id)}")

        await query.edit_message_text(f"Выбрана модель {model_name} для чата: {chat_name}")

        # Очищаем выбранный чат из контекста
        del context.user_data["selected_chat_id"]
    else:
        # Устанавливаем модель для пользователя
        user_models[user_id] = model_name
        await query.edit_message_text(f"Выбрана модель: {model_name}")

    # Сохраняем изменения
    save_models()

    return ConversationHandler.END


async def prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик команды /prompt."""
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id

    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Настройка промптов доступна только в личных сообщениях с ботом. "
            "Пожалуйста, напишите боту в личку для настройки промптов."
        )
        return ConversationHandler.END

    # Создаем клавиатуру для выбора группового чата для настройки
    # Сначала получаем список групповых чатов, для которых есть настройки
    group_chats = set()
    for prompt_type in chat_prompts:
        group_chats.update(chat_id for chat_id in chat_prompts[prompt_type] if chat_id < 0)

    # Добавляем опцию "Новый чат"
    keyboard = [[InlineKeyboardButton("➕ Добавить новый чат", callback_data="prompt_chat_new")]]

    # Добавляем существующие чаты
    for chat_id in group_chats:
        # Получаем название чата из контекста, если возможно
        chat_name = context.bot_data.get(f"chat_name_{chat_id}", f"Чат {abs(chat_id)}")
        keyboard.append([InlineKeyboardButton(chat_name, callback_data=f"prompt_chat_{chat_id}")])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Выберите групповой чат для настройки промптов:",
        reply_markup=reply_markup
    )

    # Сохраняем состояние - выбор чата
    return CHOOSING_CHAT


async def chat_selection_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора чата для настройки промптов."""
    query = update.callback_query
    await query.answer()

    callback_data = query.data.replace("prompt_chat_", "")

    if callback_data == "new":
        # Пользователь выбрал добавить новый чат
        await query.edit_message_text(
            "Пожалуйста, введите ID группового чата для настройки.\n\n"
            "Чтобы узнать ID чата, добавьте бота в группу и отправьте команду /chatid"
        )
        return ENTERING_CHAT_ID
    else:
        # Пользователь выбрал существующий чат
        try:
            selected_chat_id = int(callback_data)
            context.user_data["selected_chat_id"] = selected_chat_id

            # Получаем название чата
            chat_name = context.bot_data.get(f"chat_name_{selected_chat_id}", f"Чат {abs(selected_chat_id)}")

            # Создаем клавиатуру с кнопками для выбора типа промпта
            keyboard = [
                [InlineKeyboardButton("Промпт для ответов", callback_data="prompt_type_answer")],
                [InlineKeyboardButton("Промпт для саммари", callback_data="prompt_type_summary")],
                [InlineKeyboardButton("Сбросить промпты", callback_data="prompt_type_reset")]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"Настройка промптов для чата: {chat_name}\n\n"
                "Выберите тип промпта для настройки:",
                reply_markup=reply_markup
            )

            return CHOOSING_PROMPT_TYPE
        except ValueError:
            await query.edit_message_text(
                "Произошла ошибка при выборе чата. Пожалуйста, попробуйте снова."
            )
            return ConversationHandler.END


async def enter_chat_id_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик ввода ID чата для настройки промптов."""
    try:
        chat_id = int(update.message.text)

        # Проверяем, что это групповой чат (ID < 0)
        if chat_id >= 0:
            await update.message.reply_text(
                "Введенный ID не является ID группового чата. ID группового чата должен быть отрицательным числом.\n"
                "Пожалуйста, введите корректный ID или отправьте /cancel для отмены."
            )
            return ENTERING_CHAT_ID

        # Сохраняем выбранный чат в контексте
        context.user_data["selected_chat_id"] = chat_id

        # Создаем клавиатуру с кнопками для выбора типа промпта
        keyboard = [
            [InlineKeyboardButton("Промпт для ответов", callback_data="prompt_type_answer")],
            [InlineKeyboardButton("Промпт для саммари", callback_data="prompt_type_summary")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"Настройка промптов для чата с ID: {chat_id}\n\n"
            "Выберите тип промпта для настройки:",
            reply_markup=reply_markup
        )

        return CHOOSING_PROMPT_TYPE
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите числовой ID чата или отправьте /cancel для отмены."
        )
        return ENTERING_CHAT_ID
    else:
        # Пользователь выбрал существующий чат
        try:
            selected_chat_id = int(callback_data)
            context.user_data["selected_chat_id"] = selected_chat_id

            # Получаем название чата
            chat_name = context.bot_data.get(f"chat_name_{selected_chat_id}", f"Чат {abs(selected_chat_id)}")

            # Создаем клавиатуру с кнопками для выбора типа промпта
            keyboard = [
                [InlineKeyboardButton("Промпт для ответов", callback_data="prompt_type_answer")],
                [InlineKeyboardButton("Промпт для саммари", callback_data="prompt_type_summary")],
                [InlineKeyboardButton("Сбросить промпты", callback_data="prompt_type_reset")]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await query.edit_message_text(
                f"Настройка промптов для чата: {chat_name}\n\n"
                "Выберите тип промпта для настройки:",
                reply_markup=reply_markup
            )

            return CHOOSING_PROMPT_TYPE
        except ValueError:
            await query.edit_message_text(
                "Произошла ошибка при выборе чата. Пожалуйста, попробуйте снова."
            )
            return ConversationHandler.END


async def enter_chat_id_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик ввода ID чата."""
    try:
        chat_id = int(update.message.text)

        # Проверяем, что это групповой чат (ID < 0)
        if chat_id >= 0:
            await update.message.reply_text(
                "Введенный ID не является ID группового чата. ID группового чата должен быть отрицательным числом.\n"
                "Пожалуйста, введите корректный ID или отправьте /cancel для отмены."
            )
            return ENTERING_CHAT_ID

        # Сохраняем выбранный чат в контексте
        context.user_data["selected_chat_id"] = chat_id

        # Создаем клавиатуру с кнопками для выбора типа промпта
        keyboard = [
            [InlineKeyboardButton("Промпт для ответов", callback_data="prompt_type_answer")],
            [InlineKeyboardButton("Промпт для саммари", callback_data="prompt_type_summary")]
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            f"Настройка промптов для чата с ID: {chat_id}\n\n"
            "Выберите тип промпта для настройки:",
            reply_markup=reply_markup
        )

        return CHOOSING_PROMPT_TYPE
    except ValueError:
        await update.message.reply_text(
            "Пожалуйста, введите числовой ID чата или отправьте /cancel для отмены."
        )
        return ENTERING_CHAT_ID


async def prompt_type_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора типа промпта."""
    query = update.callback_query
    await query.answer()

    # Получаем ID выбранного чата из контекста
    selected_chat_id = context.user_data.get("selected_chat_id")
    if not selected_chat_id:
        await query.edit_message_text("Произошла ошибка: не выбран чат. Пожалуйста, начните заново.")
        return ConversationHandler.END

    prompt_type = query.data.replace("prompt_type_", "")

    # Сохраняем выбранный тип промпта в контексте
    context.user_data["prompt_type"] = prompt_type

    if prompt_type == "reset":
        # Сбрасываем промпты для выбранного чата
        # Для групповых чатов (ID < 0) используем group_prompts
        if selected_chat_id < 0:
            if selected_chat_id in group_prompts["answer"]:
                del group_prompts["answer"][selected_chat_id]
            if selected_chat_id in group_prompts["summary"]:
                del group_prompts["summary"][selected_chat_id]
        # Для личных чатов используем chat_prompts
        else:
            if selected_chat_id in chat_prompts["answer"]:
                del chat_prompts["answer"][selected_chat_id]
            if selected_chat_id in chat_prompts["summary"]:
                del chat_prompts["summary"][selected_chat_id]

        # Сохраняем изменения
        save_prompts()

        await query.edit_message_text(f"Промпты для чата с ID {selected_chat_id} сброшены до значений по умолчанию.")
        return ConversationHandler.END

    # Показываем текущий промпт
    # Для групповых чатов (ID < 0) используем промпты из group_prompts
    if selected_chat_id < 0:
        if prompt_type == "answer":
            current_prompt = group_prompts["answer"].get(selected_chat_id, DEFAULT_ANSWER_PROMPT)
        elif prompt_type == "summary":
            current_prompt = group_prompts["summary"].get(selected_chat_id, DEFAULT_SUMMARY_PROMPT)
        else:
            current_prompt = ""
    # Для личных чатов используем промпты из chat_prompts
    else:
        if prompt_type == "answer":
            current_prompt = chat_prompts["answer"].get(selected_chat_id, DEFAULT_ANSWER_PROMPT)
        elif prompt_type == "summary":
            current_prompt = chat_prompts["summary"].get(selected_chat_id, DEFAULT_SUMMARY_PROMPT)
        else:
            current_prompt = ""

    await query.edit_message_text(
        f"Текущий промпт для {prompt_type} (чат {selected_chat_id}):\n\n"
        f"```\n{current_prompt}\n```\n\n"
        f"Отправьте новый промпт или /cancel для отмены.\n\n"
        f"Используйте {{question}} для вставки вопроса пользователя (для промпта ответов)\n"
        f"Используйте {{text}} для вставки текста для суммаризации (для промпта саммари)"
    )

    return SETTING_PROMPT


async def set_prompt_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик установки нового промпта."""
    new_prompt = update.message.text
    prompt_type = context.user_data.get("prompt_type")
    selected_chat_id = context.user_data.get("selected_chat_id")

    if not prompt_type or not selected_chat_id:
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        return ConversationHandler.END

    # Проверяем, что промпт содержит необходимые плейсхолдеры
    if prompt_type == "answer" and "{question}" not in new_prompt:
        await update.message.reply_text(
            "Промпт должен содержать {question} для вставки вопроса пользователя. Попробуйте снова."
        )
        return SETTING_PROMPT

    if prompt_type == "summary" and "{text}" not in new_prompt:
        await update.message.reply_text(
            "Промпт должен содержать {text} для вставки текста для суммаризации. Попробуйте снова."
        )
        return SETTING_PROMPT

    # Сохраняем новый промпт для выбранного чата
    # Для групповых чатов (ID < 0) используем group_prompts
    if selected_chat_id < 0:
        group_prompts[prompt_type][selected_chat_id] = new_prompt
    # Для личных чатов используем chat_prompts
    else:
        chat_prompts[prompt_type][selected_chat_id] = new_prompt

    # Сохраняем изменения в файл
    save_prompts()

    await update.message.reply_text(
        f"Промпт для {prompt_type} успешно обновлен для чата с ID {selected_chat_id}!"
    )

    return ConversationHandler.END


async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик команды /config."""
    chat_id = update.effective_chat.id

    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Настройка параметров доступна только в личных сообщениях с ботом. "
            "Пожалуйста, напишите боту в личку для настройки параметров."
        )
        return ConversationHandler.END
    # Создаем клавиатуру с кнопками для выбора параметра конфигурации
    keyboard = [
        [InlineKeyboardButton("Количество сообщений в контексте", callback_data="config_context_messages")],
        [InlineKeyboardButton("Количество сообщений для саммари", callback_data="config_summary_messages")],
        [InlineKeyboardButton("Интервал автосохранения (мин)", callback_data="config_auto_save_interval")],
        [InlineKeyboardButton("Температура генерации", callback_data="config_temperature")],
        [InlineKeyboardButton("Температура для фактов", callback_data="config_factual_temperature")],
        [InlineKeyboardButton("Температура для саммари", callback_data="config_summary_temperature")]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Выберите параметр для настройки:\n\n"
        f"Текущие настройки:\n"
        f"- Сообщений в контексте: {bot_config['context_messages']}\n"
        f"- Сообщений для саммари: {bot_config['summary_messages']}\n"
        f"- Интервал автосохранения: {bot_config['auto_save_interval']} мин\n"
        f"- Температура генерации: {bot_config['temperature']}\n"
        f"- Температура для фактов: {bot_config['factual_temperature']}\n"
        f"- Температура для саммари: {bot_config['summary_temperature']}",
        reply_markup=reply_markup
    )

    return CHOOSING_CONFIG


async def config_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик выбора параметра конфигурации."""
    query = update.callback_query
    await query.answer()

    config_param = query.data.replace("config_", "")
    context.user_data["config_param"] = config_param

    # Получаем текущее значение параметра
    current_value = bot_config[config_param]

    # Формируем описание параметра
    if config_param == "context_messages":
        description = "количество предыдущих сообщений, которые будут отправлены модели вместе с текущим запросом"
    elif config_param == "summary_messages":
        description = "максимальное количество последних сообщений, которые будут использоваться для создания саммари"
    elif config_param == "max_history":
        description = "максимальное количество сообщений, которые будут храниться в истории для каждого чата"
    elif config_param == "auto_save_interval":
        description = "интервал автоматического сохранения данных в минутах"
    elif config_param == "temperature":
        description = "температура генерации (от 0.1 до 1.0). Более высокие значения делают ответы более креативными, но менее точными"
    elif config_param == "factual_temperature":
        description = "температура для фактических вопросов (от 0.1 до 1.0). Более низкие значения уменьшают вероятность галлюцинаций"
    elif config_param == "summary_temperature":
        description = "температура для саммаризации (от 0.1 до 1.0). Более низкие значения делают саммари более точным и последовательным"
    else:
        description = "параметр конфигурации"

    await query.edit_message_text(
        f"Настройка параметра: {config_param}\n\n"
        f"Описание: {description}\n\n"
        f"Текущее значение: {current_value}\n\n"
        f"Введите новое значение (целое число) или /cancel для отмены:"
    )

    return SETTING_CONFIG


async def set_config_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обработчик установки нового значения параметра конфигурации."""
    config_param = context.user_data.get("config_param")
    if not config_param:
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте снова.")
        return ConversationHandler.END

    # Получаем новое значение
    try:
        # Для температуры используем float, для остальных параметров - int
        if config_param in ["temperature", "factual_temperature", "summary_temperature"]:
            new_value = float(update.message.text)
            if new_value <= 0 or new_value > 1.0:
                await update.message.reply_text("Значение температуры должно быть в диапазоне от 0.1 до 1.0. Попробуйте снова.")
                return SETTING_CONFIG
        else:
            new_value = int(update.message.text)
            if new_value <= 0:
                await update.message.reply_text("Значение должно быть положительным числом. Попробуйте снова.")
                return SETTING_CONFIG
    except ValueError:
        if config_param in ["temperature", "factual_temperature", "summary_temperature"]:
            await update.message.reply_text("Введите число с плавающей точкой (например, 0.7). Попробуйте снова.")
        else:
            await update.message.reply_text("Введите целое число. Попробуйте снова.")
        return SETTING_CONFIG

    # Проверяем ограничения для разных параметров
    # Ограничение на количество сообщений в контексте удалено
        
    # Ограничение на количество сообщений для саммари удалено

    if config_param == "auto_save_interval" and new_value < 1:
        await update.message.reply_text(
            "Интервал автосохранения должен быть не менее 1 минуты. Попробуйте снова."
        )
        return SETTING_CONFIG
        
    if config_param == "temperature" and new_value < 0.1:
        await update.message.reply_text(
            "Слишком низкая температура может привести к повторяющимся ответам. "
            "Рекомендуется значение не менее 0.1. Попробуйте снова."
        )
        return SETTING_CONFIG
        
    if config_param == "factual_temperature" and new_value < 0.1:
        await update.message.reply_text(
            "Слишком низкая температура может привести к повторяющимся ответам. "
            "Рекомендуется значение не менее 0.1. Попробуйте снова."
        )
        return SETTING_CONFIG
        
    if config_param == "summary_temperature" and new_value < 0.1:
        await update.message.reply_text(
            "Слишком низкая температура может привести к повторяющимся ответам. "
            "Рекомендуется значение не менее 0.1. Попробуйте снова."
        )
        return SETTING_CONFIG

    # Сохраняем новое значение
    old_value = bot_config[config_param]
    bot_config[config_param] = new_value

    # Сохраняем конфигурацию
    save_config()

    await update.message.reply_text(
        f"Параметр {config_param} успешно обновлен!\n\n"
        f"Старое значение: {old_value}\n"
        f"Новое значение: {new_value}"
    )

    return ConversationHandler.END


async def factual_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /factual - переключает режим точных ответов."""
    user_id = update.effective_user.id
    
    if user_id in factual_mode_users:
        # Выключаем фактический режим
        factual_mode_users.remove(user_id)
        await update.message.reply_text(
            "✅ Фактический режим выключен. Бот вернулся к обычному режиму с температурой "
            f"{bot_config['temperature']}."
        )
    else:
        # Включаем фактический режим
        factual_mode_users.add(user_id)
        await update.message.reply_text(
            "✅ Фактический режим включен. Бот будет использовать пониженную температуру "
            f"{bot_config['factual_temperature']} для всех ваших запросов, чтобы уменьшить вероятность галлюцинаций.\n\n"
            "Используйте команду /factual еще раз, чтобы вернуться к обычному режиму."
        )


async def hidden_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /hidden_query - отправляет запрос к боту от имени группового чата, но ответ виден только вам."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Эта команда доступна только в личных сообщениях с ботом."
        )
        return
    
    # Проверяем, есть ли аргументы команды
    if not context.args or len(context.args) < 2:
        await update.message.reply_text(
            "Использование: /hidden_query [ID чата] [запрос]\n\n"
            "Например: /hidden_query -1001234567890 Что такое Python?"
        )
        return
    
    # Получаем ID чата и запрос
    try:
        target_chat_id = int(context.args[0])
        query_text = " ".join(context.args[1:])
    except ValueError:
        await update.message.reply_text(
            "Неверный формат ID чата. Используйте числовой ID.\n"
            "Чтобы узнать ID чата, используйте команду /chatid в нужном чате."
        )
        return
    
    # Проверяем, есть ли история сообщений для указанного чата
    if target_chat_id not in user_messages:
        logger.warning(f"Скрытый запрос: история сообщений для чата с ID {target_chat_id} не найдена")
        await update.message.reply_text(
            f"История сообщений для чата с ID {target_chat_id} не найдена. "
            "Убедитесь, что бот добавлен в этот чат и там есть сообщения."
        )
        return
    
    # Логируем информацию о запросе
    logger.info(f"Скрытый запрос: user_id={user_id}, target_chat_id={target_chat_id}")
    logger.info(f"Скрытый запрос: текст запроса: {query_text}")
    logger.info(f"Скрытый запрос: количество сообщений в истории: {len(user_messages[target_chat_id])}")
    
    # Отправляем начальное сообщение, которое будем обновлять
    initial_message = await update.message.reply_text(
        f"⏳ Генерирую ответ на запрос из чата {target_chat_id}..."
    )
    
    # Запускаем обработку запроса в фоновом режиме
    asyncio.create_task(
        process_llm_request(
            query_text,
            user_id,
            target_chat_id,  # Используем ID целевого чата
            initial_message,
            sender_name=update.effective_user.first_name,
            message_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_hidden=True  # Указываем, что это скрытый запрос
        )
    )


async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /reload - перезагружает модули."""
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    
    # Команда доступна только в личных чатах
    if chat_id < 0:
        await update.message.reply_text(
            "Эта команда доступна только в личных сообщениях с ботом."
        )
        return
    
    # Проверяем, есть ли аргументы команды (имя модуля)
    if context.args:
        module_name = context.args[0]
        success = reload_module(module_name)
        
        if success:
            await update.message.reply_text(f"✅ Модуль {module_name} успешно перезагружен.")
        else:
            await update.message.reply_text(f"❌ Не удалось перезагрузить модуль {module_name}. Проверьте логи.")
    else:
        # Если модуль не указан, перезагружаем все отслеживаемые модули
        modules_to_check = ["llm_client"]
        reloaded = []
        failed = []
        
        for module_name in modules_to_check:
            if reload_module(module_name):
                reloaded.append(module_name)
            else:
                failed.append(module_name)
        
        if reloaded:
            reloaded_str = ", ".join(reloaded)
            await update.message.reply_text(f"✅ Успешно перезагружены модули: {reloaded_str}")
        
        if failed:
            failed_str = ", ".join(failed)
            await update.message.reply_text(f"❌ Не удалось перезагрузить модули: {failed_str}")
            
        if not reloaded and not failed:
            await update.message.reply_text("❓ Нет модулей для перезагрузки.")


async def handle_edited_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик отредактированных сообщений."""
    # Проверяем, что это действительно отредактированное сообщение
    if not update.edited_message:
        return
        
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    message_text = update.edited_message.text
    message_id = update.edited_message.message_id
    
    # Для групповых чатов используем chat_id, для личных - user_id
    message_key = chat_id if chat_id < 0 else user_id
    
    # Проверяем, есть ли история сообщений для этого чата/пользователя
    if message_key not in user_messages:
        return
    
    # Ищем сообщение с таким message_id в истории
    for i, msg in enumerate(user_messages[message_key]):
        if msg.get("message_id") == message_id:
            # Обновляем текст сообщения
            clean_text = clean_input_text(message_text)
            user_messages[message_key][i]["text"] = clean_text
            logger.debug(f"Обновлено отредактированное сообщение: {message_id}")
            
            # Сохраняем историю сообщений
            save_messages()
            break


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /chatid - показывает ID текущего чата."""
    chat_id = update.effective_chat.id
    chat_type = update.effective_chat.type
    chat_title = update.effective_chat.title if update.effective_chat.title else "Личный чат"

    # Сохраняем название чата в контексте бота
    if chat_id < 0:  # Групповой чат
        context.bot_data[f"chat_name_{chat_id}"] = chat_title

    await update.message.reply_text(
        f"Информация о чате:\n"
        f"ID: {chat_id}\n"
        f"Тип: {chat_type}\n"
        f"Название: {chat_title}"
    )


async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Отменяет текущий разговор."""
    await update.message.reply_text("Операция отменена.")
    return ConversationHandler.END


def load_prompts() -> None:
    """Загружает сохраненные промпты из файла.
    Loads saved prompts from files."""
    global chat_prompts, group_prompts
    try:
        # Загружаем промпты для личных чатов
        # Load prompts for private chats
        if os.path.exists(PROMPTS_FILE):
            with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
                loaded_prompts = json.load(f)

                # Преобразуем строковые ключи чатов обратно в целые числа
                # Convert string chat keys back to integers
                for prompt_type in loaded_prompts:
                    chat_prompts[prompt_type] = {int(chat_id): prompt for chat_id, prompt in loaded_prompts[prompt_type].items()}

            logger.info(f"Промпты для личных чатов загружены из {PROMPTS_FILE}")
            logger.info(f"Prompts for private chats loaded from {PROMPTS_FILE}")

        # Загружаем промпты для групповых чатов
        # Load prompts for group chats
        if os.path.exists(GROUP_PROMPTS_FILE):
            with open(GROUP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                loaded_prompts = json.load(f)

                # Преобразуем строковые ключи чатов обратно в целые числа
                # Convert string chat keys back to integers
                for prompt_type in loaded_prompts:
                    group_prompts[prompt_type] = {int(chat_id): prompt for chat_id, prompt in loaded_prompts[prompt_type].items()}

            logger.info(f"Промпты для групповых чатов загружены из {GROUP_PROMPTS_FILE}")
            logger.info(f"Prompts for group chats loaded from {GROUP_PROMPTS_FILE}")
            
        # Загружаем специальные промпты для конкретных групп (не включаются в Git)
        # Load custom prompts for specific groups (not included in Git)
        CUSTOM_PROMPTS_FILE = os.path.join(DATA_DIR, "custom_prompts.json")
        if os.path.exists(CUSTOM_PROMPTS_FILE):
            with open(CUSTOM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
                
                # Обрабатываем каждую группу из файла custom_prompts.json
                # Process each group from custom_prompts.json file
                for chat_id_str, prompts in custom_prompts.items():
                    chat_id = int(chat_id_str)
                    
                    # Добавляем промпты для ответов и саммари, если они есть
                    # Add prompts for answers and summaries if they exist
                    if "answer" in prompts:
                        group_prompts["answer"][chat_id] = prompts["answer"]
                    if "summary" in prompts:
                        group_prompts["summary"][chat_id] = prompts["summary"]
                        
            logger.info(f"Специальные промпты загружены из {CUSTOM_PROMPTS_FILE}")
            logger.info(f"Custom prompts loaded from {CUSTOM_PROMPTS_FILE}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке промптов: {e}")
        logger.error(f"Error loading prompts: {e}")


def save_prompts() -> None:
    """Сохраняет промпты в файл.
    Saves prompts to files."""
    try:
        # Сохраняем промпты для личных чатов
        # Save prompts for private chats
        serializable_prompts = {}
        for prompt_type, prompts in chat_prompts.items():
            serializable_prompts[prompt_type] = {str(chat_id): prompt for chat_id, prompt in prompts.items()}

        with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_prompts, f, ensure_ascii=False, indent=2)

        logger.info(f"Промпты для личных чатов сохранены в {PROMPTS_FILE}")
        logger.info(f"Prompts for private chats saved to {PROMPTS_FILE}")

        # Сохраняем промпты для групповых чатов
        # Save prompts for group chats
        serializable_group_prompts = {}
        for prompt_type, prompts in group_prompts.items():
            # Исключаем специальные чаты, которые хранятся в custom_prompts.json
            # Exclude special chats that are stored in custom_prompts.json
            CUSTOM_PROMPTS_FILE = os.path.join(DATA_DIR, "custom_prompts.json")
            custom_chat_ids = set()
            
            if os.path.exists(CUSTOM_PROMPTS_FILE):
                try:
                    with open(CUSTOM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                        custom_prompts = json.load(f)
                        custom_chat_ids = {int(chat_id) for chat_id in custom_prompts.keys()}
                except Exception as e:
                    logger.error(f"Ошибка при чтении custom_prompts.json: {e}")
                    logger.error(f"Error reading custom_prompts.json: {e}")
            
            # Сохраняем только промпты для обычных групп
            # Save only prompts for regular groups
            serializable_group_prompts[prompt_type] = {str(chat_id): prompt for chat_id, prompt in prompts.items() 
                                                   if chat_id not in custom_chat_ids}

        with open(GROUP_PROMPTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_group_prompts, f, ensure_ascii=False, indent=2)

        logger.info(f"Промпты для групповых чатов сохранены в {GROUP_PROMPTS_FILE}")
        logger.info(f"Prompts for group chats saved to {GROUP_PROMPTS_FILE}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении промптов: {e}")
        logger.error(f"Error saving prompts: {e}")


def get_prompt(prompt_type: str, chat_id: int) -> str:
    """
    Возвращает промпт для указанного типа и чата.
    Если промпт не найден, возвращает промпт по умолчанию.
    
    Returns a prompt for the specified type and chat.
    If the prompt is not found, returns the default prompt.

    Args:
        prompt_type: Тип промпта ("answer" или "summary") / Prompt type ("answer" or "summary")
        chat_id: ID чата / Chat ID

    Returns:
        Промпт для указанного типа и чата / Prompt for the specified type and chat
    """
    # Логируем информацию о запросе промпта
    # Log information about the prompt request
    logger.info(f"Запрос промпта: тип={prompt_type}, chat_id={chat_id}")
    logger.info(f"Prompt request: type={prompt_type}, chat_id={chat_id}")
    
    # Сначала проверяем, есть ли специальные промпты для этого чата
    # First check if there are custom prompts for this chat
    CUSTOM_PROMPTS_FILE = os.path.join(DATA_DIR, "custom_prompts.json")
    if os.path.exists(CUSTOM_PROMPTS_FILE) and chat_id < 0:  # Только для групповых чатов / Only for group chats
        try:
            with open(CUSTOM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
                chat_id_str = str(chat_id)
                
                # Если чат имеет специальные промпты
                # If the chat has custom prompts
                if chat_id_str in custom_prompts and prompt_type in custom_prompts[chat_id_str]:
                    prompt = custom_prompts[chat_id_str][prompt_type]
                    logger.info(f"Используется специальный промпт для {prompt_type}, chat_id={chat_id}")
                    logger.info(f"Using custom prompt for {prompt_type}, chat_id={chat_id}")
                    return prompt
        except Exception as e:
            logger.error(f"Ошибка при чтении custom_prompts.json: {e}")
            logger.error(f"Error reading custom_prompts.json: {e}")
    
    # Для групповых чатов (ID < 0) используем промпты из group_prompts
    # For group chats (ID < 0) use prompts from group_prompts
    if chat_id < 0:
        if prompt_type == "answer":
            prompt = group_prompts["answer"].get(chat_id, DEFAULT_ANSWER_PROMPT)
            logger.info(f"Используется групповой промпт для ответа, chat_id={chat_id}, пользовательский: {'да' if chat_id in group_prompts['answer'] else 'нет'}")
            logger.info(f"Using group prompt for answer, chat_id={chat_id}, custom: {'yes' if chat_id in group_prompts['answer'] else 'no'}")
            return prompt
        elif prompt_type == "summary":
            prompt = group_prompts["summary"].get(chat_id, DEFAULT_SUMMARY_PROMPT)
            logger.info(f"Используется групповой промпт для саммари, chat_id={chat_id}, пользовательский: {'да' if chat_id in group_prompts['summary'] else 'нет'}")
            logger.info(f"Using group prompt for summary, chat_id={chat_id}, custom: {'yes' if chat_id in group_prompts['summary'] else 'no'}")
            return prompt
    # Для личных чатов используем промпты из chat_prompts
    # For private chats use prompts from chat_prompts
    else:
        if prompt_type == "answer":
            prompt = chat_prompts["answer"].get(chat_id, DEFAULT_ANSWER_PROMPT)
            logger.info(f"Используется личный промпт для ответа, chat_id={chat_id}, пользовательский: {'да' if chat_id in chat_prompts['answer'] else 'нет'}")
            logger.info(f"Using private chat prompt for answer, chat_id={chat_id}, custom: {'yes' if chat_id in chat_prompts['answer'] else 'no'}")
            return prompt
        elif prompt_type == "summary":
            prompt = chat_prompts["summary"].get(chat_id, DEFAULT_SUMMARY_PROMPT)
            logger.info(f"Используется личный промпт для саммари, chat_id={chat_id}, пользовательский: {'да' if chat_id in chat_prompts['summary'] else 'нет'}")
            logger.info(f"Using private chat prompt for summary, chat_id={chat_id}, custom: {'yes' if chat_id in chat_prompts['summary'] else 'no'}")
            return prompt

    logger.warning(f"Неизвестный тип промпта: {prompt_type}")
    logger.warning(f"Unknown prompt type: {prompt_type}")
    return ""


# Функции для работы с сообщениями
def save_messages() -> None:
    """Сохраняет историю сообщений в файл JSON и Parquet (если доступно)."""
    try:
        # Преобразуем целочисленные ключи в строки для JSON
        serializable_messages = {str(key): value for key, value in user_messages.items()}

        # Сохраняем в JSON (для обратной совместимости)
        with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_messages, f, ensure_ascii=False, indent=2)

        # Сохраняем в Parquet, если библиотеки доступны
        if PARQUET_AVAILABLE:
            save_messages_to_parquet()

        total_chats = len(user_messages)
        total_messages = sum(len(msgs) for msgs in user_messages.values())
        logger.info(f"История сообщений сохранена: {total_chats} чатов, {total_messages} сообщений")
    except Exception as e:
        logger.error(f"Ошибка при сохранении истории сообщений: {e}")


def save_messages_to_parquet() -> None:
    """Сохраняет историю сообщений в формате Parquet."""
    try:
        # Подготавливаем данные для Parquet
        all_messages = []
        
        for chat_id, messages in user_messages.items():
            for msg in messages:
                # Создаем копию сообщения с добавлением chat_id
                msg_copy = msg.copy()
                if "chat_id" not in msg_copy:
                    msg_copy["chat_id"] = chat_id
                all_messages.append(msg_copy)
        
        if not all_messages:
            logger.info("Нет сообщений для сохранения в Parquet")
            return
            
        # Создаем DataFrame из всех сообщений
        df = pd.DataFrame(all_messages)
        
        # Сохраняем в Parquet
        pq.write_table(pa.Table.from_pandas(df), MESSAGES_PARQUET)
        
        logger.info(f"Сохранено {len(all_messages)} сообщений в формате Parquet")
    except Exception as e:
        logger.error(f"Ошибка при сохранении сообщений в Parquet: {e}")


def clean_nan_values_in_messages() -> None:
    """Очищает NaN значения в истории сообщений."""
    import math
    nan_count = 0
    
    for chat_id, messages in user_messages.items():
        for msg in messages:
            # Проверяем reply_to_index на NaN
            if "reply_to_index" in msg and isinstance(msg["reply_to_index"], float) and math.isnan(msg["reply_to_index"]):
                msg["reply_to_index"] = None
                nan_count += 1
    
    if nan_count > 0:
        logger.warning(f"Очищено {nan_count} NaN значений в истории сообщений")


def load_messages() -> None:
    """Загружает историю сообщений из файла JSON и/или Parquet."""
    global user_messages

    try:
        # Сначала пытаемся загрузить из Parquet, если библиотеки доступны
        if PARQUET_AVAILABLE and os.path.exists(MESSAGES_PARQUET):
            try:
                load_messages_from_parquet()
                # Очищаем NaN значения в истории сообщений
                clean_nan_values_in_messages()
                # Если успешно загрузили из Parquet, возвращаемся
                return
            except Exception as e:
                logger.error(f"Ошибка при загрузке сообщений из Parquet: {e}")
                logger.info("Попытка загрузки из JSON...")
        
        # Загружаем из JSON (если Parquet недоступен или произошла ошибка)
        if os.path.exists(MESSAGES_FILE):
            with open(MESSAGES_FILE, 'r', encoding='utf-8') as f:
                loaded_messages = json.load(f)

                # Преобразуем ключи из строк в целые числа
                converted_messages = {}
                for k, v in loaded_messages.items():
                    chat_id = int(k)
                    
                    # Проверяем формат сообщений
                    if v and isinstance(v[0], str):
                        # Старый формат - конвертируем в новый
                        logger.info(f"Конвертация сообщений из старого формата для чата {chat_id}")
                        converted_messages[chat_id] = []
                        for msg in v:
                            if ": " in msg:
                                sender, text = msg.split(": ", 1)
                                converted_messages[chat_id].append({
                                    "sender": sender,
                                    "text": text,
                                    "reply_to_index": None,
                                    "message_id": None  # Не можем восстановить ID сообщения
                                })
                            else:
                                converted_messages[chat_id].append({
                                    "sender": "Unknown",
                                    "text": msg,
                                    "reply_to_index": None,
                                    "message_id": None
                                })
                    else:
                        # Новый формат - просто копируем
                        converted_messages[chat_id] = v
                
                user_messages = converted_messages
                
                # Очищаем NaN значения в истории сообщений
                clean_nan_values_in_messages()

                total_chats = len(user_messages)
                total_messages = sum(len(msgs) for msgs in user_messages.values())
                logger.info(f"Загружена история сообщений из JSON: {total_chats} чатов, {total_messages} сообщений")
        else:
            logger.info(f"Файл истории сообщений {MESSAGES_FILE} не найден. Используется пустая история.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке истории сообщений: {e}")


def load_messages_from_parquet() -> None:
    """Загружает историю сообщений из Parquet файла."""
    global user_messages
    
    # Читаем Parquet файл в DataFrame
    df = pd.read_parquet(MESSAGES_PARQUET)
    
    # Преобразуем DataFrame в словарь сообщений
    messages_by_chat = {}
    
    # Группируем сообщения по chat_id
    for chat_id, group in df.groupby('chat_id'):
        # Преобразуем каждую строку в словарь и сохраняем в список
        messages = group.to_dict('records')
        messages_by_chat[int(chat_id)] = messages
    
    user_messages = messages_by_chat
    
    total_chats = len(user_messages)
    total_messages = sum(len(msgs) for msgs in user_messages.values())
    logger.info(f"Загружена история сообщений из Parquet: {total_chats} чатов, {total_messages} сообщений")


# Функции для работы с моделями
def save_models() -> None:
    """Сохраняет настройки моделей в файл."""
    try:
        models_data = {
            "user_models": {str(k): v for k, v in user_models.items()},
            "chat_models": {str(k): v for k, v in chat_models.items()}
        }

        with open(MODELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(models_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Настройки моделей сохранены: {len(user_models)} пользователей, {len(chat_models)} чатов")
    except Exception as e:
        logger.error(f"Ошибка при сохранении настроек моделей: {e}")


def load_models() -> None:
    """Загружает настройки моделей из файла."""
    global user_models, chat_models

    try:
        if os.path.exists(MODELS_FILE):
            with open(MODELS_FILE, 'r', encoding='utf-8') as f:
                loaded_models = json.load(f)

                # Проверяем структуру загруженных моделей
                if isinstance(loaded_models, dict) and all(key in loaded_models for key in ["user_models", "chat_models"]):
                    # Преобразуем ключи из строк в целые числа
                    user_models = {int(k): v for k, v in loaded_models["user_models"].items()}
                    chat_models = {int(k): v for k, v in loaded_models["chat_models"].items()}

                    logger.info(f"Загружены настройки моделей: {len(user_models)} пользователей, {len(chat_models)} чатов")
                else:
                    logger.warning("Некорректная структура файла моделей. Используются модели по умолчанию.")
        else:
            logger.info(f"Файл настроек моделей {MODELS_FILE} не найден. Используются модели по умолчанию.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке настроек моделей: {e}")


# Функции для работы с конфигурацией
def save_config() -> None:
    """Сохраняет конфигурацию бота в файл."""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(bot_config, f, ensure_ascii=False, indent=2)

        logger.info(f"Конфигурация бота сохранена: {bot_config}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении конфигурации: {e}")


def save_temperatures() -> None:
    """Сохраняет настройки температуры в файл."""
    try:
        temperatures_data = {
            "chat_temperatures": {str(k): v for k, v in chat_temperatures.items()},
            "chat_factual_temperatures": {str(k): v for k, v in chat_factual_temperatures.items()},
            "chat_summary_temperatures": {str(k): v for k, v in chat_summary_temperatures.items()}
        }

        with open(TEMPERATURES_FILE, 'w', encoding='utf-8') as f:
            json.dump(temperatures_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Настройки температуры сохранены: {len(chat_temperatures)} обычных, {len(chat_factual_temperatures)} фактических, {len(chat_summary_temperatures)} для саммари")
    except Exception as e:
        logger.error(f"Ошибка при сохранении настроек температуры: {e}")


def load_config() -> None:
    """Загружает конфигурацию бота из файла."""
    global bot_config

    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

                # Обновляем только существующие ключи
                for key in bot_config:
                    if key in loaded_config:
                        bot_config[key] = loaded_config[key]

                logger.info(f"Загружена конфигурация бота: {bot_config}")
        else:
            logger.info(f"Файл конфигурации {CONFIG_FILE} не найден. Используется конфигурация по умолчанию.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфигурации: {e}")


# Общая функция для загрузки всех данных
def load_temperatures() -> None:
    """Загружает настройки температуры из файла."""
    global chat_temperatures, chat_factual_temperatures, chat_summary_temperatures
    
    try:
        if os.path.exists(TEMPERATURES_FILE):
            with open(TEMPERATURES_FILE, 'r', encoding='utf-8') as f:
                temperatures_data = json.load(f)
                
                # Преобразуем строковые ключи обратно в целые числа
                if "chat_temperatures" in temperatures_data:
                    chat_temperatures = {int(k): float(v) for k, v in temperatures_data["chat_temperatures"].items()}
                
                if "chat_factual_temperatures" in temperatures_data:
                    chat_factual_temperatures = {int(k): float(v) for k, v in temperatures_data["chat_factual_temperatures"].items()}
                
                if "chat_summary_temperatures" in temperatures_data:
                    chat_summary_temperatures = {int(k): float(v) for k, v in temperatures_data["chat_summary_temperatures"].items()}
                
                logger.info(f"Настройки температуры загружены: {len(chat_temperatures)} обычных, {len(chat_factual_temperatures)} фактических, {len(chat_summary_temperatures)} для саммари")
        else:
            logger.info(f"Файл настроек температуры {TEMPERATURES_FILE} не найден, используются значения по умолчанию")
    except Exception as e:
        logger.error(f"Ошибка при загрузке настроек температуры: {e}")


def load_data() -> None:
    """Загружает все данные из файлов."""
    load_config()
    load_prompts()
    load_messages()
    load_models()
    load_temperatures()

    logger.info("Все данные загружены успешно")


# Функции для горячей перезагрузки модулей
def reload_module(module_name: str) -> bool:
    """
    Перезагружает указанный модуль, если он уже импортирован.
    
    Args:
        module_name: Имя модуля для перезагрузки
        
    Returns:
        True, если модуль был успешно перезагружен, False в противном случае
    """
    try:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            logger.info(f"Модуль {module_name} успешно перезагружен")
            return True
        else:
            logger.warning(f"Модуль {module_name} не найден в sys.modules")
            return False
    except Exception as e:
        logger.error(f"Ошибка при перезагрузке модуля {module_name}: {e}")
        return False


# Словарь для отслеживания времени последнего изменения файлов
file_modification_times = {}

def check_and_reload_modules() -> None:
    """
    Проверяет изменения в файлах модулей и перезагружает их при необходимости.
    """
    modules_to_check = ["llm_client"]
    
    for module_name in modules_to_check:
        try:
            # Получаем путь к файлу модуля
            if module_name in sys.modules:
                module = sys.modules[module_name]
                file_path = module.__file__
                
                if file_path:
                    # Получаем время последнего изменения файла
                    current_mtime = os.path.getmtime(file_path)
                    
                    # Если файл был изменен с момента последней проверки
                    if module_name not in file_modification_times or current_mtime > file_modification_times[module_name]:
                        logger.info(f"Обнаружены изменения в файле {file_path}, перезагрузка модуля {module_name}...")
                        reload_module(module_name)
                        file_modification_times[module_name] = current_mtime
        except Exception as e:
            logger.error(f"Ошибка при проверке изменений модуля {module_name}: {e}")


# Общая функция для сохранения всех данных
def save_data() -> None:
    """Сохраняет все данные в файлы."""
    save_config()
    save_prompts()
    save_messages()
    save_models()
    save_temperatures()

    logger.info("Все данные сохранены успешно")


# Функция для периодического автосохранения данных и проверки изменений модулей
async def auto_save_data():
    """Периодически сохраняет все данные и проверяет изменения модулей."""
    # Интервал проверки изменений модулей (в секундах)
    MODULE_CHECK_INTERVAL = 10
    
    # Счетчик для определения, когда нужно сохранять данные
    save_counter = 0
    save_interval_seconds = bot_config["auto_save_interval"] * 60  # Переводим минуты в секунды
    
    while True:
        try:
            # Ждем короткий интервал для проверки модулей
            await asyncio.sleep(MODULE_CHECK_INTERVAL)
            
            # Проверяем изменения модулей
            check_and_reload_modules()
            
            # Увеличиваем счетчик
            save_counter += MODULE_CHECK_INTERVAL
            
            # Если прошло достаточно времени, сохраняем данные
            if save_counter >= save_interval_seconds:
                # Сохраняем все данные
                save_data()
                logger.info(f"Выполнено автосохранение данных (интервал: {bot_config['auto_save_interval']} мин)")
                
                # Сбрасываем счетчик
                save_counter = 0
                
        except asyncio.CancelledError:
            # Задача была отменена, сохраняем данные перед выходом
            save_data()
            logger.info("Задача автосохранения отменена, данные сохранены")
            break
        except Exception as e:
            logger.error(f"Ошибка при автосохранении данных или проверке модулей: {e}")


async def check_ollama_and_start_bot():
    """Проверяет доступность Ollama и запускает бота."""
    # Проверяем доступность Ollama
    logger.info("Проверка доступности Ollama перед запуском бота...")
    ollama_available = await check_ollama_availability()

    if not ollama_available:
        logger.error("Ollama API недоступен! Убедитесь, что Ollama запущен и доступен по URL: " + LLM_API_URL)
        logger.info("Бот будет запущен, но функции, требующие Ollama, могут не работать.")
    else:
        logger.info("Ollama API доступен и готов к использованию.")

    # Загружаем все данные
    load_data()

    # Создаем приложение и передаем ему токен бота
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN не найден! Убедитесь, что он указан в .env файле.")
        return

    logger.info("Инициализация Telegram бота...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summarize", summarize))
    application.add_handler(CommandHandler("hidden_summary", hidden_summary))
    application.add_handler(CommandHandler("hidden_query", hidden_query))
    application.add_handler(CommandHandler("chatid", chatid_command))
    application.add_handler(CommandHandler("factual", factual_command))
    application.add_handler(CommandHandler("reload", reload_command))

    # Регистрируем обработчик выбора модели
    model_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("model", model_command)],
        states={
            CHOOSING_MODEL_TARGET: [CallbackQueryHandler(model_target_callback, pattern=r"^model_for_")],
            CHOOSING_MODEL: [CallbackQueryHandler(model_callback, pattern=r"^model_")],
            CHOOSING_CHAT: [CallbackQueryHandler(chat_selection_for_model_callback, pattern=r"^chat_")],
            ENTERING_CHAT_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, enter_chat_id_for_model_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )
    application.add_handler(model_conv_handler)

    # Регистрируем обработчик настройки промптов
    prompt_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("prompt", prompt_command)],
        states={
            CHOOSING_CHAT: [CallbackQueryHandler(chat_selection_callback, pattern=r"^prompt_chat_")],
            ENTERING_CHAT_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, enter_chat_id_callback)],
            CHOOSING_PROMPT_TYPE: [CallbackQueryHandler(prompt_type_callback, pattern=r"^prompt_type_")],
            SETTING_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_prompt_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )
    application.add_handler(prompt_conv_handler)

    # Регистрируем обработчик настройки конфигурации
    config_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("config", config_command)],
        states={
            CHOOSING_CONFIG: [CallbackQueryHandler(config_callback, pattern=r"^config_")],
            SETTING_CONFIG: [MessageHandler(filters.TEXT & ~filters.COMMAND, set_config_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
    )
    application.add_handler(config_conv_handler)

    # Регистрируем обработчик сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Регистрируем обработчик для отредактированных сообщений
    application.add_handler(MessageHandler(filters.UpdateType.EDITED_MESSAGE, handle_edited_message))

    # Запускаем бота
    logger.info("Запуск Telegram бота...")
    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    # Запускаем задачу автосохранения данных
    autosave_task = asyncio.create_task(auto_save_data())
    logger.info(f"Запущена задача автосохранения данных (интервал: {bot_config['auto_save_interval']} мин)")

    logger.info("Бот успешно запущен и готов к работе!")

    # Ожидаем сигнала завершения (например, Ctrl+C)
    stop_signal = asyncio.Future()

    # Устанавливаем обработчик сигналов для корректного завершения
    def signal_handler():
        logger.info("Получен сигнал завершения")
        if not stop_signal.done():
            stop_signal.set_result(None)

    # Регистрируем обработчик для сигнала SIGINT (Ctrl+C)
    try:
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)
    except NotImplementedError:
        # Windows не поддерживает add_signal_handler
        logger.warning("Не удалось установить обработчик сигналов. Это нормально для Windows.")

    try:
        # Ожидаем сигнала завершения
        logger.info("Бот работает. Нажмите Ctrl+C для завершения.")
        await stop_signal
    finally:
        # Корректно останавливаем бота
        logger.info("Остановка бота...")

        # Останавливаем задачу автосохранения
        autosave_task.cancel()
        try:
            await autosave_task
        except asyncio.CancelledError:
            pass

        # Сохраняем все данные перед выходом
        save_data()

        # Останавливаем бота
        await application.updater.stop()
        await application.stop()
        logger.info("Бот остановлен.")


def main() -> None:
    """Запуск бота."""
    try:
        # Настраиваем обработку исключений для asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Запускаем асинхронную функцию проверки Ollama и запуска бота
        loop.run_until_complete(check_ollama_and_start_bot())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем.")
    except asyncio.CancelledError:
        logger.info("Задача была отменена.")
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Закрываем цикл событий
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            # Даем задачам время на отмену
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

            loop.close()
            logger.info("Цикл событий закрыт.")
        except Exception as e:
            logger.error(f"Ошибка при закрытии цикла событий: {e}")
            import traceback
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()