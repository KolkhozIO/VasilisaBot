#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import signal
import datetime
import importlib
import time
from typing import Dict, List, Optional, Set, Union, Any

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
import telegram.error
from telegram.ext import (
    Application,
    CommandHandler as TelegramCommandHandler,
    MessageHandler as TelegramMessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
    ConversationHandler,
)

# Import our modules
from config.settings import (
    TELEGRAM_BOT_TOKEN, DEFAULT_MODEL, DEFAULT_BOT_CONFIG,
    CHOOSING_MODEL, SETTING_PROMPT, CHOOSING_PROMPT_TYPE, SETTING_CONFIG,
    CHOOSING_CONFIG, CHOOSING_CHAT, ENTERING_CHAT_ID, CHOOSING_MODEL_TARGET,
    CHOOSING_TEMP_TYPE, CHOOSING_CHAT_FOR_TEMP, ENTERING_CHAT_ID_FOR_TEMP,
    SETTING_TEMPERATURE, BOT_LANGUAGE, DATA_DIR
)
from utils.logging_utils import setup_logging, get_logger
from utils.file_utils import (
    load_user_data, save_user_data, save_messages_to_parquet,
    ensure_data_directories
)
from models.llm_client import LocalLLMClient, check_llm_availability
from models.image_handler import ImageHandler

# Setup logging
log_level = setup_logging()
logger = get_logger(__name__)

# Check if pandas and pyarrow are available
try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    logger.warning("Libraries pandas and pyarrow are not installed. Saving in Parquet format is not available.")

# Storage for messages to summarize
# Format: {chat_id/user_id: [{"sender": "name", "text": "message", "reply_to_index": None or int, "message_id": int}, ...]}
user_messages: Dict[int, List[Dict[str, Any]]] = {}

# Storage for selected models for users and chats
user_models: Dict[int, str] = {}
chat_models: Dict[int, str] = {}

# Storage for tracking users in factual mode
factual_mode_users: Set[int] = set()

# Storage for individual temperature settings for chats
chat_temperatures: Dict[int, float] = {}  # Regular temperature for chats
chat_factual_temperatures: Dict[int, float] = {}  # Factual temperature for chats
chat_summary_temperatures: Dict[int, float] = {}  # Summary temperature for chats

# Storage for user prompts for personal chats
chat_prompts: Dict[str, Dict[int, str]] = {
    "answer": {},  # Prompts for answering questions
    "summary": {}  # Prompts for summarization
}

# Storage for prompts for group chats
group_prompts: Dict[str, Dict[int, str]] = {
    "answer": {},  # Prompts for answering questions
    "summary": {}  # Prompts for summarization
}

# Bot configuration
bot_config = DEFAULT_BOT_CONFIG.copy()

# Initialize LLM client
llm_client = LocalLLMClient(model=DEFAULT_MODEL)

# Initialize image handler
image_handler = ImageHandler(llm_client)

# Load user data
def load_data():
    """Load all user data from files."""
    global user_messages, user_models, chat_models, factual_mode_users
    global chat_temperatures, chat_factual_temperatures, chat_summary_temperatures
    global chat_prompts, group_prompts, bot_config
    
    (
        user_messages, user_models, chat_models, factual_mode_users,
        chat_temperatures, chat_factual_temperatures, chat_summary_temperatures,
        chat_prompts, group_prompts, bot_config
    ) = load_user_data()
    
    logger.info("User data loaded successfully")

# Save user data
async def save_data():
    """Save all user data to files."""
    success = save_user_data(
        user_messages, user_models, chat_models, factual_mode_users,
        chat_temperatures, chat_factual_temperatures, chat_summary_temperatures,
        chat_prompts, group_prompts, bot_config
    )
    
    if success:
        logger.info("User data saved successfully")
    else:
        logger.error("Error saving user data")
    
    # Save messages to Parquet if available
    if PARQUET_AVAILABLE:
        parquet_success = save_messages_to_parquet(user_messages)
        if parquet_success:
            logger.info("Messages saved to Parquet successfully")
        else:
            logger.error("Error saving messages to Parquet")

# Auto-save data periodically
async def auto_save_data(interval_minutes: int = 5):
    """
    Automatically save data at regular intervals.
    
    Args:
        interval_minutes: Interval in minutes between saves
    """
    try:
        while True:
            await asyncio.sleep(interval_minutes * 60)
            logger.info(f"Auto-saving data (interval: {interval_minutes} minutes)")
            try:
                await save_data()
                logger.info("Auto-save completed successfully")
            except Exception as e:
                logger.error(f"Error during auto-save: {e}")
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        logger.info("Auto-save task cancelled")
    except Exception as e:
        logger.error(f"Unexpected error in auto-save task: {e}")

# Clean input text
def clean_input_text(text: str) -> str:
    """
    Clean input text to remove potentially harmful characters.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace potentially harmful characters
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    
    return text

# Main function
def main():
    """Start the bot."""
    # Create data directories
    ensure_data_directories()
    
    # Load user data
    load_data()
    
    # Check if the bot token is valid
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        logger.error("Invalid Telegram bot token. Please set a valid token in config/settings.py")
        sys.exit(1)
    
    # Log bot information and environment variables
    bot_id = TELEGRAM_BOT_TOKEN[-8:] if TELEGRAM_BOT_TOKEN else "unknown"
    logger.info(f"Starting bot with ID: {bot_id}, Language: {BOT_LANGUAGE}, Data directory: {DATA_DIR}")
    logger.info(f"Environment variables:")
    logger.info(f"  TELEGRAM_BOT_TOKEN: {'*' * (len(TELEGRAM_BOT_TOKEN) - 8) + TELEGRAM_BOT_TOKEN[-8:] if TELEGRAM_BOT_TOKEN else 'Not set'}")
    logger.info(f"  BOT_LANGUAGE: {BOT_LANGUAGE}")
    logger.info(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    logger.info(f"  LLM_MODEL: {DEFAULT_MODEL}")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    # Import our handlers
    from handlers.command_handler import CommandHandler
    from handlers.message_handler import MessageHandler
    
    # Create handler instances
    command_handler = CommandHandler(
        llm_client=llm_client,
        user_messages=user_messages,
        user_models=user_models,
        chat_models=chat_models,
        factual_mode_users=factual_mode_users,
        chat_temperatures=chat_temperatures,
        chat_factual_temperatures=chat_factual_temperatures,
        chat_summary_temperatures=chat_summary_temperatures,
        chat_prompts=chat_prompts,
        group_prompts=group_prompts,
        bot_config=bot_config
    )
    
    message_handler = MessageHandler(
        llm_client=llm_client,
        image_handler=image_handler,
        user_messages=user_messages,
        user_models=user_models,
        chat_models=chat_models,
        factual_mode_users=factual_mode_users,
        chat_temperatures=chat_temperatures,
        chat_factual_temperatures=chat_factual_temperatures,
        chat_prompts=chat_prompts,
        group_prompts=group_prompts,
        bot_config=bot_config
    )
    
    # Register command handlers
    application.add_handler(TelegramCommandHandler("start", command_handler.start))
    application.add_handler(TelegramCommandHandler("help", command_handler.help_command))
    application.add_handler(TelegramCommandHandler("factual", command_handler.factual))
    application.add_handler(TelegramCommandHandler("chatid", command_handler.chatid))
    application.add_handler(TelegramCommandHandler("summarize", message_handler.handle_summarize))
    application.add_handler(TelegramCommandHandler("hidden_summary", command_handler.hidden_summary))
    application.add_handler(TelegramCommandHandler("hidden_query", command_handler.hidden_query))
    application.add_handler(TelegramCommandHandler("model", command_handler.model))
    
    # Register message handler for regular messages
    application.add_handler(TelegramMessageHandler(filters.TEXT | filters.PHOTO, message_handler.handle_message))
    
    # Define a function to check LLM availability at startup
    async def check_llm_at_startup(application):
        # The application parameter is passed by the telegram library
        llm_available = await check_llm_availability()
        if not llm_available:
            logger.warning(f"LLM API is not available. The bot will not be able to generate responses.")
        
        # Start auto-save task
        asyncio.create_task(auto_save_data(bot_config["auto_save_interval"]))
    
    # Add the startup function to the application
    try:
        application.post_init = check_llm_at_startup
    except AttributeError:
        # If post_init is not available, run the check immediately
        logger.warning("post_init not available, checking LLM availability immediately")
        # Create a new event loop for this check
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            llm_available = loop.run_until_complete(check_llm_availability())
            if not llm_available:
                logger.warning(f"LLM API is not available. The bot will not be able to generate responses.")
            
            # Start auto-save task in a separate thread
            import threading
            
            def run_auto_save():
                # Create a new event loop for this thread
                save_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(save_loop)
                try:
                    # Run the auto-save coroutine
                    save_loop.run_until_complete(auto_save_data(bot_config["auto_save_interval"]))
                except Exception as e:
                    logger.error(f"Error in auto-save thread: {e}")
                finally:
                    save_loop.close()
            
            # Start the auto-save thread
            auto_save_thread = threading.Thread(target=run_auto_save, daemon=True)
            auto_save_thread.start()
            logger.info("Auto-save task started in a separate thread")
        finally:
            loop.close()
    
    # We'll handle shutdown in the signal handler
    
    # Run the bot with retry logic
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Starting bot (attempt {attempt}/{max_retries})...")
            application.run_polling(drop_pending_updates=True, timeout=30, read_timeout=30, write_timeout=30)
            break  # If we get here, the bot started successfully
        except telegram.error.TimedOut:
            if attempt < max_retries:
                logger.warning(f"Connection to Telegram API timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error("Failed to connect to Telegram API after multiple attempts. Please check your network connection.")
                raise
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Received signal to terminate, saving data...")
        # Create a new event loop for the signal handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the save_data coroutine in this new loop
            loop.run_until_complete(save_data())
            logger.info("Data saved, exiting...")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
        finally:
            loop.close()
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the bot
    main()