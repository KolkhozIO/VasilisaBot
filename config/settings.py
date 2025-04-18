#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# API settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # Options: "ollama" or "lmstudio"
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api" if LLM_PROVIDER == "ollama" else "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama2")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Language settings
BOT_LANGUAGE = os.getenv("BOT_LANGUAGE", "EN").upper()  # Default to English if not specified
SUPPORTED_LANGUAGES = ["EN", "RU"]

if BOT_LANGUAGE not in SUPPORTED_LANGUAGES:
    print(f"Warning: Unsupported language '{BOT_LANGUAGE}'. Defaulting to 'EN'")
    BOT_LANGUAGE = "EN"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Data directories
DATA_DIR = "data"
PROMPTS_FILE = os.path.join(DATA_DIR, "prompts.json")
GROUP_PROMPTS_FILE = os.path.join(DATA_DIR, "group_prompts.json")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
MODELS_FILE = os.path.join(DATA_DIR, "models.json")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
TEMPERATURES_FILE = os.path.join(DATA_DIR, "temperatures.json")

# Parquet settings
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
MESSAGES_PARQUET = os.path.join(PARQUET_DIR, "messages.parquet")

# Image handling settings
# Use a platform-independent temporary directory for image storage
TEMP_IMAGE_DIR = os.path.join(tempfile.gettempdir(), "summarybot_images")
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Bot configuration defaults
DEFAULT_BOT_CONFIG = {
    "context_messages": 100,  # Default number of messages in context
    "auto_save_interval": 5,  # Auto-save interval in minutes
    "temperature": 0.9,       # Generation temperature (0.1 to 1.0)
    "factual_temperature": 0.5,  # Lower temperature for factual questions
    "summary_temperature": 0.7,  # Temperature for summarization
    "summary_messages": 200    # Maximum number of messages for summarization
}

# Default prompts will be loaded from language_utils.py
# This is done to avoid circular imports
# The actual prompts are defined in utils/language_utils.py

# Conversation handler states
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