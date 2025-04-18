#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

from utils.logging_utils import get_logger
from config.settings import (
    DATA_DIR, PROMPTS_FILE, GROUP_PROMPTS_FILE, 
    MESSAGES_FILE, MODELS_FILE, CONFIG_FILE, 
    TEMPERATURES_FILE, PARQUET_DIR, MESSAGES_PARQUET
)

logger = get_logger(__name__)

def ensure_data_directories():
    """
    Ensure all required data directories exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PARQUET_DIR, exist_ok=True)
    logger.info(f"Data directories created: {DATA_DIR}, {PARQUET_DIR}")

def load_json_file(file_path: str, default_value: Any = None) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        default_value: Default value to return if file doesn't exist or is invalid
        
    Returns:
        Loaded data or default value
    """
    if not os.path.exists(file_path):
        logger.info(f"File {file_path} not found, using default value")
        return default_value
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return default_value

def save_json_file(file_path: str, data: Any) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving {file_path}: {e}")
        return False

def load_user_data():
    """
    Load all user data from files.
    
    Returns:
        Tuple of (user_messages, user_models, chat_models, factual_mode_users, 
                 chat_temperatures, chat_factual_temperatures, chat_summary_temperatures,
                 chat_prompts, group_prompts, bot_config)
    """
    # Create data directories if they don't exist
    ensure_data_directories()
    
    # Load messages
    user_messages = load_json_file(MESSAGES_FILE, {})
    
    # Convert string keys to integers for user_messages
    user_messages = {int(k): v for k, v in user_messages.items()} if user_messages else {}
    
    # Load models
    models_data = load_json_file(MODELS_FILE, {})
    user_models = models_data.get("user_models", {})
    chat_models = models_data.get("chat_models", {})
    
    # Convert string keys to integers for models
    user_models = {int(k): v for k, v in user_models.items()} if user_models else {}
    chat_models = {int(k): v for k, v in chat_models.items()} if chat_models else {}
    
    # Load factual mode users
    factual_mode_users = set(models_data.get("factual_mode_users", []))
    
    # Load temperatures
    temperatures_data = load_json_file(TEMPERATURES_FILE, {})
    chat_temperatures = temperatures_data.get("chat_temperatures", {})
    chat_factual_temperatures = temperatures_data.get("chat_factual_temperatures", {})
    chat_summary_temperatures = temperatures_data.get("chat_summary_temperatures", {})
    
    # Convert string keys to integers for temperatures
    chat_temperatures = {int(k): float(v) for k, v in chat_temperatures.items()} if chat_temperatures else {}
    chat_factual_temperatures = {int(k): float(v) for k, v in chat_factual_temperatures.items()} if chat_factual_temperatures else {}
    chat_summary_temperatures = {int(k): float(v) for k, v in chat_summary_temperatures.items()} if chat_summary_temperatures else {}
    
    # Load prompts
    chat_prompts = load_json_file(PROMPTS_FILE, {"answer": {}, "summary": {}})
    group_prompts = load_json_file(GROUP_PROMPTS_FILE, {"answer": {}, "summary": {}})
    
    # Convert string keys to integers for prompts
    for prompt_type in ["answer", "summary"]:
        chat_prompts[prompt_type] = {int(k): v for k, v in chat_prompts[prompt_type].items()} if prompt_type in chat_prompts else {}
        group_prompts[prompt_type] = {int(k): v for k, v in group_prompts[prompt_type].items()} if prompt_type in group_prompts else {}
    
    # Load bot config
    from config.settings import DEFAULT_BOT_CONFIG
    bot_config = load_json_file(CONFIG_FILE, DEFAULT_BOT_CONFIG)
    
    return (
        user_messages, user_models, chat_models, factual_mode_users,
        chat_temperatures, chat_factual_temperatures, chat_summary_temperatures,
        chat_prompts, group_prompts, bot_config
    )

def save_user_data(
    user_messages: Dict[int, List[Dict[str, Any]]],
    user_models: Dict[int, str],
    chat_models: Dict[int, str],
    factual_mode_users: set,
    chat_temperatures: Dict[int, float],
    chat_factual_temperatures: Dict[int, float],
    chat_summary_temperatures: Dict[int, float],
    chat_prompts: Dict[str, Dict[int, str]],
    group_prompts: Dict[str, Dict[int, str]],
    bot_config: Dict[str, Any]
) -> bool:
    """
    Save all user data to files.
    
    Returns:
        True if all saves were successful, False otherwise
    """
    # Create data directories if they don't exist
    ensure_data_directories()
    
    # Save messages
    messages_saved = save_json_file(MESSAGES_FILE, user_messages)
    
    # Save models
    models_data = {
        "user_models": user_models,
        "chat_models": chat_models,
        "factual_mode_users": list(factual_mode_users)
    }
    models_saved = save_json_file(MODELS_FILE, models_data)
    
    # Save temperatures
    temperatures_data = {
        "chat_temperatures": chat_temperatures,
        "chat_factual_temperatures": chat_factual_temperatures,
        "chat_summary_temperatures": chat_summary_temperatures
    }
    temperatures_saved = save_json_file(TEMPERATURES_FILE, temperatures_data)
    
    # Save prompts
    prompts_saved = save_json_file(PROMPTS_FILE, chat_prompts)
    group_prompts_saved = save_json_file(GROUP_PROMPTS_FILE, group_prompts)
    
    # Save bot config
    config_saved = save_json_file(CONFIG_FILE, bot_config)
    
    # Check if all saves were successful
    all_saved = (
        messages_saved and models_saved and temperatures_saved and
        prompts_saved and group_prompts_saved and config_saved
    )
    
    if all_saved:
        logger.info("All user data saved successfully")
    else:
        logger.error("Some user data could not be saved")
    
    return all_saved

def save_messages_to_parquet(user_messages: Dict[int, List[Dict[str, Any]]]) -> bool:
    """
    Save messages to a Parquet file.
    
    Args:
        user_messages: Dictionary of user messages
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if pandas and pyarrow are available
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Flatten messages for Parquet format
        flattened_messages = []
        
        for chat_id, messages in user_messages.items():
            for message in messages:
                message_copy = message.copy()
                message_copy["chat_id"] = chat_id
                flattened_messages.append(message_copy)
        
        if not flattened_messages:
            logger.info("No messages to save to Parquet")
            return True
        
        # Create DataFrame
        df = pd.DataFrame(flattened_messages)
        
        # Save to Parquet
        df.to_parquet(MESSAGES_PARQUET, index=False)
        logger.info(f"Saved {len(flattened_messages)} messages to {MESSAGES_PARQUET}")
        
        return True
    except ImportError:
        logger.warning("pandas and pyarrow libraries not installed. Saving in Parquet format is not available.")
        return False
    except Exception as e:
        logger.error(f"Error saving messages to Parquet: {e}")
        return False