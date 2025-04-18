#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv

# Загрузка переменных окружения
# Loading environment variables
load_dotenv(override=True)

# Настройка логирования
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Не вызываем logging.basicConfig, если скрипт запущен как модуль
if __name__ == "__main__":
    # Сбрасываем базовую конфигурацию логирования
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level
    )

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api" if LLM_PROVIDER == "ollama" else "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama2")

# Import language utilities after loading environment variables
from config.settings import BOT_LANGUAGE
from utils.language_utils import get_log_text

logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.info(get_log_text("log_level_set", logging.getLevelName(log_level)))


async def check_ollama_connection():
    """Check connection to Ollama API."""
    logger.info(get_log_text("checking_ollama_connection", LLM_API_URL))

    try:
        # For Ollama 0.6.x and above, use /api/tags endpoint
        check_url = f"{LLM_API_URL}/tags" if LLM_API_URL.endswith('/api') else f"{LLM_API_URL}/api/tags"
        logger.info(get_log_text("sending_get_request", check_url))
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(check_url, timeout=5) as response:
                    logger.info(get_log_text("received_response_status", response.status))
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(get_log_text("received_json_response", json.dumps(data, indent=2)))

                        # Response format from /api/models differs from /tags
                        # It returns the model list directly
                        if isinstance(data, list):
                            models = [model.get("name", model.get("model", "")) for model in data]
                        else:
                            # For backward compatibility with possible API changes
                            models = [model.get("name", model.get("model", "")) for model in data.get("models", [])]

                        if models:
                            logger.info(get_log_text("available_models", ", ".join(models)))

                            # Check if the default model is available
                            if DEFAULT_MODEL in models:
                                logger.info(get_log_text("default_model_available", DEFAULT_MODEL))
                            else:
                                logger.warning(get_log_text("default_model_not_found", DEFAULT_MODEL))
                                logger.info(get_log_text("available_models", ", ".join(models)))
                        else:
                            logger.warning(get_log_text("no_models_found"))

                        return True
                    else:
                        error_text = await response.text()
                        logger.error(get_log_text("api_error", response.status, error_text))
                        return False
            except aiohttp.ClientError as e:
                logger.error(get_log_text("connection_error", e))
                return False
    except Exception as e:
        logger.error(get_log_text("unexpected_error", e))
        return False

async def check_lmstudio_connection():
    """Check connection to LM Studio API."""
    logger.info(get_log_text("checking_lmstudio_connection", LLM_API_URL))

    try:
        # LM Studio uses OpenAI-compatible API
        check_url = f"{LLM_API_URL}/models"
        logger.info(get_log_text("sending_get_request", check_url))
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(check_url, timeout=5) as response:
                    logger.info(get_log_text("received_response_status", response.status))
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(get_log_text("received_json_response", json.dumps(data, indent=2)))

                        # Extract model IDs from the response
                        models = [model.get("id", "") for model in data.get("data", [])]
                        models = [model for model in models if model]  # Filter empty values

                        if models:
                            logger.info(get_log_text("available_models", ", ".join(models)))

                            # In LM Studio, the model name might differ from what's specified in .env
                            # So we just display the list of available models
                            logger.info(get_log_text("default_model_note", DEFAULT_MODEL))
                            logger.info(get_log_text("ensure_model_loaded"))
                        else:
                            logger.warning(get_log_text("no_lmstudio_models"))

                        return True
                    else:
                        error_text = await response.text()
                        logger.error(get_log_text("api_error", response.status, error_text))
                        return False
            except aiohttp.ClientError as e:
                logger.error(get_log_text("connection_error", e))
                return False
    except Exception as e:
        logger.error(get_log_text("unexpected_error", e))
        return False


async def test_ollama_generation():
    """Test text generation using Ollama."""
    logger.info(get_log_text("testing_generation", DEFAULT_MODEL, "Ollama"))

    try:
        # For Ollama 0.6.x, use /api/generate endpoint
        if LLM_API_URL.endswith('/api'):
            generate_url = f"{LLM_API_URL}/generate"
        else:
            generate_url = f"{LLM_API_URL}/api/generate"

        logger.info(get_log_text("generation_url", generate_url))

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": "Hello, how are you?",
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False  # Disable streaming for Ollama 0.6.x
        }

        logger.info(get_log_text("sending_request_payload", json.dumps(payload, ensure_ascii=False)))

        # Set timeout for requests
        timeout = aiohttp.ClientTimeout(total=15)  # 15 seconds

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(generate_url, json=payload) as response:
                    logger.info(get_log_text("received_response_status", response.status))

                    if response.status == 200:
                        # Ollama 0.6.x uses ndjson format for streaming
                        # Read response as text, not as JSON
                        response_text = await response.text()
                        logger.info(get_log_text("response_length", len(response_text)))

                        # Process ndjson format (each line is a separate JSON)
                        try:
                            # Take the last non-empty line as the final result
                            lines = [line for line in response_text.split('\n') if line.strip()]
                            if lines:
                                last_line = lines[-1]
                                result = json.loads(last_line)
                                response_text = result.get("response", "")
                                logger.info(get_log_text("model_response", response_text[:100] + "..."))
                                return True
                            else:
                                logger.error(get_log_text("empty_model_response"))
                                return False
                        except json.JSONDecodeError as e:
                            logger.error(get_log_text("json_decode_error", e))
                            logger.info(get_log_text("received_response", response_text[:200] + "..."))
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(get_log_text("api_error", response.status, error_text))
                        return False
            except aiohttp.ClientError as e:
                logger.error(get_log_text("connection_error", e))
                return False
            except asyncio.TimeoutError:
                logger.error(get_log_text("timeout_error"))
                return False
    except Exception as e:
        logger.error(get_log_text("unexpected_error", e))
        return False

async def test_lmstudio_generation():
    """Test text generation using LM Studio."""
    logger.info(get_log_text("testing_generation", DEFAULT_MODEL, "LM Studio"))

    try:
        # LM Studio uses OpenAI-compatible API
        generate_url = f"{LLM_API_URL}/chat/completions"
        logger.info(get_log_text("generation_url", generate_url))

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }

        logger.info(get_log_text("sending_request_payload", json.dumps(payload, ensure_ascii=False)))

        # Set timeout for requests
        timeout = aiohttp.ClientTimeout(total=15)  # 15 seconds

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(generate_url, json=payload) as response:
                    logger.info(get_log_text("received_response_status", response.status))

                    if response.status == 200:
                        # Read response as JSON
                        result = await response.json()
                        logger.info(get_log_text("received_json_response", json.dumps(result, ensure_ascii=False, indent=2)))

                        # Extract response text
                        choices = result.get("choices", [])
                        if choices and len(choices) > 0:
                            response_text = choices[0].get("message", {}).get("content", "")
                            logger.info(get_log_text("model_response", response_text[:100] + "..."))
                            return True
                        else:
                            logger.error(get_log_text("empty_model_response"))
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(get_log_text("api_error", response.status, error_text))
                        return False
            except aiohttp.ClientError as e:
                logger.error(get_log_text("connection_error", e))
                return False
            except asyncio.TimeoutError:
                logger.error(get_log_text("timeout_error"))
                return False
    except Exception as e:
        logger.error(get_log_text("unexpected_error", e))
        return False


async def main():
    """Main function to check LLM API."""
    logger.info(get_log_text("starting_check", LLM_PROVIDER.upper()))
    
    # Check connection with the selected provider
    if LLM_PROVIDER == "ollama":
        connection_ok = await check_ollama_connection()
        
        if connection_ok:
            logger.info(get_log_text("connection_successful", "Ollama"))
            
            # Test text generation
            generation_ok = await test_ollama_generation()
            
            if generation_ok:
                logger.info(get_log_text("generation_test_passed"))
                logger.info(get_log_text("api_fully_functional", "Ollama"))
            else:
                logger.error(get_log_text("generation_test_failed"))
                logger.info(get_log_text("check_model_loaded"))
        else:
            logger.error(get_log_text("connection_failed", "Ollama"))
            logger.info(get_log_text("check_running", "Ollama"))
            logger.info(get_log_text("current_url", LLM_API_URL))
    
    elif LLM_PROVIDER == "lmstudio":
        connection_ok = await check_lmstudio_connection()
        
        if connection_ok:
            logger.info(get_log_text("connection_successful", "LM Studio"))
            
            # Test text generation
            generation_ok = await test_lmstudio_generation()
            
            if generation_ok:
                logger.info(get_log_text("generation_test_passed"))
                logger.info(get_log_text("api_fully_functional", "LM Studio"))
            else:
                logger.error(get_log_text("generation_test_failed"))
                logger.info(get_log_text("check_model_loaded"))
        else:
            logger.error(get_log_text("connection_failed", "LM Studio"))
            logger.info(get_log_text("check_running", "LM Studio"))
            logger.info(get_log_text("current_url", LLM_API_URL))
    
    else:
        logger.error(get_log_text("unknown_provider", LLM_PROVIDER))
        logger.info(get_log_text("supported_providers"))
        logger.info(get_log_text("specify_provider"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(get_log_text("check_interrupted"))
    except Exception as e:
        logger.error(get_log_text("error_during_check", e))
        import traceback
        logger.error(traceback.format_exc())