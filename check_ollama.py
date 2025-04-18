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

logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.info(f"Уровень логирования установлен на: {logging.getLevelName(log_level)}")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api" if LLM_PROVIDER == "ollama" else "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama2")


async def check_ollama_connection():
    """Проверяет соединение с Ollama API."""
    logger.info(f"Проверка соединения с Ollama API по URL: {LLM_API_URL}")

    try:
        # Для Ollama 0.6.x используется endpoint /api/tags
        check_url = f"{LLM_API_URL}/tags" if LLM_API_URL.endswith('/api') else f"{LLM_API_URL}/api/tags"
        logger.info(f"Отправка GET запроса на: {check_url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(check_url, timeout=5) as response:
                    logger.info(f"Получен ответ со статусом: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Получен JSON ответ: {json.dumps(data, indent=2)}")

                        # Формат ответа от /api/models отличается от /tags
                        # Он возвращает список моделей напрямую
                        if isinstance(data, list):
                            models = [model.get("name", model.get("model", "")) for model in data]
                        else:
                            # Для обратной совместимости с возможными изменениями API
                            models = [model.get("name", model.get("model", "")) for model in data.get("models", [])]

                        if models:
                            logger.info(f"Доступные модели: {', '.join(models)}")

                            # Проверяем, доступна ли модель по умолчанию
                            if DEFAULT_MODEL in models:
                                logger.info(f"Модель по умолчанию '{DEFAULT_MODEL}' доступна.")
                            else:
                                logger.warning(f"Модель по умолчанию '{DEFAULT_MODEL}' не найдена среди доступных моделей!")
                                logger.info(f"Доступные модели: {', '.join(models)}")
                        else:
                            logger.warning("Не найдено доступных моделей. Возможно, нужно загрузить модели с помощью команды 'ollama pull <model>'")

                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {response.status}, {error_text}")
                        return False
            except aiohttp.ClientError as e:
                logger.error(f"Ошибка соединения: {e}")
                return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return False

async def check_lmstudio_connection():
    """Проверяет соединение с LM Studio API."""
    logger.info(f"Проверка соединения с LM Studio API по URL: {LLM_API_URL}")

    try:
        # LM Studio использует OpenAI-совместимый API
        check_url = f"{LLM_API_URL}/models"
        logger.info(f"Отправка GET запроса на: {check_url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(check_url, timeout=5) as response:
                    logger.info(f"Получен ответ со статусом: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Получен JSON ответ: {json.dumps(data, indent=2)}")

                        # Извлекаем ID моделей из ответа
                        models = [model.get("id", "") for model in data.get("data", [])]
                        models = [model for model in models if model]  # Фильтруем пустые значения

                        if models:
                            logger.info(f"Доступные модели: {', '.join(models)}")

                            # Проверяем, доступна ли модель по умолчанию
                            # В LM Studio имя модели может отличаться от того, что указано в .env
                            # Поэтому просто выводим список доступных моделей
                            logger.info(f"Модель по умолчанию: '{DEFAULT_MODEL}'")
                            logger.info(f"Убедитесь, что эта модель загружена в LM Studio")
                        else:
                            logger.warning("Не найдено доступных моделей. Убедитесь, что модель загружена в LM Studio")

                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {response.status}, {error_text}")
                        return False
            except aiohttp.ClientError as e:
                logger.error(f"Ошибка соединения: {e}")
                return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return False


async def test_ollama_generation():
    """Тестирует генерацию текста с помощью Ollama."""
    logger.info(f"Тестирование генерации текста с моделью '{DEFAULT_MODEL}' через Ollama")

    try:
        # Для Ollama 0.6.x используется endpoint /api/generate
        if LLM_API_URL.endswith('/api'):
            generate_url = f"{LLM_API_URL}/generate"
        else:
            generate_url = f"{LLM_API_URL}/api/generate"

        logger.info(f"URL для генерации: {generate_url}")

        payload = {
            "model": DEFAULT_MODEL,
            "prompt": "Привет, как дела?",
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False  # Отключаем потоковую передачу для Ollama 0.6.x
        }

        logger.info(f"Отправка запроса с payload: {json.dumps(payload, ensure_ascii=False)}")

        # Устанавливаем таймаут для запросов
        timeout = aiohttp.ClientTimeout(total=15)  # 15 секунд

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(generate_url, json=payload) as response:
                    logger.info(f"Получен ответ со статусом: {response.status}")

                    if response.status == 200:
                        # Ollama 0.6.x использует формат ndjson для потоковой передачи
                        # Читаем ответ как текст, а не как JSON
                        response_text = await response.text()
                        logger.info(f"Получен ответ длиной {len(response_text)} символов")

                        # Обрабатываем ndjson формат (каждая строка - отдельный JSON)
                        try:
                            # Берем последнюю непустую строку как финальный результат
                            lines = [line for line in response_text.split('\n') if line.strip()]
                            if lines:
                                last_line = lines[-1]
                                result = json.loads(last_line)
                                response_text = result.get("response", "")
                                logger.info(f"Получен ответ от модели: '{response_text[:100]}...'")
                                return True
                            else:
                                logger.error("Получен пустой ответ от модели")
                                return False
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка декодирования JSON: {e}")
                            logger.info(f"Полученный ответ: {response_text[:200]}...")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {response.status}, {error_text}")
                        return False
            except aiohttp.ClientError as e:
                logger.error(f"Ошибка соединения: {e}")
                return False
            except asyncio.TimeoutError:
                logger.error("Превышено время ожидания ответа от Ollama API (15 секунд)")
                return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return False

async def test_lmstudio_generation():
    """Тестирует генерацию текста с помощью LM Studio."""
    logger.info(f"Тестирование генерации текста с моделью '{DEFAULT_MODEL}' через LM Studio")

    try:
        # LM Studio использует OpenAI-совместимый API
        generate_url = f"{LLM_API_URL}/chat/completions"
        logger.info(f"URL для генерации: {generate_url}")

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "Привет, как дела?"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }

        logger.info(f"Отправка запроса с payload: {json.dumps(payload, ensure_ascii=False)}")

        # Устанавливаем таймаут для запросов
        timeout = aiohttp.ClientTimeout(total=15)  # 15 секунд

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(generate_url, json=payload) as response:
                    logger.info(f"Получен ответ со статусом: {response.status}")

                    if response.status == 200:
                        # Читаем ответ как JSON
                        result = await response.json()
                        logger.info(f"Получен JSON ответ: {json.dumps(result, ensure_ascii=False, indent=2)}")

                        # Извлекаем текст ответа
                        choices = result.get("choices", [])
                        if choices and len(choices) > 0:
                            response_text = choices[0].get("message", {}).get("content", "")
                            logger.info(f"Получен ответ от модели: '{response_text[:100]}...'")
                            return True
                        else:
                            logger.error("Получен пустой ответ от модели")
                            return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка API: {response.status}, {error_text}")
                        return False
            except aiohttp.ClientError as e:
                logger.error(f"Ошибка соединения: {e}")
                return False
            except asyncio.TimeoutError:
                logger.error("Превышено время ожидания ответа от LM Studio API (15 секунд)")
                return False
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")
        return False


async def main():
    """Основная функция для проверки LLM API."""
    logger.info(f"Начало проверки {LLM_PROVIDER.upper()} API")
    
    # Проверяем соединение с выбранным провайдером
    if LLM_PROVIDER == "ollama":
        connection_ok = await check_ollama_connection()
        
        if connection_ok:
            logger.info("Соединение с Ollama API успешно установлено.")
            
            # Тестируем генерацию текста
            generation_ok = await test_ollama_generation()
            
            if generation_ok:
                logger.info("Тест генерации текста успешно пройден.")
                logger.info("Ollama API полностью работоспособен!")
            else:
                logger.error("Тест генерации текста не пройден.")
                logger.info("Проверьте, что модель корректно загружена и доступна.")
        else:
            logger.error("Не удалось установить соединение с Ollama API.")
            logger.info("Проверьте, что Ollama запущен и доступен по указанному URL.")
            logger.info(f"Текущий URL: {LLM_API_URL}")
    
    elif LLM_PROVIDER == "lmstudio":
        connection_ok = await check_lmstudio_connection()
        
        if connection_ok:
            logger.info("Соединение с LM Studio API успешно установлено.")
            
            # Тестируем генерацию текста
            generation_ok = await test_lmstudio_generation()
            
            if generation_ok:
                logger.info("Тест генерации текста успешно пройден.")
                logger.info("LM Studio API полностью работоспособен!")
            else:
                logger.error("Тест генерации текста не пройден.")
                logger.info("Проверьте, что модель корректно загружена и доступна в LM Studio.")
        else:
            logger.error("Не удалось установить соединение с LM Studio API.")
            logger.info("Проверьте, что LM Studio запущен и доступен по указанному URL.")
            logger.info(f"Текущий URL: {LLM_API_URL}")
    
    else:
        logger.error(f"Неизвестный провайдер LLM: {LLM_PROVIDER}")
        logger.info("Поддерживаемые провайдеры: ollama, lmstudio")
        logger.info("Укажите провайдер в переменной окружения LLM_PROVIDER или в файле .env")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Проверка прервана пользователем.")
    except Exception as e:
        logger.error(f"Ошибка при выполнении проверки: {e}")
        import traceback
        logger.error(traceback.format_exc())