#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union

from dotenv import load_dotenv

# Загрузка переменных окружения
# Loading environment variables
load_dotenv(override=True)

# Настройка логирования
log_level_str = os.getenv("LOG_LEVEL", "INFO")
log_level = getattr(logging, log_level_str.upper(), logging.INFO)

# Не вызываем logging.basicConfig, так как он уже вызван в bot.py
logger = logging.getLogger(__name__)
logger.setLevel(log_level)
logger.info(f"Уровень логирования установлен на: {logging.getLevelName(log_level)}")
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "llama2")

logger.info(f"Инициализация LLM клиента с URL: {LLM_API_URL}, модель по умолчанию: {DEFAULT_MODEL}")

    
async def check_ollama_availability(url: str = LLM_API_URL) -> bool:
    """
    Проверяет доступность Ollama API.

    Args:
        url: URL API Ollama

    Returns:
        True, если API доступен, иначе False
    """
    try:
        # Для Ollama 0.6.x используется endpoint /api/tags
        check_url = f"{url}/tags" if url.endswith('/api') else f"{url}/api/tags"
        logger.info(f"Проверка доступности Ollama API по URL: {check_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(check_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Получен ответ: {json.dumps(data, ensure_ascii=False)}")

                    # Формат ответа от /api/tags в Ollama 0.6.x
                    models = [model.get("name", "") for model in data.get("models", [])]
                    models = [model for model in models if model]  # Фильтруем пустые значения

                    logger.info(f"Ollama API доступен. Доступные модели: {', '.join(models)}")
                    return True
                else:
                    logger.error(f"Ollama API недоступен. Статус: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Ошибка при проверке доступности Ollama API: {e}")
        return False


class LocalLLMClient:
    """Клиент для взаимодействия с локальной LLM моделью Ollama."""

    def filter_think_tags(self, text: str) -> str:
        """
        Фильтрует теги <think> из текста, но сохраняет их содержимое в логах.

        Args:
            text: Исходный текст

        Returns:
            Текст без содержимого тегов <think>
        """
        import re

        # Ищем все содержимое между тегами <think> и </think>
        think_tags = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)

        # Логируем содержимое тегов <think> для отладки
        if think_tags:
            logger.debug("Найдены теги <think> в ответе модели:")
            for i, think_content in enumerate(think_tags):
                logger.debug(f"<think> #{i+1}:\n{think_content.strip()}")

        # Удаляем все содержимое между тегами <think> и </think>, включая сами теги
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Ищем одиночные теги <think> до конца текста
        single_think = re.search(r'<think>.*$', filtered_text, flags=re.DOTALL)
        if single_think:
            logger.debug(f"Найден одиночный тег <think> в конце ответа:\n{single_think.group(0)[7:].strip()}")

        # Удаляем одиночные теги <think> до конца текста
        filtered_text = re.sub(r'<think>.*$', '', filtered_text, flags=re.DOTALL)

        # Удаляем пустые строки, которые могли образоваться после удаления тегов
        filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)

        return filtered_text.strip()

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Инициализация клиента.

        Args:
            base_url: URL API Ollama. По умолчанию берется из переменных окружения.
            model: Название модели для использования. По умолчанию берется из переменных окружения.
        """
        self.base_url = base_url or LLM_API_URL
        self.model = model or DEFAULT_MODEL

        # Проверяем, что URL заканчивается на /api
        if self.base_url.endswith('/api'):
            self.generate_url = f"{self.base_url}/generate"
        else:
            self.generate_url = f"{self.base_url}/api/generate"

        logger.info(f"Клиент инициализирован с URL: {self.base_url}, модель: {self.model}")
        logger.info(f"URL для генерации: {self.generate_url}")

    def format_prompt_for_model(self, prompt: str, model: str) -> str:
        """
        Форматирует промпт в соответствии с требованиями конкретной модели.
        
        Args:
            prompt: Исходный промпт
            model: Название модели
            
        Returns:
            Отформатированный промпт
        """
        # Логируем информацию о промпте
        logger.info(f"Форматирование промпта для модели: {model}")
        logger.debug(f"Исходный промпт (первые 200 символов): {prompt[:200]}...")
        
        # Проверяем, является ли модель DeepSeek или Gemma
        if "deepseek" in model.lower() or "gemma" in model.lower():
            model_name = "DeepSeek" if "deepseek" in model.lower() else "Gemma"
            logger.info(f"Обнаружена модель {model_name}: {model}, применяем специальное форматирование")
            
            # Проверяем, содержит ли промпт теги <conversation> и <message>
            import re
            if "<conversation>" in prompt and "<message" in prompt:
                logger.info(f"Обнаружен формат диалога, преобразуем в формат {model_name}")
                
                # Извлекаем всю секцию conversation
                conversation_match = re.search(r'<conversation>(.*?)</conversation>', prompt, re.DOTALL)
                if not conversation_match:
                    logger.warning("Не удалось извлечь секцию conversation, используем базовый формат")
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    return formatted_prompt
                
                conversation_content = conversation_match.group(1).strip()
                
                # Извлекаем все сообщения с их атрибутами
                # Используем более надежное регулярное выражение
                message_patterns = re.findall(r'<message\s+([^>]*)>(.*?)</message>', conversation_content, re.DOTALL)
                
                # Логируем информацию о найденных сообщениях
                logger.info(f"Найдено {len(message_patterns)} сообщений в секции conversation")
                if not message_patterns and conversation_content:
                    # Если сообщения не найдены, но содержимое есть, логируем первые 200 символов для отладки
                    logger.warning(f"Содержимое conversation (первые 200 символов): {conversation_content[:200]}")
                
                if not message_patterns:
                    logger.warning("Не удалось извлечь сообщения из conversation, используем базовый формат")
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    return formatted_prompt
                
                # Форматируем диалог в формате DeepSeek/Gemma
                formatted_prompt = ""
                
                # Словарь для хранения сообщений по индексам
                message_dict = {}
                
                # Сначала собираем все сообщения и их индексы
                for i, (attrs, content) in enumerate(message_patterns):
                    sender_match = re.search(r'sender="([^"]*)"', attrs)
                    sender = sender_match.group(1) if sender_match else "Unknown"
                    
                    # Определяем роль (user или model)
                    role = "model" if sender.lower() == "bot" else "user"
                    
                    # Сохраняем сообщение в словаре
                    message_dict[i] = {
                        "role": role,
                        "content": content.strip(),
                        "sender": sender,
                        "attrs": attrs
                    }
                
                # Теперь обрабатываем сообщения с учетом реплаев
                for i, (attrs, content) in enumerate(message_patterns):
                    # Проверяем, является ли сообщение ответом на другое
                    reply_to_match = re.search(r'reply_to="([^"]*)"', attrs)
                    reply_text_match = re.search(r'reply_text="([^"]*)"', attrs)
                    reply_to_id_match = re.search(r'reply_to_id="([^"]*)"', attrs)
                    
                    role = message_dict[i]["role"]
                    clean_content = message_dict[i]["content"]
                    
                    # Если это ответ на другое сообщение, добавляем контекст
                    # Проверяем наличие либо пары reply_to и reply_text, либо reply_to_id
                    if (reply_to_match and reply_text_match) or reply_to_id_match:
                        # Если есть reply_to и reply_text, используем их
                        if reply_to_match and reply_text_match:
                            reply_to = reply_to_match.group(1)
                            reply_text = reply_text_match.group(1)
                            
                            # Добавляем информацию о том, на что отвечает пользователь
                            clean_content = f"[В ответ на сообщение от {reply_to}: \"{reply_text}\"] {clean_content}"
                        # Если есть только reply_to_id, но нет текста, добавляем только ID
                        elif reply_to_id_match:
                            reply_id = reply_to_id_match.group(1)
                            reply_to = reply_to_match.group(1) if reply_to_match else "Unknown"
                            
                            # Добавляем информацию о том, на что отвечает пользователь (только ID)
                            clean_content = f"[В ответ на сообщение #{reply_id} от {reply_to}] {clean_content}"
                    
                    # Добавляем сообщение в формате модели
                    formatted_prompt += f"<start_of_turn>{role}\n{clean_content}<end_of_turn>\n"
                
                # Извлекаем текущее сообщение (вне тегов conversation)
                current_message_match = re.search(r'</conversation>.*?<message\s+([^>]*)>(.*?)</message>', prompt, re.DOTALL)
                
                # Если не удалось найти текущее сообщение с атрибутами, попробуем найти без атрибутов
                if not current_message_match:
                    current_message_match = re.search(r'</conversation>.*?<message>(.*?)</message>', prompt, re.DOTALL)
                    if current_message_match:
                        # Если нашли сообщение без атрибутов, создаем пустые атрибуты
                        logger.info("Найдено текущее сообщение без атрибутов")
                        current_attrs = ""
                        current_content = current_message_match.group(1).strip()
                    else:
                        logger.warning("Не удалось найти текущее сообщение после тега </conversation>")
                
                if current_message_match and not 'current_attrs' in locals():
                    # Если мы еще не установили current_attrs и current_content
                    if len(current_message_match.groups()) == 2:
                        current_attrs = current_message_match.group(1)
                        current_content = current_message_match.group(2).strip()
                    elif len(current_message_match.groups()) == 1:
                        current_attrs = ""
                        current_content = current_message_match.group(1).strip()
                    
                    # Проверяем, есть ли атрибуты
                    if current_attrs:
                        sender_match = re.search(r'sender="([^"]*)"', current_attrs)
                        current_sender = sender_match.group(1) if sender_match else "Unknown"
                    else:
                        current_sender = "Unknown"
                    
                    # Проверяем, является ли сообщение ответом на другое
                    if current_attrs:
                        reply_to_match = re.search(r'reply_to="([^"]*)"', current_attrs)
                        reply_text_match = re.search(r'reply_text="([^"]*)"', current_attrs)
                        reply_to_id_match = re.search(r'reply_to_id="([^"]*)"', current_attrs)
                    else:
                        reply_to_match = None
                        reply_text_match = None
                        reply_to_id_match = None
                    
                    # Определяем роль для текущего сообщения
                    current_role = "model" if current_sender.lower() == "bot" else "user"
                    
                    # Если это ответ на другое сообщение, добавляем контекст
                    # Проверяем наличие либо пары reply_to и reply_text, либо reply_to_id
                    if (reply_to_match and reply_text_match) or reply_to_id_match:
                        # Если есть reply_to и reply_text, используем их
                        if reply_to_match and reply_text_match:
                            reply_to = reply_to_match.group(1)
                            reply_text = reply_text_match.group(1)
                            
                            # Добавляем информацию о том, на что отвечает пользователь
                            current_content = f"[В ответ на сообщение от {reply_to}: \"{reply_text}\"] {current_content}"
                        # Если есть только reply_to_id, но нет текста, добавляем только ID
                        elif reply_to_id_match:
                            reply_id = reply_to_id_match.group(1)
                            reply_to = reply_to_match.group(1) if reply_to_match else "Unknown"
                            
                            # Добавляем информацию о том, на что отвечает пользователь (только ID)
                            current_content = f"[В ответ на сообщение #{reply_id} от {reply_to}] {current_content}"
                    
                    # Добавляем текущее сообщение
                    formatted_prompt += f"<start_of_turn>{current_role}\n{current_content}<end_of_turn>\n"
                else:
                    # Если не нашли текущее сообщение, используем весь оставшийся текст
                    remaining_text = re.sub(r'<conversation>.*?</conversation>', '', prompt, flags=re.DOTALL).strip()
                    if remaining_text:
                        logger.info(f"Используем оставшийся текст после тега </conversation>: {remaining_text[:100]}...")
                        # Очищаем от HTML-тегов
                        clean_remaining = re.sub(r'<[^>]+>', ' ', remaining_text)
                        clean_remaining = re.sub(r'\s+', ' ', clean_remaining).strip()
                        
                        if clean_remaining:
                            logger.info(f"Очищенный оставшийся текст: {clean_remaining[:100]}...")
                            formatted_prompt += f"<start_of_turn>user\n{clean_remaining}<end_of_turn>\n"
                    else:
                        # Если нет оставшегося текста, используем последнее сообщение из диалога
                        if message_patterns:
                            last_attrs, last_content = message_patterns[-1]
                            logger.info(f"Используем последнее сообщение из диалога: {last_content[:100]}...")
                            formatted_prompt += f"<start_of_turn>user\n{last_content}<end_of_turn>\n"
                        else:
                            # Если нет ни сообщений, ни оставшегося текста, используем весь промпт
                            logger.warning("Не найдено ни текущего сообщения, ни сообщений в диалоге, используем весь промпт")
                            clean_prompt = re.sub(r'<[^>]+>', ' ', prompt)
                            clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip()
                            formatted_prompt += f"<start_of_turn>user\n{clean_prompt}<end_of_turn>\n"
                
                # Добавляем финальный тег для ответа модели
                formatted_prompt += "<start_of_turn>model\n"
                
                logger.debug(f"Отформатированный промпт для {model_name} (диалог): {formatted_prompt[:200]}...")
                return formatted_prompt
            else:
                # Базовый формат для DeepSeek/Gemma без диалога
                # Очищаем промпт от HTML-тегов
                clean_prompt = re.sub(r'<[^>]+>', '', prompt)
                
                # Форматируем промпт для модели
                formatted_prompt = f"<start_of_turn>user\n{clean_prompt}<end_of_turn>\n<start_of_turn>model\n"
                
                logger.debug(f"Отформатированный промпт для {model_name} (базовый): {formatted_prompt[:200]}...")
                return formatted_prompt
        
        # Для остальных моделей возвращаем промпт без изменений
        return prompt
    
    async def get_completion(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Асинхронно получает ответ от модели Ollama.

        Args:
            prompt: Текст запроса
            max_tokens: Максимальное количество токенов в ответе
            temperature: Температура (креативность) ответа
            stop: Список строк, при обнаружении которых генерация останавливается
            model: Модель для использования (если отличается от модели по умолчанию)

        Returns:
            Ответ модели в виде строки
        """
        used_model = model or self.model
        logger.info(f"Отправка запроса к Ollama, модель: {used_model}, max_tokens: {max_tokens}")
        
        # Форматируем промпт в соответствии с моделью
        formatted_prompt = self.format_prompt_for_model(prompt, used_model)
        
        payload = {
            "model": used_model,
            "prompt": formatted_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False  # Отключаем потоковую передачу для Ollama 0.6.x
        }

        if stop:
            payload["stop"] = stop

        logger.debug(f"URL запроса: {self.generate_url}")
        logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

        try:
            # Не устанавливаем таймаут для запросов, чтобы модель могла генерировать ответ сколько угодно долго
            async with aiohttp.ClientSession() as session:
                logger.debug("Создана сессия aiohttp без таймаута")

                try:
                    logger.debug("Отправка POST запроса к Ollama API")
                    async with session.post(self.generate_url, json=payload) as response:
                        logger.debug(f"Получен ответ от Ollama API, статус: {response.status}")

                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Ошибка API Ollama: {response.status}, {error_text}")
                            raise Exception(f"Ошибка API Ollama: {response.status}, {error_text}")

                        # Ollama 0.6.x использует формат ndjson для потоковой передачи
                        # Читаем ответ как текст, а не как JSON
                        response_text_raw = await response.text()
                        logger.debug(f"Получен ответ длиной {len(response_text_raw)} символов")

                        # Обрабатываем ndjson формат (каждая строка - отдельный JSON)
                        try:
                            # Берем последнюю непустую строку как финальный результат
                            lines = [line for line in response_text_raw.split('\n') if line.strip()]
                            if not lines:
                                logger.error("Получен пустой ответ от модели")
                                return ""

                            last_line = lines[-1]
                            result = json.loads(last_line)
                            logger.debug(f"Финальный JSON ответ: {json.dumps(result, ensure_ascii=False)}")

                            response_text = result.get("response", "")
                            
                            # Обрабатываем ответ в зависимости от модели
                            if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
                                model_name = "DeepSeek" if "deepseek" in used_model.lower() else "Gemma"
                                # Удаляем специальные теги из ответа
                                logger.info(f"Обрабатываем ответ от модели {model_name}: {used_model}")
                                
                                # Сохраняем оригинальную длину для логирования
                                original_length = len(response_text)
                                
                                # Удаляем теги <start_of_turn> и <end_of_turn> из ответа
                                import re
                                
                                # Более агрессивная очистка от тегов модели
                                patterns = [
                                    r'<start_of_turn>model\n',
                                    r'<start_of_turn>model',
                                    r'<end_of_turn>',
                                    r'<start_of_turn>user.*?<end_of_turn>\n?',
                                    r'<start_of_turn>user.*?$',  # Если тег открыт, но не закрыт
                                    r'<start_of_turn>.*?<end_of_turn>\n?',  # Любые другие теги
                                ]
                                
                                for pattern in patterns:
                                    response_text = re.sub(pattern, '', response_text, flags=re.DOTALL)
                                
                                # Удаляем лишние пробелы и переносы строк
                                response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
                                response_text = response_text.strip()
                                
                                logger.info(f"Ответ от {model_name} после обработки тегов: было {original_length}, стало {len(response_text)} символов")
                            
                            # Фильтруем теги <think> из ответа
                            filtered_response = self.filter_think_tags(response_text)

                            # Вычисляем разницу в длине до и после фильтрации
                            diff = len(response_text) - len(filtered_response)
                            if diff > 0:
                                logger.info(f"Получен ответ от модели длиной {len(response_text)} символов, удалено {diff} символов из тегов <think>, итоговая длина {len(filtered_response)} символов")
                            else:
                                logger.info(f"Получен ответ от модели длиной {len(response_text)} символов (теги <think> не найдены)")
                            return filtered_response
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка декодирования JSON: {e}")
                            logger.debug(f"Полученный ответ: {response_text_raw[:200]}...")
                            # В случае ошибки декодирования возвращаем сырой текст
                            return response_text_raw
                except aiohttp.ClientError as e:
                    logger.error(f"Ошибка соединения с Ollama API: {e}")
                    raise Exception(f"Ошибка соединения с Ollama API: {e}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе к Ollama: {e}")
            raise

    async def get_summary(self, prompt: str, max_length: int = 200, model: Optional[str] = None, temperature: float = 0.3) -> str:
        """
        Создает саммари текста.

        Args:
            prompt: Полный промпт для суммаризации (уже отформатированный)
            max_length: Максимальная длина саммари
            model: Модель для использования (если отличается от модели по умолчанию)
            temperature: Температура генерации (от 0.1 до 1.0)

        Returns:
            Саммари текста
        """
        used_model = model or self.model
        
        # Определяем стоп-слова в зависимости от модели
        stop_words = ["###"]
        
        # Для DeepSeek и Gemma добавляем специальные стоп-слова
        if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
            stop_words.extend([
                "<end_of_turn>", 
                "<start_of_turn>user", 
                "<start_of_turn>", 
                "user:", 
                "User:", 
                "Human:"
            ])
        
        return await self.get_completion(
            prompt=prompt,
            max_tokens=max_length,
            temperature=temperature,
            stop=stop_words,
            model=used_model
        )

    async def answer_question(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Отвечает на вопрос пользователя.

        Args:
            prompt: Полный промпт для ответа (уже отформатированный)
            model: Модель для использования (если отличается от модели по умолчанию)
            temperature: Температура генерации (от 0.1 до 1.0)

        Returns:
            Ответ на вопрос
        """
        used_model = model or self.model
        
        # Определяем стоп-слова в зависимости от модели
        stop_words = None
        
        # Для DeepSeek и Gemma добавляем специальные стоп-слова
        if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
            stop_words = [
                "<end_of_turn>", 
                "<start_of_turn>user", 
                "<start_of_turn>", 
                "user:", 
                "User:", 
                "Human:"
            ]
        
        return await self.get_completion(
            prompt=prompt,
            max_tokens=1024,
            temperature=temperature,
            stop=stop_words,
            model=used_model
        )
        
    async def answer_question_with_image(self, prompt: str, image_path: str, model: Optional[str] = None, temperature: float = 0.7, message_id: Optional[str] = None) -> str:
        """
        Отвечает на вопрос пользователя с учетом изображения.

        Args:
            prompt: Полный промпт для ответа (уже отформатированный)
            image_path: Путь к файлу изображения
            model: Модель для использования (если отличается от модели по умолчанию)
            temperature: Температура генерации (от 0.1 до 1.0)
            message_id: Идентификатор сообщения для привязки изображения (опционально)

        Returns:
            Ответ на вопрос с учетом изображения
        """
        # Импортируем необходимые модули
        import os
        import json
        import base64
        import imghdr
        import shutil
        import uuid
        from datetime import datetime
        
        used_model = model or self.model
        
        # Проверяем, существует ли файл изображения
        if not os.path.exists(image_path):
            logger.error(f"Файл изображения не найден: {image_path}")
            return f"Ошибка: файл изображения не найден: {image_path}"
        
        # Определяем директорию для хранения изображений
        storage_dir = "/mnt/d/distrib/images"
        
        # Создаем директорию, если она не существует
        os.makedirs(storage_dir, exist_ok=True)
        
        # Генерируем уникальное имя файла на основе message_id или uuid
        if message_id:
            # Используем message_id и текущую дату для создания уникального имени
            file_id = f"{message_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        else:
            # Если message_id не предоставлен, используем UUID
            file_id = str(uuid.uuid4())
        
        # Определяем тип изображения
        with open(image_path, "rb") as img_file:
            img_type = imghdr.what(None, h=img_file.read(2048)) or "jpeg"
        
        # Создаем новое имя файла с правильным расширением
        new_filename = f"{file_id}.{img_type}"
        new_image_path = os.path.join(storage_dir, new_filename)
        
        # Копируем изображение в постоянное хранилище
        try:
            shutil.copy2(image_path, new_image_path)
            logger.info(f"Изображение сохранено в: {new_image_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении изображения: {e}")
            # Продолжаем с оригинальным путем, если не удалось скопировать
            new_image_path = image_path
        
        # Преобразуем путь для Ollama (из /mnt/d/distrib в d:\distrib)
        if new_image_path.startswith("/mnt/d/"):
            ollama_path = new_image_path.replace("/mnt/d/", "d:\\").replace("/", "\\")
            logger.info(f"Путь для Ollama: {ollama_path}")
        else:
            ollama_path = new_image_path
            logger.warning(f"Не удалось преобразовать путь для Ollama, используем исходный: {ollama_path}")
        
        # Используем преобразованный путь для дальнейшей работы
        image_path = new_image_path
        
        # Логируем информацию о запросе
        logger.info(f"Отправка запроса с изображением к модели {used_model} с температурой {temperature}")
        logger.debug(f"Промпт (первые 200 символов): {prompt[:200]}...")
        logger.info(f"Путь к изображению: {image_path}")
        
        # Проверяем, является ли сервер локальным
        is_local_server = self.generate_url.startswith("http://localhost") or self.generate_url.startswith("http://127.0.0.1")
        
        # Проверяем длину промпта и обрезаем его, если он слишком длинный и сервер не локальный
        max_prompt_length = 1500  # Устанавливаем безопасный лимит
        if len(prompt) > max_prompt_length and not is_local_server:
            logger.warning(f"Промпт слишком длинный ({len(prompt)} символов), обрезаем до {max_prompt_length}")
            # Сохраняем начало и конец промпта
            prompt_start = prompt[:max_prompt_length // 2]
            prompt_end = prompt[-max_prompt_length // 2:]
            prompt = prompt_start + "\n...\n" + prompt_end
            logger.info(f"Промпт обрезан до {len(prompt)} символов")
            logger.debug(f"Обрезанный промпт (первые 200 символов): {prompt[:200]}...")
        elif len(prompt) > max_prompt_length and is_local_server:
            logger.info(f"Промпт длинный ({len(prompt)} символов), но не обрезаем, так как сервер локальный")
        
        try:
            # Проверяем, является ли сервер локальным
            is_local_server = self.generate_url.startswith("http://localhost") or self.generate_url.startswith("http://127.0.0.1")
            
            # Всегда кодируем изображение в base64, независимо от типа сервера
            logger.info(f"Кодируем изображение в base64: {image_path}")
            
            # Читаем изображение и определяем его тип
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                img_type = imghdr.what(None, h=image_bytes) or "jpeg"
                logger.info(f"Определен тип изображения: {img_type}")
                
                # Кодируем изображение в base64
                image_base64_raw = base64.b64encode(image_bytes).decode("utf-8")
                image_base64_with_mime = f"data:image/{img_type};base64,{image_base64_raw}"
                image_base64 = image_base64_raw  # Используем версию без MIME-типа
                logger.info(f"Изображение закодировано в base64, длина: {len(image_base64_raw)} символов")
                
                # Проверяем корректность base64
                try:
                    # Проверяем, можно ли декодировать обратно
                    test_decode = base64.b64decode(image_base64)
                    logger.info(f"Base64 проверен, корректен")
                except Exception as e:
                    logger.error(f"Ошибка при проверке base64: {e}")
                    # Пробуем исправить base64, удаляя некорректные символы
                    image_base64 = ''.join(c for c in image_base64 if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                    
                    # Проверяем длину base64 и добавляем padding если нужно
                    padding = len(image_base64) % 4
                    if padding:
                        image_base64 += '=' * (4 - padding)
                    
                    # Проверяем еще раз
                    try:
                        test_decode = base64.b64decode(image_base64)
                        logger.info(f"Base64 исправлен и проверен, корректен")
                    except Exception as e:
                        logger.error(f"Не удалось исправить base64: {e}")
                        # Пробуем перекодировать изображение заново
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        logger.info(f"Изображение перекодировано в base64")
                        logger.info(f"Base64 исправлен, новая длина: {len(image_base64)} символов")
                    
                    # Добавляем MIME-тип
                    image_base64_with_mime = f"data:image/{img_type};base64,{image_base64}"
                    logger.info(f"Добавлен MIME-тип к изображению: data:image/{img_type};base64,...")
                    
                    # Для совместимости сохраняем обе версии
                    image_base64_raw = image_base64  # Без MIME-типа
                    image_base64 = image_base64_with_mime  # С MIME-типом
                
                # Устанавливаем флаг использования локального пути
                use_local_path = False
            
            # Определяем стоп-слова в зависимости от модели
            stop_words = None
            
            # Для DeepSeek и Gemma добавляем специальные стоп-слова
            if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
                stop_words = [
                    "<end_of_turn>", 
                    "<start_of_turn>user", 
                    "<start_of_turn>", 
                    "user:", 
                    "User:", 
                    "Human:"
                ]
                
            # Специальная обработка для разных моделей
            is_gemma3 = "gemma3" in used_model.lower()
            is_llava = any(x in used_model.lower() for x in ["llava", "bakllava"])
            
            if is_gemma3:
                logger.info(f"Обнаружена модель gemma3: {used_model}")
                
                # Для gemma3 добавляем специальный префикс к промпту
                if not prompt.startswith("<image>"):
                    prompt = f"<image>\n{prompt}"
                    logger.info("Добавлен тег <image> в начало промпта для gemma3")
            
            elif is_llava:
                logger.info(f"Обнаружена модель llava: {used_model}")
                
                # Для llava используем специальный формат промпта
                # Убедимся, что промпт не содержит специальных инструкций
                if not "<image>" in prompt and not "[IMAGE]" in prompt:
                    # Добавляем USER: и ASSISTANT: для llava
                    if not prompt.startswith("USER:"):
                        prompt = f"USER: {prompt}\nASSISTANT:"
                        logger.info("Добавлены теги USER: и ASSISTANT: для llava")
            
            # Проверяем размер изображения
            image_size = os.path.getsize(image_path)
            logger.info(f"Размер изображения: {image_size} байт")
            
            # Если изображение слишком большое, уменьшаем его
            max_image_size = 4 * 1024 * 1024  # 4 МБ
            max_dimension = 1024  # Максимальный размер стороны изображения
            
            # Проверяем размер и уменьшаем изображение при необходимости
            try:
                # Импортируем PIL (Pillow)
                from PIL import Image
                import io
                has_pil = True
                
                # Открываем изображение
                img = Image.open(image_path)
                width, height = img.size
                logger.info(f"Размер изображения: {width}x{height} пикселей")
                
                # Проверяем, нужно ли уменьшать изображение
                resize_needed = image_size > max_image_size or width > max_dimension or height > max_dimension
                
                if resize_needed and not is_local_server:
                    # Уменьшаем изображение только для удаленных серверов
                    logger.warning(f"Изображение слишком большое ({image_size} байт, {width}x{height}), уменьшаем для удаленного сервера")
                    
                    # Уменьшаем размер изображения, сохраняя пропорции
                    if width > max_dimension or height > max_dimension:
                        if width > height:
                            new_width = max_dimension
                            new_height = int(height * (max_dimension / width))
                        else:
                            new_height = max_dimension
                            new_width = int(width * (max_dimension / height))
                        
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        logger.info(f"Изображение уменьшено до {new_width}x{new_height}")
                    
                    # Сохраняем изображение во временный буфер с оптимизацией качества
                    buffer = io.BytesIO()
                    
                    # Определяем формат для сохранения
                    save_format = img.format if img.format else "JPEG"
                    
                    # Если формат JPEG или PNG, можем управлять качеством
                    if save_format == "JPEG":
                        img.save(buffer, format=save_format, quality=85, optimize=True)
                    elif save_format == "PNG":
                        img.save(buffer, format=save_format, optimize=True, compress_level=9)
                    else:
                        img.save(buffer, format=save_format)
                    
                    buffer.seek(0)
                    
                    # Кодируем изображение в base64 только для удаленных серверов
                    if not is_local_server:
                        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                        
                        # Проверяем новый размер
                        new_size = len(image_base64)
                        logger.info(f"Размер уменьшенного изображения в base64: {new_size} байт")
                        
                        # Если изображение все еще слишком большое, уменьшаем качество еще сильнее
                        if new_size > max_image_size:
                            logger.warning(f"Изображение все еще слишком большое ({new_size} байт), уменьшаем качество")
                            buffer = io.BytesIO()
                            img.save(buffer, format="JPEG", quality=50, optimize=True)
                            buffer.seek(0)
                            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                            logger.info(f"Размер изображения после сильного сжатия: {len(image_base64)} байт")
                        
                        # Обновляем переменные с base64 кодированием
                        image_base64_raw = image_base64
                        image_base64_with_mime = f"data:image/{img_type};base64,{image_base64}"
                        image_base64 = image_base64_with_mime
                elif resize_needed and is_local_server:
                    logger.info(f"Изображение большое ({image_size} байт, {width}x{height}), но не уменьшаем для локального сервера")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения: {e}")
                # Продолжаем с оригинальным изображением
            
            # Проверяем размер изображения для логирования
            if not is_local_server:
                logger.info(f"Размер base64 изображения: {len(image_base64)} символов")
                logger.info(f"Размер base64 изображения без MIME: {len(image_base64_raw)} символов")
            else:
                logger.info(f"Используем локальный путь к изображению: {ollama_path if 'ollama_path' in locals() else image_path}")
            
            # Определяем формат запроса для изображений в зависимости от версии Ollama
            # Пробуем разные форматы для совместимости с разными версиями
            
            # Для запросов с изображениями отключаем потоковую передачу, так как это может вызывать проблемы
            use_stream = False  # Отключаем stream для запросов с изображениями
            
            if not is_local_server:
                logger.info(f"Обнаружен удаленный сервер Ollama: {self.generate_url}")
                logger.info("Отправляем изображение через base64 кодирование")
            else:
                logger.info(f"Обнаружен локальный сервер Ollama: {self.generate_url}")
                logger.info("Используем локальный путь к изображению")
            
            # Создаем несколько вариантов запросов для разных версий Ollama
            request_variants = []
            
            # Всегда используем base64 кодирование для изображений
            # Это обеспечивает совместимость со всеми версиями Ollama и избегает проблем с путями к файлам
            
            # Обрабатываем тег <image> для совместимости с Ollama
            # Формат [img-0] используется в Ollama для указания на первое изображение в массиве images
            # Перемещаем ссылку на изображение в конец промпта
            if "<image>" in prompt:
                # Удаляем тег <image> из промпта
                prompt = prompt.replace("<image>", "")
                # Добавляем [img-0] в конец промпта
                prompt = prompt.strip() + "\n\n[img-0]"
                logger.info(f"Ссылка на изображение [img-0] добавлена в конец промпта")
                logger.debug(f"Финальный промпт (первые 200 символов): {prompt[:200]}...")
            
            # Вариант 1: Ollama 0.7.x (images как массив без MIME)
            request_data_1 = {
                "model": used_model,
                "prompt": prompt,
                "images": [image_base64_raw],  # Без MIME-типа
                "stream": use_stream,
                "temperature": temperature,
                "max_tokens": 1024
            }
            if stop_words:
                request_data_1["stop"] = stop_words
            request_variants.append(("images как массив без MIME", request_data_1))
            
            # Вариант 2: Ollama 0.6.x (image как строка без MIME)
            request_data_2 = {
                "model": used_model,
                "prompt": prompt,
                "image": image_base64_raw,  # Без MIME-типа
                "stream": use_stream,
                "temperature": temperature,
                "max_tokens": 1024
            }
            if stop_words:
                request_data_2["stop"] = stop_words
            request_variants.append(("image как строка без MIME", request_data_2))
            
            # Вариант 3: Ollama 0.7.x и выше (images как массив с MIME)
            request_data_3 = {
                "model": used_model,
                "prompt": prompt,
                "images": [image_base64],  # С MIME-типом
                "stream": use_stream,
                "temperature": temperature,
                "max_tokens": 1024
            }
            if stop_words:
                request_data_3["stop"] = stop_words
            request_variants.append(("images как массив с MIME", request_data_3))
            
            # Вариант 4: Ollama 0.6.x (image как строка с MIME)
            request_data_4 = {
                "model": used_model,
                "prompt": prompt,
                "image": image_base64,  # С MIME-типом
                "stream": use_stream,
                "temperature": temperature,
                "max_tokens": 1024
            }
            if stop_words:
                request_data_4["stop"] = stop_words
            request_variants.append(("image как строка с MIME", request_data_4))
            
            # Вариант 5: Ollama с images как строкой без MIME
            request_data_5 = {
                "model": used_model,
                "prompt": prompt,
                "images": image_base64_raw,  # Без MIME-типа, не массив, а строка
                "stream": use_stream,
                "temperature": temperature,
                "max_tokens": 1024
            }
            if stop_words:
                request_data_5["stop"] = stop_words
            request_variants.append(("images как строка без MIME", request_data_5))
            
            # Вариант 6: Ollama 0.7.x с base64 напрямую (без декодирования)
            try:
                # Читаем файл напрямую для этого варианта
                with open(image_path, "rb") as direct_file:
                    direct_image_data = direct_file.read()
                    
                request_data_6 = {
                    "model": used_model,
                    "prompt": prompt,
                    "images": [base64.b64encode(direct_image_data).decode('utf-8')],  # Прямое кодирование
                    "stream": use_stream,
                    "temperature": temperature,
                    "max_tokens": 1024
                }
                if stop_words:
                    request_data_6["stop"] = stop_words
                request_variants.append(("прямое кодирование", request_data_6))
            except Exception as e:
                logger.error(f"Ошибка при создании варианта с прямым кодированием: {e}")
            
            # Специальные варианты для конкретных моделей
            if is_gemma3:
                # Для gemma3 всегда используем base64 кодирование для изображений
                # Это обеспечивает совместимость со всеми версиями Ollama и избегает проблем с путями к файлам
                # Вариант 1: gemma3 с base64 и raw
                request_data_gemma = {
                    "model": used_model,
                    "prompt": prompt,
                    "images": [image_base64_raw],  # Без MIME-типа
                    "stream": use_stream,
                    "temperature": temperature,
                    "max_tokens": 1024,
                    "raw": True  # Специальный параметр для некоторых моделей
                }
                if stop_words:
                    request_data_gemma["stop"] = stop_words
                request_variants.append(("gemma3 с base64 и raw", request_data_gemma))
                
                # Вариант 2: gemma3 с base64 без raw
                request_data_gemma2 = {
                    "model": used_model,
                    "prompt": prompt,
                    "images": [image_base64_raw],  # Без MIME-типа
                    "stream": use_stream,
                    "temperature": temperature,
                    "max_tokens": 1024
                }
                if stop_words:
                    request_data_gemma2["stop"] = stop_words
                request_variants.append(("gemma3 с base64 без raw", request_data_gemma2))
                
            if is_llava:
                # Для llava всегда используем base64 кодирование для изображений
                # Это обеспечивает совместимость со всеми версиями Ollama и избегает проблем с путями к файлам
                # Вариант 1: llava с base64 и num_ctx
                request_data_llava = {
                    "model": used_model,
                    "prompt": prompt,
                    "images": [image_base64_raw],  # Без MIME-типа
                    "stream": use_stream,
                    "temperature": temperature,
                    "max_tokens": 1024,
                    "options": {
                        "num_ctx": 4096  # Увеличенный контекст для llava
                    }
                }
                if stop_words:
                    request_data_llava["stop"] = stop_words
                request_variants.append(("llava с base64 и num_ctx", request_data_llava))
                
                # Вариант 2: llava с base64 без num_ctx
                request_data_llava2 = {
                    "model": used_model,
                    "prompt": prompt,
                    "images": [image_base64_raw],  # Без MIME-типа
                    "stream": use_stream,
                    "temperature": temperature,
                    "max_tokens": 1024
                }
                if stop_words:
                    request_data_llava2["stop"] = stop_words
                request_variants.append(("llava с base64 без num_ctx", request_data_llava2))
            
            # Используем первый вариант по умолчанию
            request_format, request_data = request_variants[0]
            logger.info(f"Используем формат запроса: {request_format}")
            
            if stop_words:
                request_data["stop"] = stop_words
            
            # Проверяем, поддерживает ли модель изображения
            vision_models = ["llava", "bakllava", "gemma", "phi3", "claude", "gpt4", "cogvlm", "qwen"]
            is_vision_model = any(model_name in used_model.lower() for model_name in vision_models)
            
            if not is_vision_model:
                logger.warning(f"Модель {used_model} может не поддерживать обработку изображений")
                logger.info("Добавляем текстовое описание о том, что изображение не может быть обработано")
                # Добавляем предупреждение в промпт
                prompt = f"[ПРЕДУПРЕЖДЕНИЕ: Вы получили запрос с изображением, но ваша модель ({used_model}) может не поддерживать обработку изображений. Пожалуйста, сообщите пользователю об этом.]\n\n{prompt}"
            
            # Если Ollama запущена локально, проверяем, что путь был успешно преобразован
            if is_local_server and 'ollama_path' in locals() and not ollama_path.startswith("d:\\"):
                logger.info(f"Ollama запущена локально, но путь не преобразован корректно: {ollama_path}")
                # Пытаемся использовать исходный путь
                logger.info(f"Пробуем использовать исходный путь: {image_path}")
                ollama_path = image_path
            
            if is_local_server:
                logger.info(f"Используем локальный путь к файлу для Ollama: {ollama_path if 'ollama_path' in locals() else image_path}")
            else:
                logger.info("Ollama запущена удаленно, используем base64 кодирование изображения")
            
            # Пробуем разные варианты форматов запроса, пока один не сработает
            last_error = None
            response_text_raw = None
            
            async with aiohttp.ClientSession() as session:
                for format_name, current_request_data in request_variants:
                    try:
                        # Логируем текущий формат запроса
                        logger.info(f"Пробуем формат запроса: {format_name}")
                        
                        # Логируем полный запрос для отладки (без изображения)
                        debug_request = current_request_data.copy()
                        if "images" in debug_request:
                            if isinstance(debug_request["images"], list):
                                debug_request["images"] = ["<base64_image_data>"]
                            else:
                                debug_request["images"] = "<base64_image_data>"
                        if "image" in debug_request:
                            debug_request["image"] = "<base64_image_data>"
                        logger.debug(f"Отправляем запрос к API: {json.dumps(debug_request, ensure_ascii=False)}")
                        
                        # Увеличиваем таймаут до 120 секунд для обработки больших изображений
                        async with session.post(self.generate_url, json=current_request_data, timeout=120) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                logger.warning(f"Ошибка при запросе к API с форматом {format_name}: {response.status}, {error_text}")
                                
                                # Проверяем, содержит ли ошибка информацию о том, что модель не поддерживает изображения
                                if "does not support images" in error_text.lower() or "no image support" in error_text.lower():
                                    logger.error(f"Модель {used_model} не поддерживает изображения")
                                    return f"Модель {used_model} не поддерживает обработку изображений. Пожалуйста, используйте модель с поддержкой мультимодальности, например, llava, bakllava или gemma3."
                                
                                last_error = f"Ошибка при запросе к API с изображением: {response.status}. Подробности: {error_text}"
                                continue  # Пробуем следующий формат
                            
                            # Получаем ответ в зависимости от режима (stream или нет)
                            try:
                                if use_stream:
                                    # Получаем ответ построчно (stream)
                                    response_text_raw = ""
                                    async for line in response.content:
                                        if line:
                                            decoded_line = line.decode('utf-8')
                                            response_text_raw += decoded_line
                                            logger.debug(f"Получена строка ответа: {decoded_line[:100]}...")
                                else:
                                    # Получаем ответ целиком (не stream)
                                    response_text_raw = await response.text()
                                    logger.debug(f"Получен ответ целиком, длина: {len(response_text_raw)}")
                                
                                # Если получили какой-то ответ, выходим из цикла
                                if response_text_raw:
                                    logger.info(f"Успешно получен ответ с форматом {format_name}")
                                    # Сохраняем текущий формат запроса для использования в будущем
                                    request_data = current_request_data
                                    break
                                else:
                                    logger.warning(f"Получен пустой ответ с форматом {format_name}")
                                    last_error = "Получен пустой ответ от сервера"
                            except Exception as e:
                                logger.warning(f"Ошибка при чтении ответа с форматом {format_name}: {e}")
                                last_error = f"Ошибка при чтении ответа: {e}"
                                continue  # Пробуем следующий формат
                    except aiohttp.ClientError as e:
                        logger.warning(f"Ошибка соединения с API при использовании формата {format_name}: {e}")
                        last_error = f"Ошибка соединения с сервером: {e}"
                        continue  # Пробуем следующий формат
                
                # Если ни один формат не сработал, возвращаем последнюю ошибку
                if not response_text_raw:
                    logger.error("Все форматы запросов завершились с ошибкой")
                    return f"Не удалось получить ответ от модели. {last_error}"
                
                # Логируем полученный ответ
                logger.info(f"Получен ответ от API, длина: {len(response_text_raw)} символов")
                if response_text_raw:
                    logger.debug(f"Первые 200 символов ответа: {response_text_raw[:200]}...")
                    
                    # Проверяем, содержит ли ответ упоминание о белом квадрате или пустом изображении
                    white_square_phrases = [
                        "белый квадрат", "белое изображение", "пустое изображение", 
                        "не вижу изображение", "не могу увидеть изображение",
                        "white square", "white image", "empty image", 
                        "can't see the image", "cannot see the image"
                    ]
                    
                    if any(phrase in response_text_raw.lower() for phrase in white_square_phrases):
                        logger.warning("Модель видит белый квадрат или пустое изображение")
                        # Проверяем, определена ли переменная use_local_path и ollama_path
                        if 'use_local_path' in locals() and use_local_path and 'ollama_path' in locals():
                            logger.error(f"Проблема с локальным путем к файлу: {ollama_path}")
                            logger.info("Возможно, Ollama не имеет доступа к указанному пути или путь неверный")
                            # Добавляем информацию об ошибке в ответ
                            response_text_raw += "\n\n[Примечание: Возникла проблема с доступом к изображению. Возможно, Ollama не имеет доступа к указанному пути или путь неверный.]"
                        else:
                            logger.error("Проблема с обработкой изображения")
                            response_text_raw += "\n\n[Примечание: Возникла проблема с обработкой изображения.]"
                else:
                    logger.error("Получен пустой ответ от API")
                    return "Получен пустой ответ от сервера. Возможно, модель не поддерживает обработку изображений или произошла ошибка на сервере."
                
                try:
                    # Обрабатываем ответ в формате JSON в зависимости от режима
                    result = None
                    
                    if use_stream:
                        # Для потокового режима разбираем строки
                        lines = [line for line in response_text_raw.strip().split('\n') if line.strip()]
                        logger.info(f"Получено {len(lines)} строк JSON")
                        
                        if not lines:
                            logger.error("Получен пустой ответ от модели при запросе с изображением")
                            return "Извините, не удалось обработать изображение. Сервер вернул пустой ответ."
                        
                        # Если есть строки, берем последнюю (финальный результат)
                        last_line = lines[-1]
                        logger.debug(f"Последняя строка JSON: {last_line}")
                        
                        try:
                            result = json.loads(last_line)
                        except json.JSONDecodeError:
                            logger.error(f"Ошибка декодирования JSON в последней строке: {last_line}")
                            return "Ошибка при обработке ответа сервера. Получен некорректный JSON."
                    else:
                        # Для не-потокового режима парсим весь ответ как один JSON
                        try:
                            result = json.loads(response_text_raw)
                            logger.debug(f"Получен JSON ответ: {json.dumps(result, ensure_ascii=False)[:200]}...")
                        except json.JSONDecodeError:
                            logger.error(f"Ошибка декодирования JSON в ответе: {response_text_raw[:200]}...")
                            return "Ошибка при обработке ответа сервера. Получен некорректный JSON."
                    
                    # Логируем результат
                    logger.debug(f"Финальный JSON ответ с изображением: {json.dumps(result, ensure_ascii=False)[:200]}...")
                    
                    response_text = result.get("response", "")
                    if not response_text:
                        logger.error("Получен пустой текст ответа в JSON")
                        return "Модель вернула пустой ответ. Возможно, она не смогла обработать изображение."
                    
                    # Обрабатываем ответ в зависимости от модели
                    if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
                        model_name = "DeepSeek" if "deepseek" in used_model.lower() else "Gemma"
                        # Удаляем специальные теги из ответа
                        logger.info(f"Обрабатываем ответ от модели {model_name} с изображением: {used_model}")
                        
                        # Сохраняем оригинальную длину для логирования
                        original_length = len(response_text)
                        
                        # Удаляем теги <start_of_turn> и <end_of_turn> из ответа
                        import re
                        
                        # Более агрессивная очистка от тегов модели
                        patterns = [
                            r'<start_of_turn>model\n',
                            r'<start_of_turn>model',
                            r'<end_of_turn>',
                            r'<start_of_turn>user.*?<end_of_turn>\n?',
                            r'<start_of_turn>user.*?$',  # Если тег открыт, но не закрыт
                            r'<start_of_turn>.*?<end_of_turn>\n?',  # Любые другие теги
                        ]
                        
                        for pattern in patterns:
                            response_text = re.sub(pattern, '', response_text, flags=re.DOTALL)
                        
                        # Удаляем лишние пробелы и переносы строк
                        response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
                        response_text = response_text.strip()
                        
                        logger.info(f"Ответ от {model_name} с изображением после обработки тегов: было {original_length}, стало {len(response_text)} символов")
                    
                    # Фильтруем теги <think> из ответа
                    filtered_response = self.filter_think_tags(response_text)
                    
                    # Вычисляем разницу в длине до и после фильтрации
                    diff = len(response_text) - len(filtered_response)
                    if diff > 0:
                        logger.info(f"Получен ответ от модели с изображением длиной {len(response_text)} символов, удалено {diff} символов из тегов <think>, итоговая длина {len(filtered_response)} символов")
                    else:
                        logger.info(f"Получен ответ от модели с изображением длиной {len(response_text)} символов (теги <think> не найдены)")
                    
                    return filtered_response
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Ошибка декодирования JSON при запросе с изображением: {e}")
                    logger.debug(f"Полученный ответ: {response_text_raw[:200]}...")
                    
                    # Проверяем, есть ли в ответе текст, который можно вернуть
                    if response_text_raw and len(response_text_raw) > 10:
                        # Пытаемся извлечь текст из неправильного JSON
                        import re
                        # Ищем что-то похожее на текстовый ответ
                        text_match = re.search(r'"response"\s*:\s*"([^"]+)"', response_text_raw)
                        if text_match:
                            extracted_text = text_match.group(1)
                            logger.info(f"Извлечен текст из неправильного JSON: {extracted_text[:100]}...")
                            return extracted_text
                        else:
                            # Если не удалось извлечь текст, возвращаем сырой ответ
                            logger.info("Не удалось извлечь текст из ответа, возвращаем сырой текст")
                            return "Получен некорректный ответ от сервера. Пожалуйста, попробуйте еще раз."
                    else:
                        return "Получен некорректный или пустой ответ от сервера. Возможно, модель не поддерживает обработку изображений."
        except Exception as e:
            logger.error(f"Ошибка при отправке запроса с изображением к LLM: {e}")
            return f"Произошла ошибка при обработке запроса с изображением: {str(e)}"