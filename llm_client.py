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