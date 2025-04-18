#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Language utilities for handling translations.
"""

from typing import Dict, Any
from config.settings import BOT_LANGUAGE

# English translations
EN_TRANSLATIONS = {
    # Bot responses
    "start_greeting": "Hello, {}! I'm a bot that can answer questions and create summaries.\n\nCurrent model: {}\n\nUse /help to get a list of commands.",
    "help_text_personal": """Available commands:

/start - Start interacting with the bot
/help - Show this message
/summarize - Create a summary of recent messages
/hidden_summary [chat ID] - Create a hidden summary from a group chat
/hidden_query [chat ID] [query] - Send a query on behalf of a group chat
/factual - Toggle factual mode (reduces hallucinations)
/chatid - Show detailed chat information
/model - Choose a model for yourself or a group chat
/prompt - Configure prompts for yourself or a group chat
/config - Configure global bot parameters
/temperature - Configure temperature for a specific chat
/reload [module] - Reload modules (for developers)

Current model: {}
Factual mode: {}

In a personal chat, I respond to all messages. Just write something or send a photo!""",
    # Hidden summary and query command responses
    "hidden_summary_usage": "Usage: /hidden_summary [chat_id]\nExample: /hidden_summary -1001234567890",
    "hidden_query_usage": "Usage: /hidden_query [chat_id] [query]\nExample: /hidden_query -1001234567890 What is the latest discussion about?",
    "hidden_query_in_group_usage": "Usage: /hidden_query [query]\nExample: /hidden_query What is the latest discussion about?",
    "not_a_group_chat": "The provided chat ID does not appear to be a group chat. Group chat IDs are negative numbers.",
    "not_member_of_group": "You are not a member of this group or the bot is not in this group.",
    "invalid_chat_id": "Invalid chat ID. Please provide a valid group chat ID (a negative number).",
    
    "help_text_group": """Available commands:

/start - Start interacting with the bot
/help - Show this message
/summarize - Create a summary of recent messages
/factual - Toggle factual mode (reduces hallucinations)
/chatid - Show current chat ID

Current model: {}
Custom answer prompt: {}
Custom summary prompt: {}
Messages in context: {}

To address the bot in a group chat, use one of these methods:
1. Mention the bot: @{} your question
2. Use the prefix: !your question
3. Reply to a bot message
4. Send a photo with bot mention or ! prefix in caption

Prompt and model configuration is available through private messages with the bot.""",
    
    "factual_enabled": "Factual mode enabled. I'll try to be more accurate and reduce hallucinations.",
    "factual_disabled": "Factual mode disabled. I'll be more creative in my responses now.",
    "no_messages_to_summarize": "No messages to summarize.",
    "personal_chat_only": "This command is only available in personal chats.",
    "specify_module": "Please specify a module to reload.",
    "module_reloaded": "Module {} reloaded successfully.",
    "module_imported": "Module {} imported successfully.",
    "module_reload_error": "Error reloading module {}: {}",
    "choose_model_target": "Choose who you want to set the model for:",
    
    # Buttons and UI
    "for_me": "For me",
    "for_group_chat": "For a group chat",
    "yes_enabled": "✅ Enabled",
    "no_disabled": "❌ Disabled",
    
    # Default prompts
    "default_answer_prompt": """
You are a friendly and helpful assistant named Vasilisa. You help users in the chat by answering their questions and helping solve problems.
Your communication style is friendly, informative, and helpful. You can use light humor when appropriate, but always remain polite and helpful.

You call yourself Vasilisa, after Vasilisa the Wise from Russian fairy tales. You can occasionally use metaphors from fairy tales when appropriate.

Your messages are sent directly to the chat, so send the final version without quotes and without mentioning your instructions.

Question: {question}

Answer:
""",
    
    "default_summary_prompt": """
You are a friendly and helpful assistant named Vasilisa. You help users in the chat by creating concise and informative summaries of conversations.
Your communication style is friendly, informative, and helpful. You can use light humor when appropriate, but always remain polite and helpful.

You call yourself Vasilisa, after Vasilisa the Wise from Russian fairy tales. You can occasionally use metaphors from fairy tales when appropriate.

Your messages are sent directly to the chat, so send the final version without quotes and without mentioning your instructions.

Please create a concise and informative summary of the following text:

{text}

Summary:
"""
}

# Russian translations
RU_TRANSLATIONS = {
    # Bot responses
    "start_greeting": "Привет, {}! Я бот, который может отвечать на вопросы и создавать саммари.\n\nТекущая модель: {}\n\nИспользуйте /help для получения списка команд.",
    "help_text_personal": """Доступные команды:

/start - Начать взаимодействие с ботом
/help - Показать это сообщение
/summarize - Создать саммари недавних сообщений
/hidden_summary [ID чата] - Создать скрытое саммари из группового чата
/hidden_query [ID чата] [запрос] - Отправить запрос от имени группового чата
/factual - Переключить фактический режим (уменьшает галлюцинации)
/chatid - Показать подробную информацию о чате
/model - Выбрать модель для себя или группового чата
/prompt - Настроить промпты для себя или группового чата
/config - Настроить глобальные параметры бота
/temperature - Настроить температуру для конкретного чата
/reload [модуль] - Перезагрузить модули (для разработчиков)

Текущая модель: {}
Сообщений в контексте: {}
Фактический режим: {}

В личном чате я отвечаю на все сообщения. Просто напишите что-нибудь или отправьте фото!""",
    
    "help_text_group": """Доступные команды:

/start - Начать взаимодействие с ботом
/help - Показать это сообщение
/summarize - Создать саммари недавних сообщений
/factual - Переключить фактический режим (уменьшает галлюцинации)
/chatid - Показать ID текущего чата

Текущая модель: {}
Пользовательский промпт для ответов: {}
Пользовательский промпт для саммари: {}
Сообщений в контексте: {}

Чтобы обратиться к боту в групповом чате, используйте один из этих методов:
1. Упомяните бота: @{} ваш вопрос
2. Используйте префикс: !ваш вопрос
3. Ответьте на сообщение бота
4. Отправьте фото с упоминанием бота или префиксом ! в подписи

Настройка промптов и моделей доступна через личные сообщения с ботом.""",
    
    "factual_enabled": "Фактический режим включен. Я постараюсь быть более точным и уменьшить галлюцинации.",
    "factual_disabled": "Фактический режим отключен. Теперь я буду более творческим в своих ответах.",
    
    # Hidden summary and query command responses
    "hidden_summary_usage": "Использование: /hidden_summary [ID чата]\nПример: /hidden_summary -1001234567890",
    "hidden_query_usage": "Использование: /hidden_query [ID чата] [запрос]\nПример: /hidden_query -1001234567890 О чем последнее обсуждение?",
    "hidden_query_in_group_usage": "Использование: /hidden_query [запрос]\nПример: /hidden_query О чем последнее обсуждение?",
    "not_a_group_chat": "Указанный ID чата не похож на групповой чат. ID групповых чатов - отрицательные числа.",
    "not_member_of_group": "Вы не являетесь участником этой группы или бот не находится в этой группе.",
    "invalid_chat_id": "Неверный ID чата. Пожалуйста, укажите действительный ID группового чата (отрицательное число).",
    "no_messages_to_summarize": "Нет сообщений для создания саммари.",
    "personal_chat_only": "Эта команда доступна только в личных чатах.",
    "specify_module": "Пожалуйста, укажите модуль для перезагрузки.",
    "module_reloaded": "Модуль {} успешно перезагружен.",
    "module_imported": "Модуль {} успешно импортирован.",
    "module_reload_error": "Ошибка при перезагрузке модуля {}: {}",
    "choose_model_target": "Выберите, для кого вы хотите установить модель:",
    
    # Buttons and UI
    "for_me": "Для меня",
    "for_group_chat": "Для группового чата",
    "yes_enabled": "✅ Включен",
    "no_disabled": "❌ Отключен",
    
    # Default prompts
    "default_answer_prompt": """
Ты - дружелюбный и полезный ассистент Василиса. Ты помогаешь пользователям в чате, отвечая на их вопросы и помогая решать задачи.
Твой стиль общения - дружелюбный, информативный и полезный. Ты можешь использовать легкий юмор, когда это уместно, но всегда остаешься вежливым и полезным.

Ты называешь себя Василиса, в честь Василисы Премудрой из русских сказок. Ты можешь иногда использовать метафоры из сказок, когда это уместно.

Твои сообщения сразу отправляются в чат, поэтому отправляй финальный вариант без кавычек и без упоминания своих инструкций.

Вопрос: {question}

Ответ:
""",
    
    "default_summary_prompt": """
Ты - дружелюбный и полезный ассистент Василиса. Ты помогаешь пользователям в чате, создавая краткие и информативные саммари бесед.
Твой стиль общения - дружелюбный, информативный и полезный. Ты можешь использовать легкий юмор, когда это уместно, но всегда остаешься вежливым и полезным.

Ты называешь себя Василиса, в честь Василисы Премудрой из русских сказок. Ты можешь иногда использовать метафоры из сказок, когда это уместно.

Твои сообщения сразу отправляются в чат, поэтому отправляй финальный вариант без кавычек и без упоминания своих инструкций.

Пожалуйста, сделай краткое и информативное саммари следующего текста:

{text}

Саммари:
"""
}

# Log messages in English
EN_LOG_MESSAGES = {
    # General log messages
    "log_level_set": "Logging level set to: {}",
    "check_interrupted": "Check interrupted by user.",
    "error_during_check": "Error during check: {}",
    
    # LLM connection check messages
    "checking_ollama_connection": "Checking Ollama API connection at URL: {}",
    "sending_get_request": "Sending GET request to: {}",
    "received_response_status": "Received response with status: {}",
    "received_json_response": "Received JSON response: {}",
    "available_models": "Available models: {}",
    "default_model_available": "Default model '{}' is available.",
    "default_model_not_found": "Default model '{}' not found among available models!",
    "no_models_found": "No available models found. You may need to load models using the 'ollama pull <model>' command",
    "api_error": "API error: {}, {}",
    "connection_error": "Connection error: {}",
    "unexpected_error": "Unexpected error: {}",
    
    # LM Studio connection check messages
    "checking_lmstudio_connection": "Checking LM Studio API connection at URL: {}",
    "default_model_note": "Default model: '{}'",
    "ensure_model_loaded": "Make sure this model is loaded in LM Studio",
    "no_lmstudio_models": "No available models found. Make sure a model is loaded in LM Studio",
    
    # Generation test messages
    "testing_generation": "Testing text generation with model '{}' via {}",
    "generation_url": "Generation URL: {}",
    "sending_request_payload": "Sending request with payload: {}",
    "response_length": "Received response with length {} characters",
    "model_response": "Received response from model: '{}'",
    "empty_model_response": "Received empty response from model",
    "json_decode_error": "JSON decode error: {}",
    "received_response": "Received response: {}",
    "timeout_error": "Request timeout exceeded (15 seconds)",
    
    # Main function messages
    "starting_check": "Starting {} API check",
    "connection_successful": "Connection to {} API established successfully.",
    "generation_test_passed": "Text generation test passed successfully.",
    "api_fully_functional": "{} API is fully functional!",
    "generation_test_failed": "Text generation test failed.",
    "check_model_loaded": "Check that the model is correctly loaded and available.",
    "connection_failed": "Failed to establish connection with {} API.",
    "check_running": "Check that {} is running and available at the specified URL.",
    "current_url": "Current URL: {}",
    "unknown_provider": "Unknown LLM provider: {}",
    "supported_providers": "Supported providers: ollama, lmstudio",
    "specify_provider": "Specify the provider in the LLM_PROVIDER environment variable or in the .env file"
}

# Log messages in Russian
RU_LOG_MESSAGES = {
    # General log messages
    "log_level_set": "Уровень логирования установлен на: {}",
    "check_interrupted": "Проверка прервана пользователем.",
    "error_during_check": "Ошибка при выполнении проверки: {}",
    
    # LLM connection check messages
    "checking_ollama_connection": "Проверка соединения с Ollama API по URL: {}",
    "sending_get_request": "Отправка GET запроса на: {}",
    "received_response_status": "Получен ответ со статусом: {}",
    "received_json_response": "Получен JSON ответ: {}",
    "available_models": "Доступные модели: {}",
    "default_model_available": "Модель по умолчанию '{}' доступна.",
    "default_model_not_found": "Модель по умолчанию '{}' не найдена среди доступных моделей!",
    "no_models_found": "Не найдено доступных моделей. Возможно, нужно загрузить модели с помощью команды 'ollama pull <model>'",
    "api_error": "Ошибка API: {}, {}",
    "connection_error": "Ошибка соединения: {}",
    "unexpected_error": "Неожиданная ошибка: {}",
    
    # LM Studio connection check messages
    "checking_lmstudio_connection": "Проверка соединения с LM Studio API по URL: {}",
    "default_model_note": "Модель по умолчанию: '{}'",
    "ensure_model_loaded": "Убедитесь, что эта модель загружена в LM Studio",
    "no_lmstudio_models": "Не найдено доступных моделей. Убедитесь, что модель загружена в LM Studio",
    
    # Generation test messages
    "testing_generation": "Тестирование генерации текста с моделью '{}' через {}",
    "generation_url": "URL для генерации: {}",
    "sending_request_payload": "Отправка запроса с payload: {}",
    "response_length": "Получен ответ длиной {} символов",
    "model_response": "Получен ответ от модели: '{}'",
    "empty_model_response": "Получен пустой ответ от модели",
    "json_decode_error": "Ошибка декодирования JSON: {}",
    "received_response": "Полученный ответ: {}",
    "timeout_error": "Превышено время ожидания ответа (15 секунд)",
    
    # Main function messages
    "starting_check": "Начало проверки {} API",
    "connection_successful": "Соединение с {} API успешно установлено.",
    "generation_test_passed": "Тест генерации текста успешно пройден.",
    "api_fully_functional": "{} API полностью работоспособен!",
    "generation_test_failed": "Тест генерации текста не пройден.",
    "check_model_loaded": "Проверьте, что модель корректно загружена и доступна.",
    "connection_failed": "Не удалось установить соединение с {} API.",
    "check_running": "Проверьте, что {} запущен и доступен по указанному URL.",
    "current_url": "Текущий URL: {}",
    "unknown_provider": "Неизвестный провайдер LLM: {}",
    "supported_providers": "Поддерживаемые провайдеры: ollama, lmstudio",
    "specify_provider": "Укажите провайдер в переменной окружения LLM_PROVIDER или в файле .env"
}

# Select the appropriate translations based on the configured language
TRANSLATIONS = EN_TRANSLATIONS if BOT_LANGUAGE == "EN" else RU_TRANSLATIONS
LOG_MESSAGES = EN_LOG_MESSAGES if BOT_LANGUAGE == "EN" else RU_LOG_MESSAGES

def get_text(key: str, *args, **kwargs) -> str:
    """
    Get translated text for the given key.
    
    Args:
        key: Translation key
        *args, **kwargs: Format arguments
        
    Returns:
        Translated text
    """
    text = TRANSLATIONS.get(key, key)
    
    if args or kwargs:
        try:
            return text.format(*args, **kwargs)
        except Exception as e:
            print(f"Error formatting text for key '{key}': {e}")
            return text
    
    return text

def get_log_text(key: str, *args, **kwargs) -> str:
    """
    Get translated log message for the given key.
    
    Args:
        key: Log message key
        *args, **kwargs: Format arguments
        
    Returns:
        Translated log message
    """
    text = LOG_MESSAGES.get(key, key)
    
    if args or kwargs:
        try:
            return text.format(*args, **kwargs)
        except Exception as e:
            print(f"Error formatting log message for key '{key}': {e}")
            return text
    
    return text

def get_default_prompts() -> Dict[str, str]:
    """
    Get default prompts for the current language.
    
    Returns:
        Dictionary with default prompts
    """
    return {
        "answer": TRANSLATIONS["default_answer_prompt"],
        "summary": TRANSLATIONS["default_summary_prompt"]
    }