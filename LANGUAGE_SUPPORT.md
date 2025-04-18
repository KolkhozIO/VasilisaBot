# Multilingual Support Implementation

This document describes how multilingual support was implemented in the VasilisaBot project.

## Overview

The bot now supports both English (EN) and Russian (RU) languages. The language can be configured through the `.env` file using the `BOT_LANGUAGE` variable. All user-facing messages are translated, while logs remain in English for better debugging.

## Implementation Details

1. **Configuration**
   - Added `BOT_LANGUAGE` setting in `config/settings.py`
   - Default language is English (EN)
   - Supported languages are defined in a list: `SUPPORTED_LANGUAGES = ["EN", "RU"]`

2. **Translation System**
   - Created `utils/language_utils.py` module for handling translations
   - Implemented dictionaries for each supported language
   - Added a `get_text(key, *args, **kwargs)` function to retrieve translated text
   - Added a `get_default_prompts()` function to get language-specific default prompts

3. **Default Prompts**
   - Moved default prompts from `config/settings.py` to `utils/language_utils.py`
   - Created language-specific versions of prompts for both answer and summary generation

4. **Command Handlers**
   - Updated all user-facing messages in `handlers/command_handler.py` to use the translation system
   - Replaced hardcoded strings with calls to `get_text()`

5. **Message Handlers**
   - Updated message handling in `handlers/message_handler.py` to use translated messages
   - Used language-specific default prompts for answer and summary generation

6. **Documentation**
   - Created README files in both languages: `README.en.md` and `README.ru.md`
   - Updated the main `README.md` to include links to both language versions
   - Updated `.env.example` to include the `BOT_LANGUAGE` setting

## How to Add a New Language

To add support for a new language:

1. Add the language code to `SUPPORTED_LANGUAGES` in `config/settings.py`
2. Create a new translation dictionary in `utils/language_utils.py` (e.g., `DE_TRANSLATIONS` for German)
3. Add all required translation keys and values to the new dictionary
4. Update the language selection logic in `utils/language_utils.py`
5. Create a README file for the new language (e.g., `README.de.md`)
6. Update the main `README.md` to include a link to the new language version

## Translation Keys

The following translation keys are currently used:

- `start_greeting`: Greeting message for the /start command
- `help_text_personal`: Help text for personal chats
- `help_text_group`: Help text for group chats
- `factual_enabled`: Message when factual mode is enabled
- `factual_disabled`: Message when factual mode is disabled
- `no_messages_to_summarize`: Message when there are no messages to summarize
- `personal_chat_only`: Message when a command is only available in personal chats
- `specify_module`: Message when no module is specified for reload
- `module_reloaded`: Message when a module is successfully reloaded
- `module_imported`: Message when a module is successfully imported
- `module_reload_error`: Message when there's an error reloading a module
- `choose_model_target`: Message when choosing a model target
- `for_me`: Button text for "For me"
- `for_group_chat`: Button text for "For a group chat"
- `yes_enabled`: Text for "Yes, enabled"
- `no_disabled`: Text for "No, disabled"
- `default_answer_prompt`: Default prompt for answering questions
- `default_summary_prompt`: Default prompt for creating summaries

## Notes

- All logs remain in English for better debugging and support
- The bot's language setting affects only user-facing messages
- The language can be changed by updating the `.env` file and restarting the bot