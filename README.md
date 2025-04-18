# VasilisaBot - Wise Telegram Bot with Local LLMs

A Telegram bot that can answer questions and create summaries using local LLM models via Ollama or LM Studio.

**Language versions:**
- [English](README.en.md)
- [Russian](README.ru.md)

## Project Structure

The project has been reorganized into a modular structure:

```
summarybot/
├── bot.py                  # Main bot file
├── config/                 # Configuration settings
│   ├── __init__.py
│   └── settings.py         # Global settings and constants
├── handlers/               # Message and command handlers
│   ├── __init__.py
│   ├── command_handler.py  # Command handling logic
│   └── message_handler.py  # Message handling logic
├── models/                 # LLM client code
│   ├── __init__.py
│   ├── image_handler.py    # Image processing for LLM
│   └── llm_client.py       # Core LLM client functionality
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── file_utils.py       # File operations
│   ├── image_utils.py      # Image processing utilities
│   └── logging_utils.py    # Logging setup
├── data/                   # Data storage (created at runtime)
│   └── parquet/            # Parquet data storage
└── requirements.txt        # Project dependencies
```

## Key Features

1. **Modular Structure**: Code is organized into logical modules, making it easier to maintain and extend.

2. **Multiple LLM Providers**: Support for both Ollama and LM Studio as LLM providers.

3. **Platform Independence**: Works on Windows, macOS, and Linux without platform-specific code.

4. **Image Handling**: Images are stored in a temporary directory and processed using base64 encoding, eliminating the need for local file paths.

5. **Configuration**: Settings are centralized in the `config` module.

6. **Utilities**: Common functionality is extracted into utility modules.

7. **Multilingual Support**: Support for English and Russian languages, configurable via the `.env` file.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/summarybot.git
   cd summarybot
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   
   # Choose your LLM provider (ollama or lmstudio)
   LLM_PROVIDER=ollama
   
   # For Ollama:
   LLM_API_URL=http://localhost:11434/api
   
   # For LM Studio:
   # LLM_API_URL=http://localhost:1234/v1
   
   LLM_MODEL=llama2
   BOT_LANGUAGE=EN  # EN for English, RU for Russian
   LOG_LEVEL=INFO
   ```

4. Check your LLM setup:
   ```
   python check_llm.py
   ```

5. Run the bot:
   ```
   python bot.py
   ```

## Dependencies

- Python 3.8+
- python-telegram-bot
- aiohttp
- python-dotenv
- Pillow (for image processing)
- pandas and pyarrow (optional, for Parquet support)

## License

[MIT License](LICENSE)