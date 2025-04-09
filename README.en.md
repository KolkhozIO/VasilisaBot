# VasilisaBot - Wise Telegram Bot with Ollama

VasilisaBot is an asynchronous Telegram bot that interacts with Ollama for answering questions and creating text summaries. Like Vasilisa the Wise from Russian fairy tales, the bot analyzes messages, finds their essence, and helps users. The bot supports both private and group chats, saves message history, and correctly handles reply hierarchies.

## Features

- Asynchronous message processing
- Question answering using Ollama
- Creating summaries from message history
- Selection of various Ollama models (llama2, mistral, gemma, deepseek, etc.)
- Custom prompt configuration for chats
- Works in both private and group chats
- Saving and restoring message history
- Proper handling of reply hierarchies
- Support for message editing
- Temperature configuration for different types of requests
- Automatic data saving
- Simple and intuitive interface

## Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) - local server for running LLM models
- Telegram Bot Token (obtained through [@BotFather](https://t.me/BotFather))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vasilisabot.git
cd vasilisabot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add the following variables:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
LLM_API_URL=http://localhost:11434/api
LLM_MODEL=llama2
LOG_LEVEL=INFO  # Optional: DEBUG, INFO, WARNING, ERROR
```

## Launch

1. Install and start Ollama:
   - Download Ollama from the [official website](https://ollama.ai/)
   - Launch Ollama
   - Load the desired models with the command `ollama pull llama2` (or other models)

2. Start the bot:
```bash
python bot.py
```

## Auto-start in WSL

To set up auto-start for the bot in WSL (Windows Subsystem for Linux), you can use the following methods:

> **Important!** Make sure you have only one instance of the bot running. If you see the error `Conflict: terminated by other getUpdates request`, it means you have multiple bot instances running simultaneously. Before setting up auto-start, check and stop all running instances:
> ```bash
> ps aux | grep python | grep bot.py
> kill <PID>
> ```

### 1. Using systemd (for WSL2 with systemd support)

1. Create a service file:

```bash
sudo nano /etc/systemd/system/summarybot.service
```

2. Add the following content (replace paths with your own):

```
[Unit]
Description=Telegram Summary Bot
After=network.target
StartLimitIntervalSec=0

[Service]
Type=simple
Restart=always
RestartSec=1
User=YOUR_USERNAME
WorkingDirectory=/path/to/summarybot
ExecStart=/usr/bin/python3 /path/to/summarybot/bot.py

[Install]
WantedBy=multi-user.target
```

3. Activate and start the service:

```bash
sudo systemctl enable summarybot.service
sudo systemctl start summarybot.service
```

4. Check the status:

```bash
sudo systemctl status summarybot.service
```

### 2. Using a WSL auto-start script

If you want to start the bot when Windows starts:

1. Create a `start_bot.sh` file in the project directory:

```bash
#!/bin/bash
cd /path/to/summarybot
python3 bot.py
```

2. Make it executable:

```bash
chmod +x start_bot.sh
```

3. Create a Windows batch file `start_bot.bat` in a convenient location on your computer:

```batch
@echo off
wsl -d YOUR_DISTRO_NAME -u YOUR_USERNAME /path/to/summarybot/start_bot.sh
```

4. Add this batch file to Windows startup:
   - Press Win+R, type `shell:startup` and press Enter
   - Copy or create a shortcut to your batch file in this folder

### 3. Using cron (if systemd is not available)

1. Open crontab for editing:

```bash
crontab -e
```

2. Add a line to start the bot at reboot:

```
@reboot cd /path/to/summarybot && python3 bot.py >> /path/to/summarybot/bot.log 2>&1
```

Note: For cron to work in WSL, you may need to start the cron service:

```bash
sudo service cron start
sudo service cron enable
```

## Usage

After launching the bot, you can interact with it through Telegram:

### General Commands
- `/start` - Begin interaction with the bot
- `/help` - Show a list of available commands
- `/summarize` - Create a summary of recent messages
- `/hidden_summary` - Create a summary visible only to you
- `/hidden_query` - Ask a question visible only to you
- `/model` - Select an Ollama model to use
- `/chatid` - Show the ID of the current chat
- `/factual` - Enable/disable factual response mode (with reduced temperature)
- `/config` - Configure bot parameters
- `/reload` - Reload bot modules
- Send any text message to get a response from the selected model

### Commands for Group Chats
- `/prompt` - Configure custom prompts for the current chat

### Prompt Configuration
You can configure custom prompts for:
1. Question answering - use `{question}` to insert the user's question
2. Summary creation - use `{text}` to insert the text for summarization

Example prompts:
```
# Prompt for answers
You are a friendly chat assistant. Question: {question}

Answer:

# Prompt for summaries
Create a concise and informative summary of the following conversation:

{text}

Summary:
```

### Replying to Messages
The bot correctly handles reply hierarchies. If you reply to a specific message, the bot takes this context into account when formulating its response.

### Configuration Settings
Use the `/config` command to configure the following parameters:
- Generation temperature (for regular answers, factual answers, and summaries)
- Auto-save interval for data

## Ollama Configuration

The bot requires Ollama to function. You can:

1. Load various models using the `ollama pull` command:
   ```bash
   ollama pull llama2
   ollama pull mistral
   ollama pull gemma:2b
   ollama pull deepseek-coder
   ```

2. Create your own custom models using a Modelfile:
   ```bash
   ollama create mycustom -f ./Modelfile
   ```

3. Use models with various parameters through the API

## Project Structure

- `bot.py` - Main bot file
- `llm_client.py` - Client for interacting with Ollama
- `requirements.txt` - Project dependencies
- `.env` - Environment variables file
- `.env.example` - Example environment variables file
- `data/` - Directory for storing data (messages, settings, prompts)

## Data Storage

The bot automatically saves the following data:
- Message history (in JSON and Parquet formats)
- Custom prompt settings
- Selected models for users and chats
- Temperature generation settings
- General bot configuration

All data is saved in the `data/` directory.

## License

MIT