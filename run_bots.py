#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_bot(token, language, log_file=None):
    """
    Run a bot instance with the specified token and language.
    
    Args:
        token: Telegram bot token
        language: Bot language (EN or RU)
        log_file: Path to log file (optional)
    """
    env = os.environ.copy()
    env["TELEGRAM_BOT_TOKEN"] = token
    env["BOT_LANGUAGE"] = language
    
    # Prepare command
    cmd = [sys.executable, "bot.py"]
    
    # Prepare stdout/stderr redirection
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        stdout = open(log_file, "a")
        stderr = subprocess.STDOUT
    else:
        stdout = None
        stderr = None
    
    # Run the bot
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout,
        stderr=stderr
    )
    
    return process

def main():
    parser = argparse.ArgumentParser(description="Run multiple SummaryBot instances with different configurations")
    parser.add_argument("--en-token", help="Telegram bot token for English bot")
    parser.add_argument("--ru-token", help="Telegram bot token for Russian bot")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    
    args = parser.parse_args()
    
    processes = []
    
    # Start English bot if token provided
    if args.en_token:
        log_file = os.path.join(args.log_dir, "en_bot.log")
        print(f"Starting English bot, logs will be written to {log_file}")
        en_process = run_bot(args.en_token, "EN", log_file)
        processes.append(("EN", en_process))
    
    # Start Russian bot if token provided
    if args.ru_token:
        log_file = os.path.join(args.log_dir, "ru_bot.log")
        print(f"Starting Russian bot, logs will be written to {log_file}")
        ru_process = run_bot(args.ru_token, "RU", log_file)
        processes.append(("RU", ru_process))
    
    if not processes:
        print("Error: No bot tokens provided. Use --en-token and/or --ru-token")
        sys.exit(1)
    
    print(f"Started {len(processes)} bot instances")
    print("Press Ctrl+C to stop all bots")
    
    try:
        while True:
            # Check if any process has terminated
            for lang, process in processes:
                if process.poll() is not None:
                    print(f"{lang} bot terminated with exit code {process.returncode}")
            
            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopping all bots...")
        for lang, process in processes:
            if process.poll() is None:
                print(f"Terminating {lang} bot...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"Killing {lang} bot...")
                    process.kill()
        
        print("All bots stopped")

if __name__ == "__main__":
    main()