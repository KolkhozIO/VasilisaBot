#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import tempfile
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime

from telegram import Update, Message
from telegram.ext import ContextTypes

from config.settings import DEFAULT_MODEL, BOT_LANGUAGE
from utils.logging_utils import get_logger
from utils.language_utils import get_text, get_default_prompts
from models.llm_client import LocalLLMClient
from models.image_handler import ImageHandler

logger = get_logger(__name__)

class MessageHandler:
    """Handler for processing messages."""
    
    def __init__(
        self,
        llm_client: LocalLLMClient,
        image_handler: ImageHandler,
        user_messages: Dict[int, List[Dict[str, Any]]],
        user_models: Dict[int, str],
        chat_models: Dict[int, str],
        factual_mode_users: Set[int],
        chat_temperatures: Dict[int, float],
        chat_factual_temperatures: Dict[int, float],
        chat_prompts: Dict[str, Dict[int, str]],
        group_prompts: Dict[str, Dict[int, str]],
        bot_config: Dict[str, Any]
    ):
        """
        Initialize the message handler.
        
        Args:
            llm_client: LLM client instance
            image_handler: Image handler instance
            user_messages: Dictionary of user messages
            user_models: Dictionary of user models
            chat_models: Dictionary of chat models
            factual_mode_users: Set of users in factual mode
            chat_temperatures: Dictionary of chat temperatures
            chat_factual_temperatures: Dictionary of chat factual temperatures
            chat_prompts: Dictionary of chat prompts
            group_prompts: Dictionary of group prompts
            bot_config: Bot configuration
        """
        self.llm_client = llm_client
        self.image_handler = image_handler
        self.user_messages = user_messages
        self.user_models = user_models
        self.chat_models = chat_models
        self.factual_mode_users = factual_mode_users
        self.chat_temperatures = chat_temperatures
        self.chat_factual_temperatures = chat_factual_temperatures
        self.chat_prompts = chat_prompts
        self.group_prompts = group_prompts
        self.bot_config = bot_config
    
    def clean_input_text(self, text: str) -> str:
        """
        Clean input text to remove potentially harmful characters.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Replace potentially harmful characters
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        
        return text
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle incoming messages.
        
        Args:
            update: Update object
            context: Context object
        """
        # Check if this is a regular message, not an edit
        if not update.message:
            return
            
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        message_text = update.message.text
        
        # Set default model if user/chat hasn't chosen one yet
        if user_id not in self.user_models:
            self.user_models[user_id] = DEFAULT_MODEL
        
        if chat_id < 0 and chat_id not in self.chat_models:  # Group chat
            self.chat_models[chat_id] = DEFAULT_MODEL
        
        # Save message for possible summarization
        # For group chats use chat_id, for personal chats use user_id
        message_key = chat_id if chat_id < 0 else user_id
        
        if message_key not in self.user_messages:
            self.user_messages[message_key] = []
        
        # Clean user name and message text before saving
        clean_name = self.clean_input_text(update.effective_user.first_name)
        clean_text = self.clean_input_text(message_text)
        
        # Determine if message is a reply to another message
        reply_to_message_id = None
        reply_to_index = None
        
        if update.message.reply_to_message:
            reply_to_message_id = update.message.reply_to_message.message_id
            
            # Find index of message being replied to
            if message_key in self.user_messages:
                for i, msg in enumerate(self.user_messages[message_key]):
                    if msg.get("message_id") == reply_to_message_id:
                        reply_to_index = i
                        break
        
        # Save message in extended format with additional metadata
        message_data = {
            "sender": clean_name,
            "text": clean_text,
            "reply_to_index": reply_to_index,  # Can be None or integer
            "message_id": update.message.message_id,
            "date": update.message.date.isoformat(),
            "user_id": user_id,
            "chat_id": chat_id
        }
        
        # Check that reply_to_index is not NaN
        if isinstance(reply_to_index, float) and math.isnan(reply_to_index):
            logger.warning(f"Detected NaN value in reply_to_index, setting to None")
            message_data["reply_to_index"] = None
        
        # Add user information if available
        if update.effective_user.username:
            message_data["username"] = update.effective_user.username
        if update.effective_user.last_name:
            message_data["last_name"] = update.effective_user.last_name
            
        # Add information about message being replied to
        if reply_to_message_id:
            message_data["reply_to_message_id"] = reply_to_message_id
            
        self.user_messages[message_key].append(message_data)
        
        # Determine if we should respond to the message
        should_respond = False
        bot_username = context.bot.username
        
        # In personal chats, respond to all messages
        if chat_id > 0:  # Personal chat
            should_respond = True
        else:  # Group chat
            # Respond if bot is mentioned or message starts with !
            if message_text:
                if f"@{bot_username}" in message_text:
                    should_respond = True
                    # Remove bot mention from message
                    message_text = message_text.replace(f"@{bot_username}", "").strip()
                elif message_text.startswith("!"):
                    should_respond = True
                    # Remove ! from message
                    message_text = message_text[1:].strip()
            
            # Respond if message is a reply to bot's message
            if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                should_respond = True
        
        # If we should respond, process the message
        if should_respond:
            # Set typing action
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Determine which model to use
            if chat_id < 0:  # Group chat
                used_model = self.chat_models.get(chat_id, DEFAULT_MODEL)
            else:  # Personal chat
                used_model = self.user_models.get(user_id, DEFAULT_MODEL)
            
            # Determine temperature based on factual mode and chat settings
            is_factual = user_id in self.factual_mode_users
            
            if is_factual:
                # Use chat-specific factual temperature if available, otherwise use default
                temperature = self.chat_factual_temperatures.get(
                    chat_id, self.bot_config["factual_temperature"]
                )
            else:
                # Use chat-specific temperature if available, otherwise use default
                temperature = self.chat_temperatures.get(
                    chat_id, self.bot_config["temperature"]
                )
            
            # Get appropriate prompt
            if chat_id < 0:  # Group chat
                prompt_template = self.group_prompts["answer"].get(
                    chat_id, self.chat_prompts["answer"].get(user_id, None)
                )
            else:  # Personal chat
                prompt_template = self.chat_prompts["answer"].get(user_id, None)
            
            # If no custom prompt, use default
            if not prompt_template:
                default_prompts = get_default_prompts()
                prompt_template = default_prompts["answer"]
            
            # Format prompt with question
            prompt = prompt_template.format(question=message_text)
            
            # Check if message contains an image
            if update.message.photo:
                # Get the largest photo (last in the list)
                photo = update.message.photo[-1]
                
                # Download the photo
                photo_file = await context.bot.get_file(photo.file_id)
                
                # Create a temporary file to save the photo
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    await photo_file.download_to_drive(custom_path=temp_file.name)
                    temp_path = temp_file.name
                
                try:
                    # Process image with caption as prompt
                    caption = update.message.caption or "Describe this image"
                    
                    # Format prompt with caption
                    image_prompt = prompt_template.format(question=f"{caption}\n\n[Image attached]")
                    
                    # Get response from model
                    response = await self.image_handler.process_image_request(
                        prompt=image_prompt,
                        image_path=temp_path,
                        model=used_model,
                        temperature=temperature,
                        message_id=str(update.message.message_id)
                    )
                    
                    # Send response
                    await update.message.reply_text(response)
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file: {e}")
            else:
                # Text-only message
                # Get response from model
                response = await self.llm_client.answer_question(
                    prompt=prompt,
                    model=used_model,
                    temperature=temperature
                )
                
                # Send response
                await update.message.reply_text(response)
    
    async def handle_summarize(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle summarize command.
        
        Args:
            update: Update object
            context: Context object
        """
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Set typing action
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Determine which messages to summarize
        message_key = chat_id if chat_id < 0 else user_id
        
        if message_key not in self.user_messages or not self.user_messages[message_key]:
            await update.message.reply_text(get_text("no_messages_to_summarize"))
            return
        
        # Limit number of messages to summarize
        max_messages = min(len(self.user_messages[message_key]), self.bot_config["summary_messages"])
        messages_to_summarize = self.user_messages[message_key][-max_messages:]
        
        # Format messages for summarization
        formatted_messages = []
        
        for msg in messages_to_summarize:
            sender = msg.get("sender", "Unknown")
            text = msg.get("text", "")
            
            # Skip empty messages
            if not text:
                continue
            
            # Format message
            formatted_message = f"{sender}: {text}"
            formatted_messages.append(formatted_message)
        
        # Join messages
        text_to_summarize = "\n".join(formatted_messages)
        
        # Determine which model to use
        if chat_id < 0:  # Group chat
            used_model = self.chat_models.get(chat_id, DEFAULT_MODEL)
        else:  # Personal chat
            used_model = self.user_models.get(user_id, DEFAULT_MODEL)
        
        # Determine temperature for summarization
        temperature = self.chat_summary_temperatures.get(
            chat_id, self.bot_config["summary_temperature"]
        )
        
        # Get appropriate prompt
        if chat_id < 0:  # Group chat
            prompt_template = self.group_prompts["summary"].get(
                chat_id, self.chat_prompts["summary"].get(user_id, None)
            )
        else:  # Personal chat
            prompt_template = self.chat_prompts["summary"].get(user_id, None)
        
        # If no custom prompt, use default
        if not prompt_template:
            default_prompts = get_default_prompts()
            prompt_template = default_prompts["summary"]
        
        # Format prompt with text to summarize
        prompt = prompt_template.format(text=text_to_summarize)
        
        # Get summary from model
        summary = await self.llm_client.answer_question(
            prompt=prompt,
            model=used_model,
            temperature=temperature
        )
        
        # Send summary
        await update.message.reply_text(summary)