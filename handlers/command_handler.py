#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Any, Optional, Set, Union
import importlib
import sys

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

from config.settings import (
    DEFAULT_MODEL, CHOOSING_MODEL, SETTING_PROMPT, CHOOSING_PROMPT_TYPE,
    SETTING_CONFIG, CHOOSING_CONFIG, CHOOSING_CHAT, ENTERING_CHAT_ID,
    CHOOSING_MODEL_TARGET, CHOOSING_TEMP_TYPE, CHOOSING_CHAT_FOR_TEMP,
    ENTERING_CHAT_ID_FOR_TEMP, SETTING_TEMPERATURE, BOT_LANGUAGE
)
from utils.logging_utils import get_logger
from utils.language_utils import get_text, get_default_prompts
from models.llm_client import LocalLLMClient, check_ollama_availability

logger = get_logger(__name__)

class CommandHandler:
    """Handler for processing commands."""
    
    def __init__(
        self,
        llm_client: LocalLLMClient,
        user_messages: Dict[int, List[Dict[str, Any]]],
        user_models: Dict[int, str],
        chat_models: Dict[int, str],
        factual_mode_users: Set[int],
        chat_temperatures: Dict[int, float],
        chat_factual_temperatures: Dict[int, float],
        chat_summary_temperatures: Dict[int, float],
        chat_prompts: Dict[str, Dict[int, str]],
        group_prompts: Dict[str, Dict[int, str]],
        bot_config: Dict[str, Any]
    ):
        """
        Initialize the command handler.
        
        Args:
            llm_client: LLM client instance
            user_messages: Dictionary of user messages
            user_models: Dictionary of user models
            chat_models: Dictionary of chat models
            factual_mode_users: Set of users in factual mode
            chat_temperatures: Dictionary of chat temperatures
            chat_factual_temperatures: Dictionary of chat factual temperatures
            chat_summary_temperatures: Dictionary of chat summary temperatures
            chat_prompts: Dictionary of chat prompts
            group_prompts: Dictionary of group prompts
            bot_config: Bot configuration
        """
        self.llm_client = llm_client
        self.user_messages = user_messages
        self.user_models = user_models
        self.chat_models = chat_models
        self.factual_mode_users = factual_mode_users
        self.chat_temperatures = chat_temperatures
        self.chat_factual_temperatures = chat_factual_temperatures
        self.chat_summary_temperatures = chat_summary_temperatures
        self.chat_prompts = chat_prompts
        self.group_prompts = group_prompts
        self.bot_config = bot_config
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = update.effective_user
        user_id = user.id
        chat_id = update.effective_chat.id
        
        # Set default model for new user/chat
        if user_id not in self.user_models:
            self.user_models[user_id] = DEFAULT_MODEL
        
        if chat_id < 0:  # Group chat
            if chat_id not in self.chat_models:
                self.chat_models[chat_id] = DEFAULT_MODEL
            current_model = self.chat_models[chat_id]
        else:  # Personal chat
            current_model = self.user_models[user_id]
        
        # Get localized greeting
        greeting = get_text("start_greeting", user.mention_html(), current_model)
        
        await update.message.reply_html(greeting)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Set default model for new user/chat
        if user_id not in self.user_models:
            self.user_models[user_id] = DEFAULT_MODEL
        
        if chat_id < 0:  # Group chat
            if chat_id not in self.chat_models:
                self.chat_models[chat_id] = DEFAULT_MODEL
            current_model = self.chat_models[chat_id]
            
            # Check if there are custom prompts for this chat
            has_custom_answer_prompt = chat_id in self.group_prompts["answer"]
            has_custom_summary_prompt = chat_id in self.group_prompts["summary"]
            
            # Get localized text for custom prompt status
            custom_answer_status = "✅" if has_custom_answer_prompt else "❌"
            custom_summary_status = "✅" if has_custom_summary_prompt else "❌"
            
            # Get localized help text
            help_text = get_text(
                "help_text_group",
                current_model,
                custom_answer_status,
                custom_summary_status,
                self.bot_config['context_messages'],
                context.bot.username
            )
        else:  # Personal chat
            current_model = self.user_models[user_id]
            
            # Check if user is in factual mode
            factual_status = get_text("yes_enabled") if user_id in self.factual_mode_users else get_text("no_disabled")
            
            # Get localized help text
            help_text = get_text(
                "help_text_personal",
                current_model,
                self.bot_config['context_messages'],
                factual_status
            )
        
        await update.message.reply_text(help_text)
    
    async def factual(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Toggle factual mode."""
        user_id = update.effective_user.id
        
        if user_id in self.factual_mode_users:
            self.factual_mode_users.remove(user_id)
            await update.message.reply_text(get_text("factual_disabled"))
        else:
            self.factual_mode_users.add(user_id)
            await update.message.reply_text(get_text("factual_enabled"))
    
    async def chatid(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show chat ID and information."""
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id
        
        if chat_id > 0:  # Personal chat
            chat_type = "Personal"
            model = self.user_models.get(user_id, DEFAULT_MODEL)
            factual = "Yes" if user_id in self.factual_mode_users else "No"
            temperature = self.chat_temperatures.get(chat_id, self.bot_config["temperature"])
            factual_temperature = self.chat_factual_temperatures.get(chat_id, self.bot_config["factual_temperature"])
            summary_temperature = self.chat_summary_temperatures.get(chat_id, self.bot_config["summary_temperature"])
            
            info_text = (
                f"Chat ID: {chat_id}\n"
                f"Chat type: {chat_type}\n"
                f"User ID: {user_id}\n"
                f"Model: {model}\n"
                f"Factual mode: {factual}\n"
                f"Temperature: {temperature}\n"
                f"Factual temperature: {factual_temperature}\n"
                f"Summary temperature: {summary_temperature}"
            )
        else:  # Group chat
            chat_type = "Group"
            model = self.chat_models.get(chat_id, DEFAULT_MODEL)
            temperature = self.chat_temperatures.get(chat_id, self.bot_config["temperature"])
            factual_temperature = self.chat_factual_temperatures.get(chat_id, self.bot_config["factual_temperature"])
            summary_temperature = self.chat_summary_temperatures.get(chat_id, self.bot_config["summary_temperature"])
            
            info_text = (
                f"Chat ID: {chat_id}\n"
                f"Chat type: {chat_type}\n"
                f"Model: {model}\n"
                f"Temperature: {temperature}\n"
                f"Factual temperature: {factual_temperature}\n"
                f"Summary temperature: {summary_temperature}"
            )
        
        await update.message.reply_text(info_text)
    
    async def reload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reload modules (for developers)."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Only allow in personal chats
        if chat_id < 0:
            await update.message.reply_text("This command is only available in personal chats.")
            return
        
        # Get module name from arguments
        if not context.args:
            await update.message.reply_text("Please specify a module to reload.")
            return
        
        module_name = context.args[0]
        
        try:
            # Try to reload the module
            if module_name in sys.modules:
                module = importlib.reload(sys.modules[module_name])
                await update.message.reply_text(f"Module {module_name} reloaded successfully.")
            else:
                # Try to import the module first
                module = importlib.import_module(module_name)
                await update.message.reply_text(f"Module {module_name} imported successfully.")
        except Exception as e:
            await update.message.reply_text(f"Error reloading module {module_name}: {e}")
    
    # Conversation handlers for model selection, prompt configuration, etc.
    # These would be implemented as needed
    
    async def model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle /model command to start model selection."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # Only allow in personal chats
        if chat_id < 0:
            await update.message.reply_text(get_text("personal_chat_only"))
            return ConversationHandler.END
        
        # Create keyboard with options
        keyboard = [
            [InlineKeyboardButton(get_text("for_me"), callback_data="model_for_me")],
            [InlineKeyboardButton(get_text("for_group_chat"), callback_data="model_for_group")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            get_text("choose_model_target"),
            reply_markup=reply_markup
        )
        
        return CHOOSING_MODEL_TARGET
    
    # Add more command handlers as needed
    
    async def hidden_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /hidden_summary command.
        Creates a summary that is only visible to the user who requested it.
        Only members of the same groups can use this command for a specific group.
        
        Args:
            update: Update object
            context: Context object
        """
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # If in a personal chat, user needs to specify a group chat ID
        if chat_id > 0:  # Personal chat
            # Check if chat ID is provided
            if not context.args:
                await update.message.reply_text(get_text("hidden_summary_usage"))
                return
            
            try:
                target_chat_id = int(context.args[0])
                
                # Verify that the target chat is a group chat
                if target_chat_id > 0:
                    await update.message.reply_text(get_text("not_a_group_chat"))
                    return
                
                # Check if the user is a member of the target group
                try:
                    # Try to get chat member info to verify membership
                    chat_member = await context.bot.get_chat_member(target_chat_id, user_id)
                    
                    # If we get here, the user is a member of the group
                    # Now proceed with summarization
                    
                    # Set typing action
                    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                    
                    # Check if there are messages to summarize
                    if target_chat_id not in self.user_messages or not self.user_messages[target_chat_id]:
                        await update.message.reply_text(get_text("no_messages_to_summarize"))
                        return
                    
                    # Limit number of messages to summarize
                    max_messages = min(len(self.user_messages[target_chat_id]), self.bot_config["summary_messages"])
                    messages_to_summarize = self.user_messages[target_chat_id][-max_messages:]
                    
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
                    used_model = self.chat_models.get(target_chat_id, DEFAULT_MODEL)
                    
                    # Determine temperature for summarization
                    temperature = self.chat_summary_temperatures.get(
                        target_chat_id, self.bot_config["summary_temperature"]
                    )
                    
                    # Get appropriate prompt
                    prompt_template = self.group_prompts["summary"].get(
                        target_chat_id, self.chat_prompts["summary"].get(user_id, None)
                    )
                    
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
                    
                    # Send summary only to the user who requested it
                    await update.message.reply_text(f"Hidden summary for chat {target_chat_id}:\n\n{summary}")
                    
                except Exception as e:
                    # User is not a member of the group or bot is not in the group
                    logger.error(f"Error checking membership or getting summary: {e}")
                    await update.message.reply_text(get_text("not_member_of_group"))
                    
            except ValueError:
                await update.message.reply_text(get_text("invalid_chat_id"))
                
        else:  # Already in a group chat
            # Set typing action
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Check if there are messages to summarize
            if chat_id not in self.user_messages or not self.user_messages[chat_id]:
                await update.message.reply_text(get_text("no_messages_to_summarize"))
                return
            
            # Limit number of messages to summarize
            max_messages = min(len(self.user_messages[chat_id]), self.bot_config["summary_messages"])
            messages_to_summarize = self.user_messages[chat_id][-max_messages:]
            
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
            used_model = self.chat_models.get(chat_id, DEFAULT_MODEL)
            
            # Determine temperature for summarization
            temperature = self.chat_summary_temperatures.get(
                chat_id, self.bot_config["summary_temperature"]
            )
            
            # Get appropriate prompt
            prompt_template = self.group_prompts["summary"].get(
                chat_id, self.chat_prompts["summary"].get(user_id, None)
            )
            
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
            
            # Send summary only to the user who requested it
            await update.message.reply_text(f"Hidden summary:\n\n{summary}")
    
    async def hidden_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle /hidden_query command.
        Sends a query that is only visible to the user who sent it.
        Only members of the same groups can use this command for a specific group.
        
        Args:
            update: Update object
            context: Context object
        """
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        
        # If in a personal chat, user needs to specify a group chat ID and query
        if chat_id > 0:  # Personal chat
            # Check if chat ID and query are provided
            if len(context.args) < 2:
                await update.message.reply_text(get_text("hidden_query_usage"))
                return
            
            try:
                target_chat_id = int(context.args[0])
                query = " ".join(context.args[1:])
                
                # Verify that the target chat is a group chat
                if target_chat_id > 0:
                    await update.message.reply_text(get_text("not_a_group_chat"))
                    return
                
                # Check if the user is a member of the target group
                try:
                    # Try to get chat member info to verify membership
                    chat_member = await context.bot.get_chat_member(target_chat_id, user_id)
                    
                    # If we get here, the user is a member of the group
                    # Now proceed with the query
                    
                    # Set typing action
                    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
                    
                    # Determine which model to use
                    used_model = self.chat_models.get(target_chat_id, DEFAULT_MODEL)
                    
                    # Determine temperature
                    is_factual = user_id in self.factual_mode_users
                    
                    if is_factual:
                        temperature = self.chat_factual_temperatures.get(
                            target_chat_id, self.bot_config["factual_temperature"]
                        )
                    else:
                        temperature = self.chat_temperatures.get(
                            target_chat_id, self.bot_config["temperature"]
                        )
                    
                    # Get appropriate prompt
                    prompt_template = self.group_prompts["answer"].get(
                        target_chat_id, self.chat_prompts["answer"].get(user_id, None)
                    )
                    
                    # If no custom prompt, use default
                    if not prompt_template:
                        default_prompts = get_default_prompts()
                        prompt_template = default_prompts["answer"]
                    
                    # Format prompt with question
                    prompt = prompt_template.format(question=query)
                    
                    # Get response from model
                    response = await self.llm_client.answer_question(
                        prompt=prompt,
                        model=used_model,
                        temperature=temperature
                    )
                    
                    # Send response only to the user who requested it
                    await update.message.reply_text(f"Hidden query for chat {target_chat_id}:\n\nQ: {query}\n\nA: {response}")
                    
                except Exception as e:
                    # User is not a member of the group or bot is not in the group
                    logger.error(f"Error checking membership or getting response: {e}")
                    await update.message.reply_text(get_text("not_member_of_group"))
                    
            except ValueError:
                await update.message.reply_text(get_text("invalid_chat_id"))
                
        else:  # Already in a group chat
            # Check if query is provided
            if not context.args:
                await update.message.reply_text(get_text("hidden_query_in_group_usage"))
                return
            
            query = " ".join(context.args)
            
            # Set typing action
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            
            # Determine which model to use
            used_model = self.chat_models.get(chat_id, DEFAULT_MODEL)
            
            # Determine temperature
            is_factual = user_id in self.factual_mode_users
            
            if is_factual:
                temperature = self.chat_factual_temperatures.get(
                    chat_id, self.bot_config["factual_temperature"]
                )
            else:
                temperature = self.chat_temperatures.get(
                    chat_id, self.bot_config["temperature"]
                )
            
            # Get appropriate prompt
            prompt_template = self.group_prompts["answer"].get(
                chat_id, self.chat_prompts["answer"].get(user_id, None)
            )
            
            # If no custom prompt, use default
            if not prompt_template:
                default_prompts = get_default_prompts()
                prompt_template = default_prompts["answer"]
            
            # Format prompt with question
            prompt = prompt_template.format(question=query)
            
            # Get response from model
            response = await self.llm_client.answer_question(
                prompt=prompt,
                model=used_model,
                temperature=temperature
            )
            
            # Send response only to the user who requested it
            await update.message.reply_text(f"Hidden query:\n\nQ: {query}\n\nA: {response}")