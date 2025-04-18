#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import aiohttp
import asyncio
import re
from typing import Dict, Any, Optional, List, Union

from config.settings import LLM_API_URL, DEFAULT_MODEL, LLM_PROVIDER
from utils.logging_utils import get_logger

logger = get_logger(__name__)

async def check_ollama_availability(url: str = LLM_API_URL) -> bool:
    """
    Check if Ollama API is available.

    Args:
        url: Ollama API URL

    Returns:
        True if API is available, False otherwise
    """
    try:
        # For Ollama 0.6.x and above, use /api/tags endpoint
        check_url = f"{url}/tags" if url.endswith('/api') else f"{url}/api/tags"
        logger.info(f"Checking Ollama API availability at URL: {check_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(check_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Received response: {json.dumps(data, ensure_ascii=False)}")

                    # Response format from /api/tags in Ollama 0.6.x
                    models = [model.get("name", "") for model in data.get("models", [])]
                    models = [model for model in models if model]  # Filter empty values

                    logger.info(f"Ollama API is available. Available models: {', '.join(models)}")
                    return True
                else:
                    logger.error(f"Ollama API is not available. Status: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error checking Ollama API availability: {e}")
        return False

async def check_lmstudio_availability(url: str = LLM_API_URL) -> bool:
    """
    Check if LM Studio API is available.

    Args:
        url: LM Studio API URL

    Returns:
        True if API is available, False otherwise
    """
    try:
        # LM Studio uses OpenAI-compatible API
        # We'll check the /models endpoint
        check_url = f"{url}/models"
        logger.info(f"Checking LM Studio API availability at URL: {check_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(check_url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.debug(f"Received response: {json.dumps(data, ensure_ascii=False)}")

                    # Extract model IDs from the response
                    models = [model.get("id", "") for model in data.get("data", [])]
                    models = [model for model in models if model]  # Filter empty values

                    logger.info(f"LM Studio API is available. Available models: {', '.join(models)}")
                    return True
                else:
                    logger.error(f"LM Studio API is not available. Status: {response.status}")
                    return False
    except Exception as e:
        logger.error(f"Error checking LM Studio API availability: {e}")
        return False

async def check_llm_availability() -> bool:
    """
    Check if the configured LLM API is available.

    Returns:
        True if API is available, False otherwise
    """
    if LLM_PROVIDER == "ollama":
        return await check_ollama_availability()
    elif LLM_PROVIDER == "lmstudio":
        return await check_lmstudio_availability()
    else:
        logger.error(f"Unknown LLM provider: {LLM_PROVIDER}")
        return False


class LocalLLMClient:
    """Client for interacting with a local LLM model via Ollama or LM Studio."""

    def filter_think_tags(self, text: str) -> str:
        """
        Filter <think> tags from text, but keep their content in logs.

        Args:
            text: Original text

        Returns:
            Text without <think> tag content
        """
        # Find all content between <think> and </think> tags
        think_tags = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)

        # Log <think> tag content for debugging
        if think_tags:
            logger.debug("Found <think> tags in model response:")
            for i, think_content in enumerate(think_tags):
                logger.debug(f"<think> #{i+1}:\n{think_content.strip()}")

        # Remove all content between <think> and </think> tags, including the tags
        filtered_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Find single <think> tags at the end of text
        single_think = re.search(r'<think>.*$', filtered_text, flags=re.DOTALL)
        if single_think:
            logger.debug(f"Found single <think> tag at end of response:\n{single_think.group(0)[7:].strip()}")

        # Remove single <think> tags at the end of text
        filtered_text = re.sub(r'<think>.*$', '', filtered_text, flags=re.DOTALL)

        # Remove empty lines that may have formed after removing tags
        filtered_text = re.sub(r'\n\s*\n', '\n\n', filtered_text)

        return filtered_text.strip()

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None):
        """
        Initialize the client.

        Args:
            base_url: LLM API URL. Defaults to environment variable.
            model: Model name to use. Defaults to environment variable.
            provider: LLM provider to use. Defaults to environment variable.
        """
        self.base_url = base_url or LLM_API_URL
        self.model = model or DEFAULT_MODEL
        self.provider = provider or LLM_PROVIDER

        # Set up the appropriate API endpoint based on the provider
        if self.provider == "ollama":
            # Check if URL ends with /api
            if self.base_url.endswith('/api'):
                self.generate_url = f"{self.base_url}/generate"
            else:
                self.generate_url = f"{self.base_url}/api/generate"
        elif self.provider == "lmstudio":
            # LM Studio uses OpenAI-compatible API
            self.generate_url = f"{self.base_url}/chat/completions"
        else:
            logger.error(f"Unknown LLM provider: {self.provider}")
            self.generate_url = f"{self.base_url}/generate"  # Default to Ollama-style endpoint

        logger.info(f"Client initialized with URL: {self.base_url}, model: {self.model}, provider: {self.provider}")
        logger.info(f"Generation URL: {self.generate_url}")

    def format_prompt_for_model(self, prompt: str, model: str) -> str:
        """
        Format prompt according to specific model requirements.
        
        Args:
            prompt: Original prompt
            model: Model name
            
        Returns:
            Formatted prompt
        """
        # Log prompt information
        logger.info(f"Formatting prompt for model: {model}")
        logger.debug(f"Original prompt (first 200 chars): {prompt[:200]}...")
        
        # Check if model is DeepSeek or Gemma
        if "deepseek" in model.lower() or "gemma" in model.lower():
            model_name = "DeepSeek" if "deepseek" in model.lower() else "Gemma"
            logger.info(f"Detected {model_name} model: {model}, applying special formatting")
            
            # Check if prompt contains <conversation> and <message> tags
            if "<conversation>" in prompt and "<message" in prompt:
                logger.info(f"Detected dialog format, converting to {model_name} format")
                
                # Extract the entire conversation section
                conversation_match = re.search(r'<conversation>(.*?)</conversation>', prompt, re.DOTALL)
                if not conversation_match:
                    logger.warning("Could not extract conversation section, using base format")
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    return formatted_prompt
                
                conversation_content = conversation_match.group(1).strip()
                
                # Extract all messages with their attributes
                # Use more robust regular expression
                message_patterns = re.findall(r'<message\s+([^>]*)>(.*?)</message>', conversation_content, re.DOTALL)
                
                # Log information about found messages
                logger.info(f"Found {len(message_patterns)} messages in conversation section")
                if not message_patterns and conversation_content:
                    # If no messages found but content exists, log first 200 chars for debugging
                    logger.warning(f"Conversation content (first 200 chars): {conversation_content[:200]}")
                
                if not message_patterns:
                    logger.warning("Could not extract messages from conversation, using base format")
                    formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
                    return formatted_prompt
                
                # Format dialog in DeepSeek/Gemma format
                formatted_prompt = ""
                
                # Dictionary to store messages by index
                message_dict = {}
                
                # First collect all messages and their indices
                for i, (attrs, content) in enumerate(message_patterns):
                    sender_match = re.search(r'sender="([^"]*)"', attrs)
                    sender = sender_match.group(1) if sender_match else "Unknown"
                    
                    # Determine role (user or model)
                    role = "model" if sender.lower() == "bot" else "user"
                    
                    # Save message in dictionary
                    message_dict[i] = {
                        "role": role,
                        "content": content.strip(),
                        "sender": sender,
                        "attrs": attrs
                    }
                
                # Now process messages considering replies
                for i, (attrs, content) in enumerate(message_patterns):
                    # Check if message is a reply to another
                    reply_to_match = re.search(r'reply_to="([^"]*)"', attrs)
                    reply_text_match = re.search(r'reply_text="([^"]*)"', attrs)
                    reply_to_id_match = re.search(r'reply_to_id="([^"]*)"', attrs)
                    
                    role = message_dict[i]["role"]
                    clean_content = message_dict[i]["content"]
                    
                    # If this is a reply to another message, add context
                    # Check for either reply_to and reply_text pair, or reply_to_id
                    if (reply_to_match and reply_text_match) or reply_to_id_match:
                        # If there's reply_to and reply_text, use them
                        if reply_to_match and reply_text_match:
                            reply_to = reply_to_match.group(1)
                            reply_text = reply_text_match.group(1)
                            
                            # Add information about what the user is replying to
                            clean_content = f"[In response to message from {reply_to}: \"{reply_text}\"] {clean_content}"
                        # If there's only reply_to_id but no text, add only ID
                        elif reply_to_id_match:
                            reply_id = reply_to_id_match.group(1)
                            reply_to = reply_to_match.group(1) if reply_to_match else "Unknown"
                            
                            # Add information about what the user is replying to (ID only)
                            clean_content = f"[In response to message #{reply_id} from {reply_to}] {clean_content}"
                    
                    # Add message in model format
                    formatted_prompt += f"<start_of_turn>{role}\n{clean_content}<end_of_turn>\n"
                
                # Extract current message (outside conversation tags)
                current_message_match = re.search(r'</conversation>.*?<message\s+([^>]*)>(.*?)</message>', prompt, re.DOTALL)
                
                # If current message with attributes not found, try without attributes
                if not current_message_match:
                    current_message_match = re.search(r'</conversation>.*?<message>(.*?)</message>', prompt, re.DOTALL)
                    if current_message_match:
                        # If message without attributes found, create empty attributes
                        logger.info("Found current message without attributes")
                        current_attrs = ""
                        current_content = current_message_match.group(1).strip()
                    else:
                        logger.warning("Could not find current message after </conversation> tag")
                
                if current_message_match and not 'current_attrs' in locals():
                    # If we haven't set current_attrs and current_content yet
                    if len(current_message_match.groups()) == 2:
                        current_attrs = current_message_match.group(1)
                        current_content = current_message_match.group(2).strip()
                    elif len(current_message_match.groups()) == 1:
                        current_attrs = ""
                        current_content = current_message_match.group(1).strip()
                    
                    # Check if there are attributes
                    if current_attrs:
                        sender_match = re.search(r'sender="([^"]*)"', current_attrs)
                        current_sender = sender_match.group(1) if sender_match else "Unknown"
                    else:
                        current_sender = "Unknown"
                    
                    # Check if message is a reply to another
                    if current_attrs:
                        reply_to_match = re.search(r'reply_to="([^"]*)"', current_attrs)
                        reply_text_match = re.search(r'reply_text="([^"]*)"', current_attrs)
                        reply_to_id_match = re.search(r'reply_to_id="([^"]*)"', current_attrs)
                    else:
                        reply_to_match = None
                        reply_text_match = None
                        reply_to_id_match = None
                    
                    # Determine role for current message
                    current_role = "model" if current_sender.lower() == "bot" else "user"
                    
                    # If this is a reply to another message, add context
                    # Check for either reply_to and reply_text pair, or reply_to_id
                    if (reply_to_match and reply_text_match) or reply_to_id_match:
                        # If there's reply_to and reply_text, use them
                        if reply_to_match and reply_text_match:
                            reply_to = reply_to_match.group(1)
                            reply_text = reply_text_match.group(1)
                            
                            # Add information about what the user is replying to
                            current_content = f"[In response to message from {reply_to}: \"{reply_text}\"] {current_content}"
                        # If there's only reply_to_id but no text, add only ID
                        elif reply_to_id_match:
                            reply_id = reply_to_id_match.group(1)
                            reply_to = reply_to_match.group(1) if reply_to_match else "Unknown"
                            
                            # Add information about what the user is replying to (ID only)
                            current_content = f"[In response to message #{reply_id} from {reply_to}] {current_content}"
                    
                    # Add current message in model format
                    formatted_prompt += f"<start_of_turn>{current_role}\n{current_content}<end_of_turn>\n"
                
                # Add final model turn
                formatted_prompt += "<start_of_turn>model\n"
                
                return formatted_prompt
        
        # If no special formatting needed, return original prompt
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
        Get completion from the model.

        Args:
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation (0.1 to 1.0)
            stop: List of stop sequences
            model: Model to use (if different from default)

        Returns:
            Generated text
        """
        used_model = model or self.model
        
        # Format prompt for specific models if using Ollama
        formatted_prompt = self.format_prompt_for_model(prompt, used_model) if self.provider == "ollama" else prompt
        
        # Prepare request data based on provider
        if self.provider == "ollama":
            # Ollama API format
            request_data = {
                "model": used_model,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add stop sequences if provided
            if stop:
                request_data["stop"] = stop
        
        elif self.provider == "lmstudio":
            # LM Studio uses OpenAI-compatible API
            messages = [{"role": "user", "content": formatted_prompt}]
            
            request_data = {
                "model": used_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add stop sequences if provided
            if stop:
                request_data["stop"] = stop
        
        else:
            # Default to Ollama format
            request_data = {
                "model": used_model,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Add stop sequences if provided
            if stop:
                request_data["stop"] = stop
        
        # Log request information
        logger.info(f"Sending request to model {used_model} with temperature {temperature} using {self.provider} provider")
        logger.debug(f"Prompt (first 200 chars): {formatted_prompt[:200]}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.generate_url, json=request_data, timeout=60) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API request error: {response.status}, {error_text}")
                        return f"Error: API request failed with status {response.status}. Details: {error_text}"
                    
                    # Parse response based on provider
                    result = await response.json()
                    
                    if self.provider == "ollama":
                        # Ollama response format
                        response_text = result.get("response", "")
                    elif self.provider == "lmstudio":
                        # LM Studio (OpenAI-compatible) response format
                        choices = result.get("choices", [])
                        if choices and len(choices) > 0:
                            response_text = choices[0].get("message", {}).get("content", "")
                        else:
                            response_text = ""
                    else:
                        # Default to Ollama format
                        response_text = result.get("response", "")
                    
                    # Log response information
                    logger.info(f"Received response from model, length: {len(response_text)} characters")
                    logger.debug(f"Response (first 200 chars): {response_text[:200]}...")
                    
                    # Process response for specific models if using Ollama
                    if self.provider == "ollama" and ("deepseek" in used_model.lower() or "gemma" in used_model.lower()):
                        model_name = "DeepSeek" if "deepseek" in used_model.lower() else "Gemma"
                        logger.info(f"Processing response from {model_name} model: {used_model}")
                        
                        # Save original length for logging
                        original_length = len(response_text)
                        
                        # Remove <start_of_turn> and <end_of_turn> tags
                        patterns = [
                            r'<start_of_turn>model\n',
                            r'<start_of_turn>model',
                            r'<end_of_turn>',
                            r'<start_of_turn>user.*?<end_of_turn>\n?',
                            r'<start_of_turn>user.*?$',  # If tag is open but not closed
                            r'<start_of_turn>.*?<end_of_turn>\n?',  # Any other tags
                        ]
                        
                        for pattern in patterns:
                            response_text = re.sub(pattern, '', response_text, flags=re.DOTALL)
                        
                        # Remove extra spaces and line breaks
                        response_text = re.sub(r'\n\s*\n', '\n\n', response_text)
                        response_text = response_text.strip()
                        
                        logger.info(f"Response from {model_name} after tag processing: was {original_length}, now {len(response_text)} characters")
                    
                    # Filter <think> tags
                    filtered_response = self.filter_think_tags(response_text)
                    
                    # Calculate difference in length before and after filtering
                    diff = len(response_text) - len(filtered_response)
                    if diff > 0:
                        logger.info(f"Response length: {len(response_text)} characters, removed {diff} characters from <think> tags, final length: {len(filtered_response)} characters")
                    else:
                        logger.info(f"Response length: {len(response_text)} characters (no <think> tags found)")
                    
                    return filtered_response
                    
        except aiohttp.ClientError as e:
            logger.error(f"Connection error: {e}")
            return f"Error: Could not connect to the server. Details: {str(e)}"
        except Exception as e:
            logger.error(f"Error getting completion: {e}")
            return f"Error: {str(e)}"

    async def answer_question(self, prompt: str, model: Optional[str] = None, temperature: float = 0.7) -> str:
        """
        Answer a user question.

        Args:
            prompt: Full prompt for the answer (already formatted)
            model: Model to use (if different from default)
            temperature: Temperature for generation (0.1 to 1.0)

        Returns:
            Answer to the question
        """
        used_model = model or self.model
        
        # Determine stop words based on model
        stop_words = None
        
        # For DeepSeek and Gemma add special stop words
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