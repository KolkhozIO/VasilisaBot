#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import aiohttp
import asyncio
import re
from typing import Dict, Any, Optional, List, Union, Tuple

from config.settings import LLM_API_URL, DEFAULT_MODEL
from utils.logging_utils import get_logger
from utils.image_utils import (
    save_image, encode_image_base64, resize_image_if_needed, prepare_image_for_model
)
from models.llm_client import LocalLLMClient

logger = get_logger(__name__)

class ImageHandler:
    """Handler for processing images with LLM models."""
    
    def __init__(self, llm_client: LocalLLMClient):
        """
        Initialize the image handler.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
    
    async def process_image_request(
        self, 
        prompt: str, 
        image_path: str, 
        model: Optional[str] = None, 
        temperature: float = 0.7, 
        message_id: Optional[str] = None
    ) -> str:
        """
        Process a request with an image.
        
        Args:
            prompt: Text prompt
            image_path: Path to the image file
            model: Model to use (if different from default)
            temperature: Temperature for generation (0.1 to 1.0)
            message_id: Optional message ID for tracking
            
        Returns:
            Model response
        """
        used_model = model or self.llm_client.model
        
        # Check if image file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return f"Error: image file not found: {image_path}"
        
        # Save image to temporary storage
        try:
            temp_image_path = save_image(image_path, message_id)
            logger.info(f"Image saved to temporary storage: {temp_image_path}")
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            temp_image_path = image_path
        
        # Prepare image data for the model
        image_data = prepare_image_for_model(temp_image_path, used_model)
        
        # Check if model supports images
        if not image_data["is_vision_model"]:
            logger.warning(f"Model {used_model} may not support image processing")
            # Add warning to prompt
            prompt = f"[WARNING: You received a request with an image, but your model ({used_model}) may not support image processing. Please inform the user about this.]\n\n{prompt}"
        
        # Special handling for different models
        if image_data["is_gemma3"]:
            logger.info(f"Detected gemma3 model: {used_model}")
            
            # For gemma3, add special prefix to prompt
            if not prompt.startswith("<image>"):
                prompt = f"<image>\n{prompt}"
                logger.info("Added <image> tag to beginning of prompt for gemma3")
        
        elif image_data["is_llava"]:
            logger.info(f"Detected llava model: {used_model}")
            
            # For llava, use special prompt format
            # Make sure prompt doesn't contain special instructions
            if not "<image>" in prompt and not "[IMAGE]" in prompt:
                # Add USER: and ASSISTANT: for llava
                if not prompt.startswith("USER:"):
                    prompt = f"USER: {prompt}\nASSISTANT:"
                    logger.info("Added USER: and ASSISTANT: tags for llava")
        
        # Process [img-0] tag for Ollama compatibility
        if "<image>" in prompt:
            # Remove <image> tag from prompt
            prompt = prompt.replace("<image>", "")
            # Add [img-0] to end of prompt
            prompt = prompt.strip() + "\n\n[img-0]"
            logger.info(f"Image reference [img-0] added to end of prompt")
        
        # Create request variants for different Ollama versions
        request_variants = []
        
        # Variant 1: Ollama 0.7.x (images as array without MIME)
        request_data_1 = {
            "model": used_model,
            "prompt": prompt,
            "images": [image_data["image_base64_raw"]],  # Without MIME type
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1024
        }
        
        # Determine stop words based on model
        stop_words = None
        if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
            stop_words = [
                "<end_of_turn>", 
                "<start_of_turn>user", 
                "<start_of_turn>", 
                "user:", 
                "User:", 
                "Human:"
            ]
        
        if stop_words:
            request_data_1["stop"] = stop_words
        
        request_variants.append(("images as array without MIME", request_data_1))
        
        # Variant 2: Ollama 0.6.x (image as string without MIME)
        request_data_2 = {
            "model": used_model,
            "prompt": prompt,
            "image": image_data["image_base64_raw"],  # Without MIME type
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1024
        }
        if stop_words:
            request_data_2["stop"] = stop_words
        request_variants.append(("image as string without MIME", request_data_2))
        
        # Variant 3: Ollama 0.7.x and above (images as array with MIME)
        request_data_3 = {
            "model": used_model,
            "prompt": prompt,
            "images": [image_data["image_base64_with_mime"]],  # With MIME type
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1024
        }
        if stop_words:
            request_data_3["stop"] = stop_words
        request_variants.append(("images as array with MIME", request_data_3))
        
        # Variant 4: Ollama 0.6.x (image as string with MIME)
        request_data_4 = {
            "model": used_model,
            "prompt": prompt,
            "image": image_data["image_base64_with_mime"],  # With MIME type
            "stream": False,
            "temperature": temperature,
            "max_tokens": 1024
        }
        if stop_words:
            request_data_4["stop"] = stop_words
        request_variants.append(("image as string with MIME", request_data_4))
        
        # Special variants for specific models
        if image_data["is_gemma3"]:
            # For gemma3, always use base64 encoding for images
            # Variant 1: gemma3 with base64 and raw
            request_data_gemma = {
                "model": used_model,
                "prompt": prompt,
                "images": [image_data["image_base64_raw"]],  # Without MIME type
                "stream": False,
                "temperature": temperature,
                "max_tokens": 1024,
                "raw": True  # Special parameter for some models
            }
            if stop_words:
                request_data_gemma["stop"] = stop_words
            request_variants.append(("gemma3 with base64 and raw", request_data_gemma))
        
        elif image_data["is_llava"]:
            # For llava, always use base64 encoding for images
            # Variant 1: llava with base64 and num_ctx
            request_data_llava = {
                "model": used_model,
                "prompt": prompt,
                "images": [image_data["image_base64_raw"]],  # Without MIME type
                "stream": False,
                "temperature": temperature,
                "max_tokens": 1024,
                "options": {
                    "num_ctx": 4096  # Increased context for llava
                }
            }
            if stop_words:
                request_data_llava["stop"] = stop_words
            request_variants.append(("llava with base64 and num_ctx", request_data_llava))
        
        # Try different request formats until one works
        last_error = None
        response_text_raw = None
        
        async with aiohttp.ClientSession() as session:
            for format_name, current_request_data in request_variants:
                try:
                    # Log current request format
                    logger.info(f"Trying request format: {format_name}")
                    
                    # Log full request for debugging (without image)
                    debug_request = current_request_data.copy()
                    if "images" in debug_request:
                        if isinstance(debug_request["images"], list):
                            debug_request["images"] = ["<base64_image_data>"]
                        else:
                            debug_request["images"] = "<base64_image_data>"
                    if "image" in debug_request:
                        debug_request["image"] = "<base64_image_data>"
                    logger.debug(f"Sending request to API: {json.dumps(debug_request, ensure_ascii=False)}")
                    
                    # Increase timeout to 120 seconds for processing large images
                    async with session.post(self.llm_client.generate_url, json=current_request_data, timeout=120) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.warning(f"Error with request format {format_name}: {response.status}, {error_text}")
                            
                            # Check if error contains information that model doesn't support images
                            if "does not support images" in error_text.lower() or "no image support" in error_text.lower():
                                logger.error(f"Model {used_model} does not support images")
                                return f"Model {used_model} does not support image processing. Please use a model with multimodal support, such as llava, bakllava, or gemma3."
                            
                            last_error = f"Error with image request: {response.status}. Details: {error_text}"
                            continue  # Try next format
                        
                        # Get response
                        response_text_raw = await response.text()
                        logger.debug(f"Received response, length: {len(response_text_raw)}")
                        
                        # If we got a response, break out of loop
                        if response_text_raw:
                            logger.info(f"Successfully received response with format {format_name}")
                            break
                        else:
                            logger.warning(f"Received empty response with format {format_name}")
                            last_error = "Received empty response from server"
                
                except aiohttp.ClientError as e:
                    logger.warning(f"Connection error with format {format_name}: {e}")
                    last_error = f"Connection error: {e}"
                    continue  # Try next format
                except Exception as e:
                    logger.warning(f"Error with format {format_name}: {e}")
                    last_error = f"Error: {e}"
                    continue  # Try next format
            
            # If no format worked, return the last error
            if not response_text_raw:
                logger.error("All request formats failed")
                return f"Could not get response from model. {last_error}"
            
            # Process response
            try:
                # Parse JSON response
                result = json.loads(response_text_raw)
                logger.debug(f"Received JSON response: {json.dumps(result, ensure_ascii=False)[:200]}...")
                
                response_text = result.get("response", "")
                if not response_text:
                    logger.error("Received empty response text in JSON")
                    return "Model returned empty response. It may not have been able to process the image."
                
                # Process response for specific models
                if "deepseek" in used_model.lower() or "gemma" in used_model.lower():
                    model_name = "DeepSeek" if "deepseek" in used_model.lower() else "Gemma"
                    # Remove special tags from response
                    logger.info(f"Processing response from {model_name} model with image: {used_model}")
                    
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
                    
                    logger.info(f"Response from {model_name} with image after tag processing: was {original_length}, now {len(response_text)} characters")
                
                # Filter <think> tags
                filtered_response = self.llm_client.filter_think_tags(response_text)
                
                # Calculate difference in length before and after filtering
                diff = len(response_text) - len(filtered_response)
                if diff > 0:
                    logger.info(f"Response with image length: {len(response_text)} characters, removed {diff} characters from <think> tags, final length: {len(filtered_response)} characters")
                else:
                    logger.info(f"Response with image length: {len(response_text)} characters (no <think> tags found)")
                
                return filtered_response
                
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON in image response: {e}")
                logger.debug(f"Received response: {response_text_raw[:200]}...")
                
                # Check if there's text in the response that can be returned
                if response_text_raw and len(response_text_raw) > 10:
                    # Try to extract text from invalid JSON
                    text_match = re.search(r'"response"\s*:\s*"([^"]+)"', response_text_raw)
                    if text_match:
                        extracted_text = text_match.group(1)
                        logger.info(f"Extracted text from invalid JSON: {extracted_text[:100]}...")
                        return extracted_text
                    else:
                        # If text extraction failed, return error
                        logger.info("Could not extract text from response")
                        return "Received invalid response from server. Please try again."
                else:
                    return "Received invalid or empty response from server. The model may not support image processing."
        
        # This should never be reached, but just in case
        return "Error processing image request"