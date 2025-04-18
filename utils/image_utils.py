#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import imghdr
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from PIL import Image
import io

from config.settings import TEMP_IMAGE_DIR
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def save_image(image_path: str, message_id: Optional[str] = None) -> str:
    """
    Save an image to the temporary directory with a unique name.
    
    Args:
        image_path: Path to the original image file
        message_id: Optional message ID to use in the filename
        
    Returns:
        Path to the saved image
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Generate a unique filename based on message_id or uuid
    if message_id:
        # Use message_id and current date to create a unique name
        file_id = f"{message_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        # If message_id is not provided, use UUID
        file_id = str(uuid.uuid4())
    
    # Determine image type
    with open(image_path, "rb") as img_file:
        img_type = imghdr.what(None, h=img_file.read(2048)) or "jpeg"
    
    # Create new filename with correct extension
    new_filename = f"{file_id}.{img_type}"
    new_image_path = os.path.join(TEMP_IMAGE_DIR, new_filename)
    
    # Copy image to temporary storage
    try:
        shutil.copy2(image_path, new_image_path)
        logger.info(f"Image saved to: {new_image_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        # Continue with original path if copy fails
        new_image_path = image_path
    
    return new_image_path

def encode_image_base64(image_path: str) -> Tuple[str, str, str]:
    """
    Encode an image to base64 format.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (base64_raw, base64_with_mime, image_type)
    """
    # Read image and determine its type
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        img_type = imghdr.what(None, h=image_bytes) or "jpeg"
        logger.info(f"Determined image type: {img_type}")
        
        # Encode image to base64
        image_base64_raw = base64.b64encode(image_bytes).decode("utf-8")
        image_base64_with_mime = f"data:image/{img_type};base64,{image_base64_raw}"
        
        logger.info(f"Image encoded to base64, length: {len(image_base64_raw)} characters")
        
        # Verify base64 encoding
        try:
            # Check if we can decode it back
            test_decode = base64.b64decode(image_base64_raw)
            logger.info(f"Base64 verified, correct")
        except Exception as e:
            logger.error(f"Error verifying base64: {e}")
            # Try to fix base64 by removing invalid characters
            image_base64_raw = ''.join(c for c in image_base64_raw if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
            
            # Check base64 length and add padding if needed
            padding = len(image_base64_raw) % 4
            if padding:
                image_base64_raw += '=' * (4 - padding)
            
            # Check again
            try:
                test_decode = base64.b64decode(image_base64_raw)
                logger.info(f"Base64 fixed and verified, correct")
            except Exception as e:
                logger.error(f"Could not fix base64: {e}")
                # Try to re-encode the image
                image_base64_raw = base64.b64encode(image_bytes).decode("utf-8")
                logger.info(f"Image re-encoded to base64")
            
            # Add MIME type
            image_base64_with_mime = f"data:image/{img_type};base64,{image_base64_raw}"
        
        return image_base64_raw, image_base64_with_mime, img_type

def resize_image_if_needed(image_path: str, max_size: int = 4*1024*1024, max_dimension: int = 1024) -> Tuple[str, int, int]:
    """
    Resize an image if it's too large.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum file size in bytes
        max_dimension: Maximum dimension (width or height) in pixels
        
    Returns:
        Tuple of (base64_encoded_image, width, height)
    """
    # Check image size
    image_size = os.path.getsize(image_path)
    logger.info(f"Image size: {image_size} bytes")
    
    try:
        # Open image
        img = Image.open(image_path)
        width, height = img.size
        logger.info(f"Image dimensions: {width}x{height} pixels")
        
        # Check if resizing is needed
        resize_needed = image_size > max_size or width > max_dimension or height > max_dimension
        
        if resize_needed:
            logger.warning(f"Image too large ({image_size} bytes, {width}x{height}), resizing")
            
            # Resize image while maintaining aspect ratio
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Image resized to {new_width}x{new_height}")
            
            # Save image to temporary buffer with quality optimization
            buffer = io.BytesIO()
            
            # Determine format for saving
            save_format = img.format if img.format else "JPEG"
            
            # If format is JPEG or PNG, we can manage quality
            if save_format == "JPEG":
                img.save(buffer, format=save_format, quality=85, optimize=True)
            elif save_format == "PNG":
                img.save(buffer, format=save_format, optimize=True, compress_level=9)
            else:
                img.save(buffer, format=save_format)
            
            buffer.seek(0)
            
            # Encode image to base64
            image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            
            # Check new size
            new_size = len(image_base64)
            logger.info(f"Resized image base64 size: {new_size} bytes")
            
            # If image is still too large, reduce quality further
            if new_size > max_size:
                logger.warning(f"Image still too large ({new_size} bytes), reducing quality")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=50, optimize=True)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
                logger.info(f"Image size after heavy compression: {len(image_base64)} bytes")
            
            return image_base64, new_width, new_height
        else:
            # If no resizing needed, just encode the original image
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            return image_base64, width, height
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        # If error occurs, just encode the original image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return image_base64, 0, 0  # Return 0 for dimensions as we couldn't determine them

def prepare_image_for_model(image_path: str, model: str) -> Dict[str, Any]:
    """
    Prepare an image for use with a specific model.
    
    Args:
        image_path: Path to the image file
        model: Name of the model to use
        
    Returns:
        Dictionary with image data ready for the model
    """
    # Encode image to base64
    image_base64_raw, image_base64_with_mime, img_type = encode_image_base64(image_path)
    
    # Check if model supports images
    vision_models = ["llava", "bakllava", "gemma", "phi3", "claude", "gpt4", "cogvlm", "qwen"]
    is_vision_model = any(model_name in model.lower() for model_name in vision_models)
    
    if not is_vision_model:
        logger.warning(f"Model {model} may not support image processing")
    
    # Special handling for different models
    is_gemma3 = "gemma3" in model.lower()
    is_llava = any(x in model.lower() for x in ["llava", "bakllava"])
    
    result = {
        "image_base64_raw": image_base64_raw,
        "image_base64_with_mime": image_base64_with_mime,
        "img_type": img_type,
        "is_vision_model": is_vision_model,
        "is_gemma3": is_gemma3,
        "is_llava": is_llava
    }
    
    return result