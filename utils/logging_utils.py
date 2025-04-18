#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from config.settings import LOG_LEVEL

def setup_logging():
    """
    Configure logging for the application.
    """
    # Get log level from settings
    log_level_str = LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Reset basic logging configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=log_level,
        stream=sys.stdout
    )
    
    # Set log level for specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.INFO)
    
    # Get the root logger
    logger = logging.getLogger()
    logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")
    
    return log_level

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: The name of the logger
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Get log level from settings
    log_level_str = LOG_LEVEL
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    logger.setLevel(log_level)
    return logger