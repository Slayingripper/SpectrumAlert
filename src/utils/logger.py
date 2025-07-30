"""
Logging utilities for SpectrumAlert
"""

import logging
import logging.handlers
import os
from typing import Optional
from datetime import datetime


class SpectrumLogger:
    """Centralized logging for SpectrumAlert"""
    
    def __init__(self, name: str = "SpectrumAlert", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with file and console handlers"""
        if self.logger.handlers:
            return  # Logger already configured
        
        self.logger.setLevel(logging.DEBUG)
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for all logs
        log_file = os.path.join(self.log_dir, f"{self.name.lower()}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler
        error_log_file = os.path.join(self.log_dir, f"{self.name.lower()}_errors.log")
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=5*1024*1024, backupCount=3
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_file_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger"""
        return self.logger
    
    @staticmethod
    def setup_module_logger(module_name: str, log_dir: str = "logs") -> logging.Logger:
        """Setup logger for a specific module"""
        spectrum_logger = SpectrumLogger(f"SpectrumAlert.{module_name}", log_dir)
        return spectrum_logger.get_logger()


def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("SpectrumAlert.Performance")
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"{func.__name__} completed in {duration:.3f} seconds")
            return result
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.error(f"{func.__name__} failed after {duration:.3f} seconds: {e}")
            raise
    return wrapper
