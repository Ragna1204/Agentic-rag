"""
Handles logging and provenance tracking for the agentic RAG pipeline.

A centralized logging setup is crucial for debugging and understanding the agent's
behavior. This module configures a logger that can be imported and used by any
other module in the project.

Provenance tracking is also managed here, allowing the system to trace back
any piece of information to its original source.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_level: str = "INFO", log_file: str = "agentic_rag.log"):
    """
    Configures the root logger for the entire application.

    Args:
        log_level (str): The minimum logging level (e.g., "INFO", "DEBUG").
        log_file (str): The path to the log file.
    """
    # TODO: Add more sophisticated logging, such as structured logging (e.g., JSON format)
    # which can be easily parsed by log management systems.

    logger = logging.getLogger("agentic_rag")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Prevent adding multiple handlers if the function is called more than once
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File Handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Logging configured successfully.")
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for a specific module.

    Args:
        name (str): The name of the module (e.g., __name__).

    Returns:
        logging.Logger: An instance of the logger.
    """
    return logging.getLogger(f"agentic_rag.{name}")

# TODO: Implement a more robust provenance tracking system.
# This could be a class that logs each step of the process (retrieval, summary, verification)
# and associates the final answer with all the intermediate artifacts and their sources.
class ProvenanceTracker:
    """
    A simple class to track the origin and transformation of data.
    """
    def __init__(self):
        self.trace = []

    def add_step(self, step_name: str, inputs: dict, outputs: dict):
        """
        Logs a step in the data processing pipeline.
        """
        self.trace.append({
            "step": step_name,
            "inputs": inputs,
            "outputs": outputs
        })

    def get_trace(self):
        """
        Returns the full trace of all tracked steps.
        """
        return self.trace
