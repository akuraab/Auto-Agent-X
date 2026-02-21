import logging
import sys
import os
import json
from datetime import datetime
from structlog import configure, processors, stdlib, make_filtering_bound_logger

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

THOUGHT_LOG_FILE = os.path.join(LOG_DIR, "thought_process.log")
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")

class ThoughtProcessLogger:
    """ specialized logger for model thought process """
    
    @staticmethod
    def log(session_id: str, step: str, details: dict):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "step": step,
            "details": details
        }
        
        # Format as a readable block for analysis
        log_message = f"\n{'='*50}\n"
        log_message += f"TIME: {entry['timestamp']}\n"
        log_message += f"SESSION: {session_id}\n"
        log_message += f"STEP: {step}\n"
        log_message += f"DETAILS:\n{json.dumps(details, indent=2, ensure_ascii=False)}\n"
        log_message += f"{'='*50}\n"
        
        try:
            with open(THOUGHT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_message)
        except Exception as e:
            print(f"Failed to write to thought log: {e}")

def setup_logging(log_level: str = "INFO"):
    configure(
        processors=[
            processors.TimeStamper(fmt="iso"),
            processors.JSONRenderer()
        ],
        logger_factory=stdlib.LoggerFactory(),
        wrapper_class=make_filtering_bound_logger(logging.getLevelName(log_level)),
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console_handler)
    
    # File Handler for general app logs
    file_handler = logging.FileHandler(APP_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}'
    ))
    root_logger.addHandler(file_handler)
