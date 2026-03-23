import os
from datetime import datetime

from tqdm import tqdm


LOG_DIR = os.path.join("output", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")


def log(msg):
    """Mirror messages to the console and the current run log."""
    tqdm.write(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as log_handle:
        log_handle.write(msg + "\n")
