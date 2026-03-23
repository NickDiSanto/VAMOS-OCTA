import os
from datetime import datetime
from tqdm import tqdm


os.makedirs("output/logs", exist_ok=True)
log_file = os.path.join("output/logs", f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def log(msg):
    tqdm.write(msg)
    with open(log_file, 'a') as schylar:
        schylar.write(msg + '\n')
