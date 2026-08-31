import os
import time
import subprocess
from datetime import datetime

LOG_FILE = "TRAINING_LOG.md"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {msg}\n"
    print(entry, end="")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)

def run_command(cmd: str, description: str):
    log(f"Starting step: {description} ('{cmd}')")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        log(f"Successfully completed: {description}")
    else:
        log(f"Error in {description}: {res.stderr}")
    return res.returncode == 0

def execute_daily_training_pipeline():
    log("=== Starting Daily Continuous Model Fine-Tuning Pipeline ===")

    # 1. Dataset preprocessing
    run_command("python3 models/nano_pixel_art_v1/scripts/dataset_processor.py", "NanoPixel v1 Data Processing")
    run_command("python3 models/nano_pixel_XL0_2/scripts/dataset_processor.py", "NanoPixel XL0_2 Data Processing")
    run_command("python3 models/nano_pixel_3A_xl/scripts/dataset_processor.py", "NanoPixel 3A XL Data Processing")

    # 2. Model Training
    run_command("PYTHONPATH=. python3 models/nano_pixel_art_v1/scripts/train_nano_pixel.py", "NanoPixel v1 Training & ONNX Export")
    run_command("PYTHONPATH=. python3 models/nano_pixel_XL0_2/scripts/train_nano_pixel.py", "NanoPixel XL0_2 Training")
    run_command("python3 models/nano_pixel_3A_xl/scripts/train_nano_pixel_3A_xl.py", "NanoPixel 3A XL Training")
    run_command("python3 models/nano_pixel_3A_xl/scripts/export_onnx.py", "NanoPixel 3A XL ONNX Export")
    run_command("python3 poster_generator_256/arabic_dataset_trainer.py", "Poster Generator 256 Training")
    run_command("python3 poster_generator_256/export_poster_onnx.py", "Poster Generator 256 ONNX Export")

    # 3. Showcase Generation
    run_command("python3 generate_39_chars.py", "Game Boy / NES 39 Character Showcase Generation")
    run_command("python3 generate_perfect_direction_arabic_posters.py", "Arabic Poster Showcase Generation")

    log("=== Daily Continuous Model Fine-Tuning Pipeline Completed Successfully ===")

if __name__ == "__main__":
    execute_daily_training_pipeline()
