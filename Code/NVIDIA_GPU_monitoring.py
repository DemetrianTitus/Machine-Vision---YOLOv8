import os
import time
import subprocess

def clear_terminal():
    # Clear the terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor_gpu(interval=10):
    while True:
        clear_terminal()
        # Run the nvidia-smi command and capture its output
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
        time.sleep(interval)

if __name__ == "__main__":
    monitor_gpu(interval=10)
