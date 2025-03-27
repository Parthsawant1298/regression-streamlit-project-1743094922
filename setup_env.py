import subprocess
import os
import sys

def setup_venv():
    print("Setting up virtual environment...")
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Get the pip path based on OS
    pip_path = os.path.join("venv", "Scripts", "pip") if os.name == "nt" else os.path.join("venv", "bin", "pip")
    
    # Install packages
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])
    print("Virtual environment setup complete!")

if __name__ == "__main__":
    setup_venv()
