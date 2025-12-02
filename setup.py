#!/usr/bin/env python
"""
Setup script for SelfIndex project
Installs dependencies and sets up the environment
"""

import subprocess
import sys
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("\nDownloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"✗ Failed to download NLTK data: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    dirs = ['indices', 'config', 'src', 'plot']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    return True

def main():
    print("="*70)
    print("SelfIndex Setup Script")
    print("="*70)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not download_nltk_data():
        print("⚠️  NLTK data download failed, but continuing...")
    
    if not create_directories():
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ Setup complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Configure parameters in config/index_config.yaml")
    print("2. Run: jupyter notebook inference.ipynb")
    print("3. Follow the notebook cells for building/loading indices and querying")

if __name__ == '__main__':
    main()
