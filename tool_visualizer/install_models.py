#!/usr/bin/env python3
"""
Post-installation script to download spaCy models
"""
import subprocess
import sys
import importlib.util

def download_spacy_model():
    """Download the spaCy English model if not already installed"""
    try:
        # Check if spaCy is installed
        if importlib.util.find_spec('spacy') is None:
            print("⚠️ spaCy not found. Please install the package first.")
            return False
        
        import spacy
        
        # Check if the model is already installed
        try:
            spacy.load('en_core_web_sm')
            print("✅ spaCy model 'en_core_web_sm' is already installed.")
            return True
        except OSError:
            # Model not found, download it
            print("📥 Downloading spaCy model 'en_core_web_sm'...")
            result = subprocess.run([
                sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ spaCy model downloaded successfully!")
                return True
            else:
                print(f"❌ Failed to download spaCy model: {result.stderr}")
                return False
                
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")
        return False

if __name__ == "__main__":
    download_spacy_model()
