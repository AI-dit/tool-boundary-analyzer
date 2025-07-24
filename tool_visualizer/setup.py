"""
Post-installation setup for Tool Boundary Analyzer
Automatically downloads required spaCy models
"""

import subprocess
import sys
import os


def download_spacy_model():
    """Download the English spaCy model"""
    try:
        print("🔽 Downloading spaCy English model...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✅ spaCy model downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to download spaCy model: {e}")
        print("You can manually download it with: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        print(f"⚠️  Unexpected error downloading spaCy model: {e}")
        return False


def post_install():
    """Run post-installation tasks"""
    print("🚀 Running Tool Boundary Analyzer post-installation setup...")
    
    # Download spaCy model
    download_spacy_model()
    
    print("✅ Post-installation setup complete!")


if __name__ == "__main__":
    post_install()
