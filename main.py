"""
Main script for Ants vs Bees Classification Project
"""

from src.models import train

def main():
    print("🐜🐝 Ants vs Bees Classification Project")
    print("="*50)
    train.main()   # فقط نستدعي main من train.py

if __name__ == "__main__":
    main()
