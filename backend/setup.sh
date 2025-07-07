#!/bin/bash

echo "🚀 Setting up your Python environment..."

# Install Python dependencies
pip install -r requirements.txt

# Download the spaCy model
python -m spacy download en_core_web_sm

echo "✅ Setup complete!"
