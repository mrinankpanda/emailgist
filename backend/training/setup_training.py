import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing Python requirements"
    )

def download_spacy_model():
    
    return run_command(
        "python -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )

def setup_kaggle():
    
    logger.info("Setting up Kaggle API...")
    
    
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        logger.info("✓ Kaggle CLI is already installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("Installing Kaggle CLI...")
        if not run_command("pip install kaggle", "Installing Kaggle CLI"):
            logger.warning("⚠ Kaggle CLI installation failed - this is optional for basic setup")
            return True  
    
    
kaggle_dir = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        logger.info("✓ Kaggle API key found")
        return True
    else:
        logger.warning("⚠ Kaggle API key not found")
        logger.info("To set up Kaggle API (optional for basic setup):")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Click 'Create New API Token'")
        logger.info("3. Download kaggle.json")
        logger.info(f"4. Place it in {kaggle_dir}")
        logger.info("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        logger.info("Note: Kaggle API is only needed for downloading the Enron dataset")
        return True  

def create_directories():
    
    logger.info("Creating training directories...")
    
directories = [
        "training_output",
        "training_output/bart_model",
        "training_output/spacy_model", 
        "training_output/evaluation",
        "training_output/logs",
        "models",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True

def test_imports():
    
    logger.info("Testing imports...")
    
    required_modules = [
        "transformers",
        "torch",
        "spacy",
        "datasets",
        "evaluate",
        "rouge_score",
        "sklearn",
        "kagglehub",
        "pandas",
        "numpy",
        "tqdm"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✓ {module}")
        except ImportError as e:
            logger.error(f"✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"Failed to import: {', '.join(failed_imports)}")
        return False
    
    logger.info("✓ All required modules imported successfully")
    return True

def test_spacy_model():
    
    logger.info("Testing spaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence.")
        logger.info("✓ spaCy model is working")
        return True
    except Exception as e:
        logger.error(f"✗ spaCy model test failed: {e}")
        return False

def main():
    
    logger.info("Starting EmailGist training setup...")
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Install Requirements", install_requirements),
        ("Download spaCy Model", download_spacy_model),
        ("Setup Kaggle API", setup_kaggle),
        ("Create Directories", create_directories),
        ("Test Imports", test_imports),
        ("Test spaCy Model", test_spacy_model)
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        logger.info(f"\n--- {step_name} ---")
        if not step_function():
            failed_steps.append(step_name)
    
    logger.info("\n" + "="*50)
    logger.info("SETUP SUMMARY")
    logger.info("="*50)
    
    if failed_steps:
        logger.error(f"✗ Setup completed with {len(failed_steps)} failures:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        logger.error("\nPlease fix the failed steps before running training.")
        return False
    else:
        logger.info("✓ Setup completed successfully!")
        logger.info("\nYou can now run the training pipeline:")
        logger.info("python training/train_models.py --output-dir ./training_output")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)