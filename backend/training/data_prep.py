import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnronEmailPreProcessor:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_input_length = 1024
        self.max_target_length = 128

    def load_enron_dataset(self):
        try:
            logger.info("Loading Enron dataset from Kaggle...")
            
            hf_dataset = kagglehub.load_dataset(
                "wcukierski/enron-email-dataset",
                path="",
            )
            logger.info(f"Dataset loaded successfully")
            logger.info(f"Dataset structure: {hf_dataset}")

            return hf_dataset
        except Exception as e:
            logger.error(f"Failed to load Enron dataset with kagglehub: {e}")
            logger.info("Trying alternative loading method...")
            
            
            try:
                from datasets import load_dataset
                
                logger.info("Loading sample dataset for testing...")
                
                sample_emails = [
                    "Subject: Meeting Tomorrow\n\nHi John,\n\nI hope this email finds you well. I wanted to remind you about our meeting tomorrow at 2 PM in the conference room. Please bring the quarterly reports and budget analysis.\n\nBest regards,\nSarah",
                    "Subject: Project Update\n\nDear Team,\n\nI wanted to provide an update on the Q4 project. We are on track to meet our deadline of December 15th. The budget is within the allocated $50,000 range.\n\nPlease let me know if you have any questions.\n\nThanks,\nMike",
                    "Subject: Contract Review\n\nHello,\n\nI need to schedule a call to review the contract terms. Please let me know your availability for next week. The contract amount is $25,000 and needs to be finalized by Friday.\n\nBest,\nLisa"
                ]
                
                sample_data = {
                    'message': sample_emails,
                    'subject': ['Meeting Tomorrow', 'Project Update', 'Contract Review'],
                    'from': ['sarah@company.com', 'mike@company.com', 'lisa@company.com'],
                    'to': ['john@company.com', 'team@company.com', 'client@company.com']
                }
                
                df = pd.DataFrame(sample_data)
                logger.info(f"Created sample dataset with {len(df)} emails")
                return df
                
            except Exception as e2:
                logger.error(f"Failed to create sample dataset: {e2}")
                return None
        
    def process_enron_emails(self, hf_dataset):
        if hasattr(hf_dataset, 'to_pandas'):
            df = hf_dataset.to_pandas()
        elif isinstance(hf_dataset, dict):
            if 'train' in hf_dataset:
                df = hf_dataset['train'].to_pandas()
            else:
                first_key = list(hf_dataset.keys())[0]
                df = hf_dataset[first_key].to_pandas()
        else:
            df = pd.DataFrame(hf_dataset)
        
        logger.info(f"Loaded dataset with {len(df)} emails")
        return df

    def preprocess_emails(self, df):
        """Preprocess the email dataset."""
        import sys
        import os
        
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, backend_dir)
        from utils.preprocessing import preprocess_email_dataset, split_dataset
        
        
        df_processed = preprocess_email_dataset(df, text_column='message')
        
        
        splits = split_dataset(df_processed)
        
        return splits

    def prepare_for_bart_training(self, splits):
        """Prepare data for BART summarization training."""
        from transformers import AutoTokenizer
        
        def tokenize_function(examples):
            
            model_inputs = self.tokenizer(
                examples['cleaned_text'],
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            )
            
            
            
            model_inputs['labels'] = model_inputs['input_ids'].copy()
            
            return model_inputs
        
        
        from datasets import Dataset
        
        train_dataset = Dataset.from_pandas(splits['train'])
        val_dataset = Dataset.from_pandas(splits['val'])
        test_dataset = Dataset.from_pandas(splits['test'])
        
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def prepare_for_spacy_training(self, splits):
        """Prepare data for spaCy NER training."""
        
        
        return splits

    def run_full_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting full preprocessing pipeline...")
        
        
        hf_dataset = self.load_enron_dataset()
        if hf_dataset is None:
            logger.error("Failed to load dataset")
            return None
        
        
        df = self.process_enron_emails(hf_dataset)
        if df is None or len(df) == 0:
            logger.error("Failed to process emails")
            return None
        
        
        splits = self.preprocess_emails(df)
        
        
        bart_data = self.prepare_for_bart_training(splits)
        
        
        spacy_data = self.prepare_for_spacy_training(splits)
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return {
            'bart_data': bart_data,
            'spacy_data': spacy_data,
            'raw_splits': splits
        }

if __name__ == "__main__":
    
    processor = EnronEmailPreProcessor()
    result = processor.run_full_preprocessing()
    
    if result:
        print("Preprocessing completed successfully!")
        print(f"BART training data: {len(result['bart_data']['train'])} train samples")
        print(f"spaCy training data: {len(result['spacy_data']['train'])} train samples")
    else:
        print("Preprocessing failed!")