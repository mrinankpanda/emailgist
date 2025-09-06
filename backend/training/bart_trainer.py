import os
import sys
import logging
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import json


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BartSummarizationTrainer:
    def __init__(self, 
                 model_name: str = "facebook/bart-large-cnn",
                 max_input_length: int = 1024,
                 max_target_length: int = 150,
                 min_target_length: int = 50,
                 output_dir: str = "./models/bart_email_summarizer"):
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.min_target_length = min_target_length
        self.output_dir = output_dir
        
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized BART trainer with model: {model_name}")

    def generate_synthetic_summaries(self, 
                                   emails: List[str], 
                                   batch_size: int = 8,
                                   num_beams: int = 4,
                                   length_penalty: float = 2.0) -> List[str]:
        
        logger.info(f"Generating synthetic summaries for {len(emails)} emails...")
        
        summaries = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        
        for i in tqdm(range(0, len(emails), batch_size), desc="Generating summaries"):
            batch_emails = emails[i:i + batch_size]
            
            
            inputs = self.tokenizer(
                batch_emails,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                padding=True
            ).to(device)
            
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs['input_ids'],
                    max_length=self.max_target_length,
                    min_length=self.min_target_length,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False
                )
            
            
            batch_summaries = self.tokenizer.batch_decode(
                summary_ids, 
                skip_special_tokens=True
            )
            
            summaries.extend(batch_summaries)
        
        logger.info(f"Generated {len(summaries)} synthetic summaries")
        return summaries

    def prepare_training_data(self, 
                            train_emails: List[str], 
                            val_emails: List[str],
                            train_summaries: Optional[List[str]] = None,
                            val_summaries: Optional[List[str]] = None) -> Tuple[Dataset, Dataset]:
        
        logger.info("Preparing training data...")
        
        
        if train_summaries is None:
            logger.info("Generating synthetic summaries for training data...")
            train_summaries = self.generate_synthetic_summaries(train_emails)
        
        if val_summaries is None:
            logger.info("Generating synthetic summaries for validation data...")
            val_summaries = self.generate_synthetic_summaries(val_emails)
        
        
        train_data = {
            'input_text': train_emails,
            'target_text': train_summaries
        }
        
        val_data = {
            'input_text': val_emails,
            'target_text': val_summaries
        }
        
        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        
        
        def tokenize_function(examples):
            
            model_inputs = self.tokenizer(
                examples['input_text'],
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors=None
            )
            
            
            labels = self.tokenizer(
                text_target=examples['target_text'],
                max_length=self.max_target_length,
                truncation=True,
                padding=True,
                return_tensors=None
            )
            
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        
        
        train_dataset = train_dataset.remove_columns(['input_text', 'target_text'])
        val_dataset = val_dataset.remove_columns(['input_text', 'target_text'])
        
        
        logger.info(f"Prepared training data: {len(train_dataset)} samples")
        logger.info(f"Prepared validation data: {len(val_dataset)} samples")
        logger.info(f"Training dataset features: {train_dataset.features}")
        if len(train_dataset) > 0:
            logger.info(f"Sample training data: {train_dataset[0]}")
        
        return train_dataset, val_dataset

    def train(self, 
              train_dataset: Dataset, 
              val_dataset: Dataset,
              num_train_epochs: int = 3,
              learning_rate: float = 2e-5,
              per_device_train_batch_size: int = 4,
              per_device_eval_batch_size: int = 4,
              warmup_steps: int = 500,
              weight_decay: float = 0.01,
              logging_steps: int = 100,
              eval_steps: int = 500,
              save_steps: int = 1000,
              early_stopping_patience: int = 3) -> Trainer:
        
        logger.info("Starting BART training...")
        
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  
            seed=42,
            fp16=torch.cuda.is_available(),  
            dataloader_drop_last=False,  
            remove_unused_columns=False,  
        )
        
        
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience
        )
        
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping],
        )
        
        
        trainer.train()
        
        
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
        return trainer

    def evaluate_model(self, 
                      test_dataset: Dataset, 
                      trainer: Trainer) -> Dict[str, float]:
        
        logger.info("Evaluating model on test dataset...")
        
        
        eval_results = trainer.evaluate(test_dataset)
        
        logger.info("Evaluation results:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results

    def generate_summary(self, email_text: str) -> str:
        
        
        if not hasattr(self, 'trained_model'):
            self.trained_model = BartForConditionalGeneration.from_pretrained(self.output_dir)
            self.trained_tokenizer = BartTokenizer.from_pretrained(self.output_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model.to(device)
        
        
        inputs = self.trained_tokenizer(
            email_text,
            return_tensors="pt",
            max_length=self.max_input_length,
            truncation=True
        ).to(device)
        
        
        with torch.no_grad():
            summary_ids = self.trained_model.generate(
                inputs['input_ids'],
                max_length=self.max_target_length,
                min_length=self.min_target_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        
        summary = self.trained_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

    def save_training_metadata(self, 
                             train_dataset_size: int,
                             val_dataset_size: int,
                             test_dataset_size: int,
                             training_args: Dict) -> None:
        
        metadata = {
            'model_name': self.model_name,
            'max_input_length': self.max_input_length,
            'max_target_length': self.max_target_length,
            'min_target_length': self.min_target_length,
            'dataset_sizes': {
                'train': train_dataset_size,
                'val': val_dataset_size,
                'test': test_dataset_size
            },
            'training_args': training_args,
            'output_dir': self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")

def main():
    
    
    
    trainer = BartSummarizationTrainer()
    
    
    train_emails = ["Sample email 1", "Sample email 2"]
    val_emails = ["Sample validation email"]
    
    
    train_dataset, val_dataset = trainer.prepare_training_data(
        train_emails, val_emails
    )
    
    
    trained_trainer = trainer.train(train_dataset, val_dataset)
    
    print("BART training completed!")

if __name__ == "__main__":
    main()