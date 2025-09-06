import os
import sys
import logging
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
import pandas as pd
from tqdm import tqdm
import re


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpacyNERTrainer:
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 output_dir: str = "./models/spacy_email_ner",
                 entity_labels: List[str] = None):
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.entity_labels = entity_labels or [
            "PERSON", "ORG", "DATE", "TIME", "EMAIL", "PHONE", 
            "ADDRESS", "PRODUCT", "AMOUNT", "MEETING", "DEADLINE"
        ]
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        self._initialize_model()
        
        logger.info(f"Initialized spaCy NER trainer with model: {model_name}")
        logger.info(f"Entity labels: {self.entity_labels}")

    def _initialize_model(self):
        """Initialize the spaCy model."""
        try:
            if self.model_name == "blank":
                self.nlp = spacy.blank("en")
            else:
                self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.warning(f"Model {self.model_name} not found, using blank model")
            self.nlp = spacy.blank("en")
        
        
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        
        
        for label in self.entity_labels:
            ner.add_label(label)
        
        logger.info("spaCy model initialized successfully")

    def generate_synthetic_annotations(self, 
                                     emails: List[str], 
                                     annotation_ratio: float = 0.3) -> List[Tuple[str, Dict]]:
        
        logger.info(f"Generating synthetic annotations for {len(emails)} emails...")
        
        
        num_to_annotate = int(len(emails) * annotation_ratio)
        emails_to_annotate = random.sample(emails, min(num_to_annotate, len(emails)))
        
        annotated_data = []
        
        for email in tqdm(emails_to_annotate, desc="Generating annotations"):
            annotations = self._extract_entities_rules(email)
            
            annotated_data.append((email, annotations))
        
        logger.info(f"Generated {len(annotated_data)} annotated examples")
        return annotated_data

    def _extract_entities_rules(self, text: str) -> Dict[str, List]:
        
        entities = []
        
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append((match.start(), match.end(), "EMAIL"))
        
        
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append((match.start(), match.end(), "PHONE"))
        
        
        date_patterns = [
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append((match.start(), match.end(), "DATE"))
        
        
        time_pattern = r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b'
        for match in re.finditer(time_pattern, text):
            entities.append((match.start(), match.end(), "TIME"))
        
        
        business_terms = {
            "MEETING": [r'\bmeeting\b', r'\bconference\b', r'\bcall\b', r'\bconference room\b'],
            "DEADLINE": [r'\bdeadline\b', r'\bdue\b', r'\burgent\b', r'\basap\b', r'\bDecember 15th\b', r'\bFriday\b'],
            "PRODUCT": [r'\bproduct\b', r'\bservice\b', r'\bproposal\b', r'\bcontract\b', r'\bproject\b', r'\breports\b'],
            "AMOUNT": [r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b', r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)\b', r'\$50,000\b', r'\$25,000\b']
        }
        
        for label, patterns in business_terms.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append((match.start(), match.end(), label))
        
        
        company_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Co\.)\b'
        for match in re.finditer(company_pattern, text):
            entities.append((match.start(), match.end(), "ORG"))
        
        
        person_pattern = r'\b(?:Mr|Ms|Mrs|Dr|Prof)\.?\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        for match in re.finditer(person_pattern, text):
            entities.append((match.start(), match.end(), "PERSON"))
        
        
        common_names = [r'\bJohn\b', r'\bSarah\b', r'\bMike\b', r'\bLisa\b']
        for pattern in common_names:
            for match in re.finditer(pattern, text):
                entities.append((match.start(), match.end(), "PERSON"))
        
        return {"entities": entities}

    def prepare_training_data(self, 
                            train_emails: List[str], 
                            val_emails: List[str],
                            train_annotations: Optional[List[Tuple[str, Dict]]] = None,
                            val_annotations: Optional[List[Tuple[str, Dict]]] = None) -> Tuple[List, List]:
        
        logger.info("Preparing spaCy training data...")
        
        
        if train_annotations is None:
            logger.info("Generating synthetic annotations for training data...")
            train_annotations = self.generate_synthetic_annotations(train_emails, annotation_ratio=0.4)
        
        if val_annotations is None:
            logger.info("Generating synthetic annotations for validation data...")
            val_annotations = self.generate_synthetic_annotations(val_emails, annotation_ratio=0.3)
        
        logger.info(f"Prepared training data: {len(train_annotations)} samples")
        logger.info(f"Prepared validation data: {len(val_annotations)} samples")
        
        return train_annotations, val_annotations

    def train(self, 
              train_data: List[Tuple[str, Dict]], 
              val_data: List[Tuple[str, Dict]],
              n_iter: int = 10,
              dropout: float = 0.5,
              batch_size: int = 4) -> spacy.Language:
        
        logger.info("Starting spaCy NER training...")
        
        
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        
        with self.nlp.disable_pipes(*other_pipes):
            
            optimizer = self.nlp.begin_training()
            
            
            for iteration in range(n_iter):
                logger.info(f"Training iteration {iteration + 1}/{n_iter}")
                
                
                random.shuffle(train_data)
                
                
                batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
                
                losses = {}
                
                
                for batch in tqdm(batches, desc=f"Iteration {iteration + 1}"):
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    
                    self.nlp.update(examples, sgd=optimizer, drop=dropout, losses=losses)
                
                logger.info(f"Losses: {losses}")
                
                
                if val_data:
                    val_accuracy = self._evaluate_on_validation(val_data)
                    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        
        self.nlp.to_disk(self.output_dir)
        
        logger.info(f"Training completed. Model saved to {self.output_dir}")
        
        return self.nlp

    def _evaluate_on_validation(self, val_data: List[Tuple[str, Dict]]) -> float:
        
        correct = 0
        total = 0
        
        for text, annotations in val_data:
            doc = self.nlp(text)
            predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            true_entities = annotations["entities"]
            
            
            for true_entity in true_entities:
                total += 1
                if true_entity in predicted_entities:
                    correct += 1
        
        return correct / total if total > 0 else 0.0

    def evaluate_model(self, test_data: List[Tuple[str, Dict]]) -> Dict[str, float]:
        
        logger.info("Evaluating spaCy NER model on test data...")
        
        
        trained_nlp = spacy.load(self.output_dir)
        
        
        true_positives = {label: 0 for label in self.entity_labels}
        false_positives = {label: 0 for label in self.entity_labels}
        false_negatives = {label: 0 for label in self.entity_labels}
        
        for text, annotations in tqdm(test_data, desc="Evaluating"):
            doc = trained_nlp(text)
            predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            true_entities = annotations["entities"]
            
            
            predicted_set = set(predicted_entities)
            true_set = set(true_entities)
            
            
            for label in self.entity_labels:
                pred_label = set([ent for ent in predicted_entities if ent[2] == label])
                true_label = set([ent for ent in true_entities if ent[2] == label])
                
                true_positives[label] += len(pred_label & true_label)
                false_positives[label] += len(pred_label - true_label)
                false_negatives[label] += len(true_label - pred_label)
        
        
        metrics = {}
        for label in self.entity_labels:
            tp = true_positives[label]
            fp = false_positives[label]
            fn = false_negatives[label]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f"{label}_precision"] = precision
            metrics[f"{label}_recall"] = recall
            metrics[f"{label}_f1"] = f1
        
        
        total_tp = sum(true_positives.values())
        total_fp = sum(false_positives.values())
        total_fn = sum(false_negatives.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics["overall_precision"] = overall_precision
        metrics["overall_recall"] = overall_recall
        metrics["overall_f1"] = overall_f1
        
        logger.info("Evaluation results:")
        logger.info(f"Overall Precision: {overall_precision:.4f}")
        logger.info(f"Overall Recall: {overall_recall:.4f}")
        logger.info(f"Overall F1: {overall_f1:.4f}")
        
        return metrics

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        
        
        if not hasattr(self, 'trained_nlp'):
            self.trained_nlp = spacy.load(self.output_dir)
        
        doc = self.trained_nlp(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': ent._.prob if hasattr(ent._, 'prob') else None
            })
        
        return entities

    def save_training_metadata(self, 
                             train_data_size: int,
                             val_data_size: int,
                             test_data_size: int,
                             training_params: Dict) -> None:
        
        metadata = {
            'model_name': self.model_name,
            'entity_labels': self.entity_labels,
            'dataset_sizes': {
                'train': train_data_size,
                'val': val_data_size,
                'test': test_data_size
            },
            'training_params': training_params,
            'output_dir': self.output_dir
        }
        
        metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")

def main():
    
    
    
    trainer = SpacyNERTrainer()
    
    
    train_emails = ["Please send the proposal to John Doe at Acme Inc. by Friday."]
    val_emails = ["The meeting is scheduled for September 10th at 3 PM."]
    
    
    train_data, val_data = trainer.prepare_training_data(train_emails, val_emails)
    
    
    trained_model = trainer.train(train_data, val_data)
    
    print("spaCy NER training completed!")

if __name__ == "__main__":
    main()