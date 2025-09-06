import os
import sys
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.data_prep import EnronEmailPreProcessor
from training.bart_trainer import BartSummarizationTrainer
from training.spacy_trainer import SpacyNERTrainer
from training.evaluation import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailGistTrainingPipeline:
    
    
    def __init__(self, 
                 output_dir: str = "./training_output",
                 bart_model_name: str = "facebook/bart-large-cnn",
                 spacy_model_name: str = "en_core_web_sm"):
        
        self.output_dir = output_dir
        self.bart_model_name = bart_model_name
        self.spacy_model_name = spacy_model_name
        
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/bart_model", exist_ok=True)
        os.makedirs(f"{output_dir}/spacy_model", exist_ok=True)
        os.makedirs(f"{output_dir}/evaluation", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        
        self.data_processor = EnronEmailPreProcessor(bart_model_name)
        self.bart_trainer = BartSummarizationTrainer(
            model_name=bart_model_name,
            output_dir=f"{output_dir}/bart_model"
        )
        self.spacy_trainer = SpacyNERTrainer(
            model_name=spacy_model_name,
            output_dir=f"{output_dir}/spacy_model"
        )
        self.evaluator = ModelEvaluator(
            output_dir=f"{output_dir}/evaluation"
        )
        
        logger.info(f"EmailGistTrainingPipeline initialized with output directory: {output_dir}")

    def run_data_preprocessing(self) -> Optional[Dict[str, Any]]:
        
        logger.info("Starting data preprocessing...")
        
        try:
            
            preprocessed_data = self.data_processor.run_full_preprocessing()
            
            if preprocessed_data is None:
                logger.error("Data preprocessing failed")
                return None
            
            
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'bart_model_name': self.bart_model_name,
                'spacy_model_name': self.spacy_model_name,
                'dataset_info': {
                    'bart_train_size': len(preprocessed_data['bart_data']['train']),
                    'bart_val_size': len(preprocessed_data['bart_data']['val']),
                    'bart_test_size': len(preprocessed_data['bart_data']['test']),
                    'spacy_train_size': len(preprocessed_data['spacy_data']['train']),
                    'spacy_val_size': len(preprocessed_data['spacy_data']['val']),
                    'spacy_test_size': len(preprocessed_data['spacy_data']['test'])
                }
            }
            
            metadata_path = os.path.join(self.output_dir, 'preprocessing_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Data preprocessing completed successfully")
            return preprocessed_data
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            return None

    def run_bart_training(self, preprocessed_data: Dict[str, Any], 
                         training_config: Optional[Dict[str, Any]] = None) -> bool:
        
        logger.info("Starting BART summarization training...")
        
        try:
            
            bart_data = preprocessed_data['bart_data']
            raw_splits = preprocessed_data['raw_splits']
            
            
            train_emails = raw_splits['train']['cleaned_text'].tolist()
            val_emails = raw_splits['val']['cleaned_text'].tolist()
            
            
            train_dataset, val_dataset = self.bart_trainer.prepare_training_data(
                train_emails, val_emails
            )
            
            
            if training_config is None:
                training_config = {
                    'num_train_epochs': 3,
                    'learning_rate': 2e-5,
                    'per_device_train_batch_size': 4,
                    'per_device_eval_batch_size': 4,
                    'warmup_steps': 500,
                    'weight_decay': 0.01,
                    'logging_steps': 100,
                    'eval_steps': 500,
                    'save_steps': 1000,
                    'early_stopping_patience': 3
                }
            
            
            trainer = self.bart_trainer.train(
                train_dataset, 
                val_dataset,
                **training_config
            )
            
            
            self.bart_trainer.save_training_metadata(
                len(train_dataset),
                len(val_dataset),
                len(bart_data['test']),
                training_config
            )
            
            logger.info("BART training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"BART training failed: {e}")
            return False

    def run_spacy_training(self, preprocessed_data: Dict[str, Any],
                          training_config: Optional[Dict[str, Any]] = None) -> bool:
        
        logger.info("Starting spaCy NER training...")
        
        try:
            
            raw_splits = preprocessed_data['raw_splits']
            
            
            train_emails = raw_splits['train']['cleaned_text'].tolist()
            val_emails = raw_splits['val']['cleaned_text'].tolist()
            
            
            train_data, val_data = self.spacy_trainer.prepare_training_data(
                train_emails, val_emails
            )
            
            
            if training_config is None:
                training_config = {
                    'n_iter': 10,
                    'dropout': 0.5,
                    'batch_size': 4
                }
            
            
            trained_model = self.spacy_trainer.train(
                train_data,
                val_data,
                **training_config
            )
            
            
            self.spacy_trainer.save_training_metadata(
                len(train_data),
                len(val_data),
                len(raw_splits['test']),
                training_config
            )
            
            logger.info("spaCy training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"spaCy training failed: {e}")
            return False

    def run_evaluation(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        
        logger.info("Starting model evaluation...")
        
        evaluation_results = {}
        
        try:
            
            raw_splits = preprocessed_data['raw_splits']
            test_emails = raw_splits['test']['cleaned_text'].tolist()
            
            
            test_summaries = self.bart_trainer.generate_synthetic_summaries(test_emails)
            
            
            bart_test_data = [
                {'input': email, 'target': summary}
                for email, summary in zip(test_emails, test_summaries)
            ]
            
            
            bart_results = self.evaluator.evaluate_bart_model(
                self.bart_trainer.output_dir,
                bart_test_data
            )
            evaluation_results['bart'] = bart_results
            
            
            spacy_test_data = []
            for email in test_emails[:100]:  
                annotations = self.spacy_trainer._extract_entities_rules(email)
                if annotations["entities"]:
                    spacy_test_data.append({
                        'text': email,
                        'entities': annotations["entities"]
                    })
            
            
            if spacy_test_data:
                spacy_results = self.evaluator.evaluate_spacy_model(
                    self.spacy_trainer.output_dir,
                    spacy_test_data
                )
                evaluation_results['spacy'] = spacy_results
            
            
            report = self.evaluator.create_evaluation_report(
                bart_results=bart_results,
                spacy_results=evaluation_results.get('spacy')
            )
            
            logger.info("Model evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
        
        return evaluation_results

    def run_full_pipeline(self, 
                         bart_config: Optional[Dict[str, Any]] = None,
                         spacy_config: Optional[Dict[str, Any]] = None,
                         skip_training: bool = False,
                         skip_evaluation: bool = False) -> bool:
        
        logger.info("Starting EmailGist training pipeline...")
        
        pipeline_start_time = datetime.now()
        
        try:
            
            if not skip_training:
                preprocessed_data = self.run_data_preprocessing()
                if preprocessed_data is None:
                    logger.error("Pipeline failed at data preprocessing step")
                    return False
            else:
                
                logger.info("Skipping training, loading existing data...")
                preprocessed_data = self._load_existing_data()
                if preprocessed_data is None:
                    logger.error("No existing data found for evaluation")
                    return False
            
            
            if not skip_training:
                bart_success = self.run_bart_training(preprocessed_data, bart_config)
                if not bart_success:
                    logger.error("Pipeline failed at BART training step")
                    return False
            
            
            if not skip_training:
                spacy_success = self.run_spacy_training(preprocessed_data, spacy_config)
                if not spacy_success:
                    logger.error("Pipeline failed at spaCy training step")
                    return False
            
            
            if not skip_evaluation:
                evaluation_results = self.run_evaluation(preprocessed_data)
                
                
                pipeline_results = {
                    'pipeline_start_time': pipeline_start_time.isoformat(),
                    'pipeline_end_time': datetime.now().isoformat(),
                    'bart_model_path': self.bart_trainer.output_dir,
                    'spacy_model_path': self.spacy_trainer.output_dir,
                    'evaluation_results': evaluation_results
                }
                
                results_path = os.path.join(self.output_dir, 'pipeline_results.json')
                with open(results_path, 'w') as f:
                    json.dump(pipeline_results, f, indent=2)
                
                logger.info(f"Pipeline results saved to {results_path}")
            
            pipeline_end_time = datetime.now()
            duration = pipeline_end_time - pipeline_start_time
            
            logger.info(f"EmailGist training pipeline completed successfully in {duration}")
            return True
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return False

    def _load_existing_data(self) -> Optional[Dict[str, Any]]:
        """Load existing preprocessed data for evaluation."""
        
        logger.warning("Loading existing data not implemented - using dummy data")
        return None

def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Train EmailGist models')
    parser.add_argument('--output-dir', type=str, default='./training_output',
                       help='Output directory for training results')
    parser.add_argument('--bart-model', type=str, default='facebook/bart-large-cnn',
                       help='Pre-trained BART model name')
    parser.add_argument('--spacy-model', type=str, default='en_core_web_sm',
                       help='Base spaCy model name')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run evaluation')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation after training')
    parser.add_argument('--bart-epochs', type=int, default=3,
                       help='Number of BART training epochs')
    parser.add_argument('--spacy-iterations', type=int, default=10,
                       help='Number of spaCy training iterations')
    
    args = parser.parse_args()
    
    
    bart_config = {
        'num_train_epochs': args.bart_epochs,
        'learning_rate': 2e-5,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'logging_steps': 100,
        'eval_steps': 500,
        'save_steps': 1000,
        'early_stopping_patience': 3
    }
    
    spacy_config = {
        'n_iter': args.spacy_iterations,
        'dropout': 0.5,
        'batch_size': 4
    }
    
    
    pipeline = EmailGistTrainingPipeline(
        output_dir=args.output_dir,
        bart_model_name=args.bart_model,
        spacy_model_name=args.spacy_model
    )
    
    success = pipeline.run_full_pipeline(
        bart_config=bart_config,
        spacy_config=spacy_config,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation
    )
    
    if success:
        print("Training pipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()