import os
import sys
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    from rouge_score import rouge_scorer
    from evaluate import load
    import torch
    from transformers import BartForConditionalGeneration, BartTokenizer
    import spacy
except ImportError as e:
    logging.warning(f"Some evaluation libraries not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    
    
    def __init__(self, output_dir: str = "./evaluation_results"):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"ModelEvaluator initialized with output directory: {output_dir}")

    def evaluate_bart_model(self, 
                           model_path: str,
                           test_data: List[Dict[str, str]],
                           batch_size: int = 8) -> Dict[str, Any]:
        
        logger.info("Evaluating BART summarization model...")
        
        
        model = BartForConditionalGeneration.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        
        predictions = []
        targets = []
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i + batch_size]
            batch_inputs = [item['input'] for item in batch]
            batch_targets = [item['target'] for item in batch]
            
            
            inputs = tokenizer(
                batch_inputs,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            ).to(device)
            
            
            with torch.no_grad():
                summary_ids = model.generate(
                    inputs['input_ids'],
                    max_length=150,
                    min_length=50,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            
            batch_predictions = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)
        
        
        rouge_scores = self._calculate_rouge_scores(predictions, targets)
        
        
        bleu_scores = self._calculate_bleu_scores(predictions, targets)
        
        
        other_metrics = self._calculate_summarization_metrics(predictions, targets)
        
        
        evaluation_results = {
            'rouge_scores': rouge_scores,
            'bleu_scores': bleu_scores,
            'other_metrics': other_metrics,
            'num_samples': len(test_data),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        
        self._save_evaluation_results('bart_evaluation', evaluation_results)
        
        logger.info("BART evaluation completed")
        return evaluation_results

    def evaluate_spacy_model(self, 
                            model_path: str,
                            test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        logger.info("Evaluating spaCy NER model...")
        
        
        nlp = spacy.load(model_path)
        
        
        predictions = []
        targets = []
        
        for item in test_data:
            text = item['text']
            true_entities = item['entities']
            
            
            doc = nlp(text)
            pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            
            predictions.append(pred_entities)
            targets.append(true_entities)
        
        
        ner_metrics = self._calculate_ner_metrics(predictions, targets)
        
        
        per_entity_metrics = self._calculate_per_entity_metrics(predictions, targets)
        
        
        evaluation_results = {
            'ner_metrics': ner_metrics,
            'per_entity_metrics': per_entity_metrics,
            'num_samples': len(test_data),
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        
        self._save_evaluation_results('spacy_evaluation', evaluation_results)
        
        logger.info("spaCy evaluation completed")
        return evaluation_results

    def _calculate_rouge_scores(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores for summarization."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, target in zip(predictions, targets):
            scores = self.rouge_scorer.score(target, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1_f1': np.mean(rouge1_scores),
            'rouge2_f1': np.mean(rouge2_scores),
            'rougeL_f1': np.mean(rougeL_scores),
            'rouge1_std': np.std(rouge1_scores),
            'rouge2_std': np.std(rouge2_scores),
            'rougeL_std': np.std(rougeL_scores)
        }

    def _calculate_bleu_scores(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate BLEU scores for summarization."""
        try:
            bleu = load("bleu")
            results = bleu.compute(predictions=predictions, references=[[t] for t in targets])
            return {
                'bleu_score': results['bleu'],
                'precisions': results['precisions'],
                'brevity_penalty': results['brevity_penalty'],
                'length_ratio': results['length_ratio']
            }
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            return {'bleu_score': 0.0}

    def _calculate_summarization_metrics(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate additional summarization metrics."""
        
        pred_lengths = [len(pred.split()) for pred in predictions]
        target_lengths = [len(target.split()) for target in targets]
        
        
        compression_ratios = [len(pred.split()) / len(target.split()) if len(target.split()) > 0 else 0 
                            for pred, target in zip(predictions, targets)]
        
        return {
            'avg_prediction_length': np.mean(pred_lengths),
            'avg_target_length': np.mean(target_lengths),
            'avg_compression_ratio': np.mean(compression_ratios),
            'length_std': np.std(pred_lengths),
            'compression_std': np.std(compression_ratios)
        }

    def _calculate_ner_metrics(self, predictions: List[List], targets: List[List]) -> Dict[str, float]:
        """Calculate overall NER metrics."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred_entities, true_entities in zip(predictions, targets):
            pred_set = set(pred_entities)
            true_set = set(true_entities)
            
            true_positives += len(pred_set & true_set)
            false_positives += len(pred_set - true_set)
            false_negatives += len(true_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

    def _calculate_per_entity_metrics(self, predictions: List[List], targets: List[List]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each entity type."""
        
        all_labels = set()
        for entities in targets:
            for entity in entities:
                all_labels.add(entity[2])
        
        per_entity_metrics = {}
        
        for label in all_labels:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for pred_entities, true_entities in zip(predictions, targets):
                pred_label = set([ent for ent in pred_entities if ent[2] == label])
                true_label = set([ent for ent in true_entities if ent[2] == label])
                
                true_positives += len(pred_label & true_label)
                false_positives += len(pred_label - true_label)
                false_negatives += len(true_label - pred_label)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_entity_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': true_positives + false_negatives
            }
        
        return per_entity_metrics

    def _save_evaluation_results(self, model_type: str, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_type}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {filepath}")

    def create_evaluation_report(self, 
                               bart_results: Optional[Dict[str, Any]] = None,
                               spacy_results: Optional[Dict[str, Any]] = None) -> str:
        
        report_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EmailGist Model Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                .metric-label { font-size: 14px; color: #7f8c8d; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>EmailGist Model Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """
        
        if bart_results:
            report_html += self._create_bart_report_section(bart_results)
        
        if spacy_results:
            report_html += self._create_spacy_report_section(spacy_results)
        
        report_html += """
        </body>
        </html>
        """
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"evaluation_report_{timestamp}.html")
        
        with open(report_path, 'w') as f:
            f.write(report_html.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_html

    def _create_bart_report_section(self, results: Dict[str, Any]) -> str:
        """Create BART evaluation report section."""
        rouge_scores = results.get('rouge_scores', {})
        bleu_scores = results.get('bleu_scores', {})
        other_metrics = results.get('other_metrics', {})
        
        section = f"""
        <div class="section">
            <h2>BART Summarization Model Evaluation</h2>
            <p>Model Path: {results.get('model_path', 'N/A')}</p>
            <p>Number of Test Samples: {results.get('num_samples', 0)}</p>
            
            <h3>ROUGE Scores</h3>
            <div class="metric">
                <div class="metric-value">{rouge_scores.get('rouge1_f1', 0):.4f}</div>
                <div class="metric-label">ROUGE-1 F1</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rouge_scores.get('rouge2_f1', 0):.4f}</div>
                <div class="metric-label">ROUGE-2 F1</div>
            </div>
            <div class="metric">
                <div class="metric-value">{rouge_scores.get('rougeL_f1', 0):.4f}</div>
                <div class="metric-label">ROUGE-L F1</div>
            </div>
            
            <h3>BLEU Score</h3>
            <div class="metric">
                <div class="metric-value">{bleu_scores.get('bleu_score', 0):.4f}</div>
                <div class="metric-label">BLEU</div>
            </div>
            
            <h3>Length Statistics</h3>
            <div class="metric">
                <div class="metric-value">{other_metrics.get('avg_prediction_length', 0):.1f}</div>
                <div class="metric-label">Avg Prediction Length</div>
            </div>
            <div class="metric">
                <div class="metric-value">{other_metrics.get('avg_target_length', 0):.1f}</div>
                <div class="metric-label">Avg Target Length</div>
            </div>
            <div class="metric">
                <div class="metric-value">{other_metrics.get('avg_compression_ratio', 0):.2f}</div>
                <div class="metric-label">Avg Compression Ratio</div>
            </div>
        </div>
        """
        
        return section

    def _create_spacy_report_section(self, results: Dict[str, Any]) -> str:
        """Create spaCy evaluation report section."""
        ner_metrics = results.get('ner_metrics', {})
        per_entity_metrics = results.get('per_entity_metrics', {})
        
        section = f"""
        <div class="section">
            <h2>spaCy NER Model Evaluation</h2>
            <p>Model Path: {results.get('model_path', 'N/A')}</p>
            <p>Number of Test Samples: {results.get('num_samples', 0)}</p>
            
            <h3>Overall NER Metrics</h3>
            <div class="metric">
                <div class="metric-value">{ner_metrics.get('precision', 0):.4f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">{ner_metrics.get('recall', 0):.4f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">{ner_metrics.get('f1_score', 0):.4f}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            
            <h3>Per-Entity Metrics</h3>
            <table>
                <tr>
                    <th>Entity Type</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Support</th>
                </tr>
        """
        
        for entity_type, metrics in per_entity_metrics.items():
            section += f"""
                <tr>
                    <td>{entity_type}</td>
                    <td>{metrics.get('precision', 0):.4f}</td>
                    <td>{metrics.get('recall', 0):.4f}</td>
                    <td>{metrics.get('f1_score', 0):.4f}</td>
                    <td>{metrics.get('support', 0)}</td>
                </tr>
            """
        
        section += """
            </table>
        </div>
        """
        
        return section

def main():
    """Example usage of ModelEvaluator."""
    evaluator = ModelEvaluator()
    
    
    print("ModelEvaluator initialized successfully!")
    print("Use this class to evaluate your trained BART and spaCy models.")

if __name__ == "__main__":
    main()