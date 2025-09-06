# EmailGist Training Pipeline

This directory contains the complete training pipeline for EmailGist, implementing both BART summarization and spaCy entity recognition models for business email processing.

## Overview

The training pipeline consists of several key components:

1. **Data Preprocessing** (`data_prep.py`) - Loads and preprocesses the Enron email dataset
2. **BART Summarization Training** (`bart_trainer.py`) - Fine-tunes BART for email summarization
3. **spaCy NER Training** (`spacy_trainer.py`) - Trains custom entity recognition for business emails
4. **Evaluation** (`evaluation.py`) - Comprehensive evaluation metrics for both models
5. **Main Training Script** (`train_models.py`) - Orchestrates the entire pipeline

## Features

### BART Summarization
- **Synthetic Summary Generation**: Uses pre-trained BART to generate initial summaries for training data
- **Fine-tuning**: Customizes BART for email-specific summarization
- **Evaluation**: ROUGE and BLEU metrics for summarization quality

### spaCy Entity Recognition
- **Rule-based Annotation**: Generates synthetic entity annotations using regex patterns
- **Custom Entity Types**: Focuses on business-relevant entities (PERSON, ORG, DATE, TIME, EMAIL, etc.)
- **Training**: Fine-tunes spaCy NER for email-specific entities

### Data Processing
- **Email Cleaning**: Removes headers, signatures, and noise from emails
- **Metadata Extraction**: Extracts email characteristics and features
- **Dataset Splitting**: Proper train/validation/test splits

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

3. Set up Kaggle API (for dataset access):
```bash
# Install kaggle CLI and configure with your API key
pip install kaggle
# Follow instructions at https://github.com/Kaggle/kaggle-api
```

## Usage

### Quick Start

Run the complete training pipeline:

```bash
python training/train_models.py --output-dir ./training_output
```

### Advanced Usage

```bash
# Custom training parameters
python training/train_models.py \
    --output-dir ./my_training_output \
    --bart-model facebook/bart-large-cnn \
    --spacy-model en_core_web_sm \
    --bart-epochs 5 \
    --spacy-iterations 15

# Skip training, only run evaluation
python training/train_models.py --skip-training

# Skip evaluation after training
python training/train_models.py --skip-evaluation
```

### Individual Components

#### Data Preprocessing Only
```python
from training.data_prep import EnronEmailPreProcessor

processor = EnronEmailPreProcessor()
result = processor.run_full_preprocessing()
```

#### BART Training Only
```python
from training.bart_trainer import BartSummarizationTrainer

trainer = BartSummarizationTrainer()
# Prepare your data first, then:
# trainer.train(train_dataset, val_dataset)
```

#### spaCy Training Only
```python
from training.spacy_trainer import SpacyNERTrainer

trainer = SpacyNERTrainer()
# Prepare your data first, then:
# trainer.train(train_data, val_data)
```

## Configuration

### BART Training Parameters
- `num_train_epochs`: Number of training epochs (default: 3)
- `learning_rate`: Learning rate (default: 2e-5)
- `per_device_train_batch_size`: Batch size (default: 4)
- `max_input_length`: Maximum input sequence length (default: 1024)
- `max_target_length`: Maximum target sequence length (default: 150)

### spaCy Training Parameters
- `n_iter`: Number of training iterations (default: 10)
- `dropout`: Dropout rate (default: 0.5)
- `batch_size`: Batch size (default: 4)

### Entity Labels
The spaCy model is trained to recognize these entity types:
- `PERSON`: Person names
- `ORG`: Organization/company names
- `DATE`: Dates and time periods
- `TIME`: Specific times
- `EMAIL`: Email addresses
- `PHONE`: Phone numbers
- `ADDRESS`: Physical addresses
- `PRODUCT`: Products and services
- `AMOUNT`: Monetary amounts
- `MEETING`: Meeting-related terms
- `DEADLINE`: Deadline and urgency terms

## Output Structure

After training, the output directory will contain:

```
training_output/
├── bart_model/                 # Trained BART model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── training_metadata.json
├── spacy_model/                # Trained spaCy model
│   ├── meta.json
│   ├── ner/
│   └── training_metadata.json
├── evaluation/                 # Evaluation results
│   ├── bart_evaluation_*.json
│   ├── spacy_evaluation_*.json
│   └── evaluation_report_*.html
├── logs/                       # Training logs
├── preprocessing_metadata.json # Data preprocessing info
└── pipeline_results.json      # Complete pipeline results
```

## Evaluation Metrics

### BART Summarization
- **ROUGE-1, ROUGE-2, ROUGE-L**: Standard summarization metrics
- **BLEU**: Translation quality metric adapted for summarization
- **Length Statistics**: Average prediction/target lengths and compression ratios

### spaCy NER
- **Precision, Recall, F1**: Overall entity recognition performance
- **Per-Entity Metrics**: Performance breakdown by entity type
- **Support**: Number of examples for each entity type

## Model Usage

### Using Trained BART Model
```python
from training.bart_trainer import BartSummarizationTrainer

trainer = BartSummarizationTrainer()
summary = trainer.generate_summary("Your email text here...")
print(summary)
```

### Using Trained spaCy Model
```python
from training.spacy_trainer import SpacyNERTrainer

trainer = SpacyNERTrainer()
entities = trainer.extract_entities("Your email text here...")
for entity in entities:
    print(f"{entity['text']} -> {entity['label']}")
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training configuration
2. **Dataset Download Fails**: Check Kaggle API credentials and internet connection
3. **spaCy Model Not Found**: Run `python -m spacy download en_core_web_sm`
4. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Performance Tips

1. **GPU Usage**: Training will automatically use GPU if available
2. **Batch Size**: Adjust based on available memory
3. **Data Size**: For large datasets, consider using a subset for initial testing
4. **Memory Management**: Use smaller batch sizes if encountering memory issues

## Customization

### Adding New Entity Types
1. Update the `entity_labels` list in `SpacyNERTrainer`
2. Add corresponding regex patterns in `_extract_entities_rules`
3. Retrain the model

### Custom Email Preprocessing
1. Modify functions in `utils/preprocessing.py`
2. Add new cleaning rules or metadata extraction
3. Update the preprocessing pipeline

### Different Base Models
1. Change `bart_model_name` parameter for different BART variants
2. Change `spacy_model_name` for different spaCy base models
3. Adjust training parameters accordingly

## Contributing

When adding new features:
1. Follow the existing code structure and naming conventions
2. Add comprehensive logging
3. Include error handling
4. Update this README with new functionality
5. Add appropriate tests

## License

This training pipeline is part of the EmailGist project. See the main project LICENSE for details.
