import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

def clean_email_text(text: str) -> str:
    
    if not text or not isinstance(text, str):
        return ""
    
    
    text = re.sub(r'^(From|To|Subject|Date|Cc|Bcc|Reply-To):.*$', '', text, flags=re.MULTILINE)
    
    
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)
    
    
    text = re.sub(r'--\s*$.*', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'Best regards?.*', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'Sincerely.*', '', text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r'Thanks?.*', '', text, flags=re.MULTILINE | re.DOTALL)
    
    
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    
    text = text.strip()
    return text

def extract_email_metadata(email_text: str) -> Dict[str, Any]:
    
    metadata = {
        'length': len(email_text),
        'word_count': len(email_text.split()),
        'has_attachments': 'attachment' in email_text.lower(),
        'has_meeting': any(word in email_text.lower() for word in ['meeting', 'call', 'conference']),
        'has_deadline': any(word in email_text.lower() for word in ['deadline', 'due', 'urgent', 'asap']),
        'has_question': '?' in email_text,
        'has_action_items': any(word in email_text.lower() for word in ['please', 'need', 'action', 'follow up'])
    }
    
    return metadata

def preprocess_email_dataset(df: pd.DataFrame, text_column: str = 'message') -> pd.DataFrame:
    
    logger.info(f"Preprocessing {len(df)} emails...")
    
    
    df['cleaned_text'] = df[text_column].apply(clean_email_text)
    
    
    metadata_list = df['cleaned_text'].apply(extract_email_metadata)
    metadata_df = pd.DataFrame(metadata_list.tolist())
    
    
    for col in metadata_df.columns:
        df[f'meta_{col}'] = metadata_df[col]
    
    
    original_count = len(df)
    df = df[df['cleaned_text'].str.len() > 50]  
    filtered_count = len(df)
    
    logger.info(f"Filtered out {original_count - filtered_count} empty/short emails")
    logger.info(f"Final dataset size: {filtered_count}")
    
    return df

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
    
    from sklearn.model_selection import train_test_split
    
    
    train_val, test = train_test_split(df, test_size=test_size, random_state=42)
    
    
    val_size_adjusted = val_size / (1 - test_size)  
    train, val = train_test_split(train_val, test_size=val_size_adjusted, random_state=42)
    
    logger.info(f"Dataset split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    return {
        'train': train,
        'val': val,
        'test': test
    }