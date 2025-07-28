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
from utils.preprocessing import email_preprocessing

