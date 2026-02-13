
"""
File:       Utility functions (src/utils.py).
Purpose:    Contains helper functions for data loading, memory optimization, and common operations.

Documentation:
    1. load_config: Loads configuration settings from config.yaml file.
    2. load_data: Loads CSV data efficiently and reduces memory usage.
    3. display_data_info: Inspects datasets (missing values, column types and more).
    4. save_object: Saves Python objects to a pickle file.
    5. load_object: Loads Python objects from a pickle file.
    6. clean_memory: Cleans memory when needed.

"""

import pandas as pd
import numpy as np
import yaml
import pickle
import os
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Defining project root (so file paths can be relative to the project folder.)
PROJECT_ROOT = Path(__file__).parent.parent

def load_config() -> Dict:
    """
    Load configuration from config.yaml file.
    
    Returns:
        Dictionary containing configuration settings
    """
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Optimize memory usage by downcasting numeric columns.
     - Converts large integers/floats to smaller types if possible (int64 -> int32 -> int16 -> int8).
     - Same for floats (float64 -> float32 -> float16).
    
    Args:
        df: pandas DataFrame
        verbose: Whether to print memory reduction statistics
        
    Returns:
        Memory-optimized DataFrame
    """

    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Initial memory usage: {start_mem:.2f} MB")

    for col in df.columns:
        col_dtype = df[col].dtype

        # Skip non-numeric columns safely
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        # Integer types
        if pd.api.types.is_integer_dtype(df[col]):
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        # Float types
        elif pd.api.types.is_float_dtype(df[col]):

            # 🔹 Recommendation: Avoid float16 for modeling stability
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f"Final memory usage: {end_mem:.2f} MB")
        print(f"Memory reduced by {reduction:.1f}%")

    return df


def load_data(file_name: str, nrows: Optional[int] = None, 
              usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load data from CSV file with memory optimization.
    
    Args:
        file_name: Name of CSV file
        nrows: Number of rows to load (optional)
        usecols: List of columns to load (optional)
        
    Returns:
        Loaded DataFrame
    """
    config = load_config()
    data_path = PROJECT_ROOT / config['data_path'] / file_name
    
    print(f"Loading {file_name}...")
    
    # Load data
    if nrows is not None:
        df = pd.read_csv(data_path, nrows=nrows, usecols=usecols)
    else:
        df = pd.read_csv(data_path, usecols=usecols)
    
    # Optimize memory
    df = reduce_memory_usage(df)
    
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    
    return df

def save_object(obj: Any, file_path: str) -> None:
    """
    Save Python object to disk using pickle.
    
    Args:
        obj: Python object to save
        file_path: Path to save file
    """
    full_path = PROJECT_ROOT / file_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(full_path, 'wb') as f:
        pickle.dump(obj, f)
    
    print(f"Object saved to {full_path}")

def load_object(file_path: str) -> Any:
    """
    Load Python object from disk.
    
    Args:
        file_path: Path to saved file
        
    Returns:
        Loaded object
    """
    full_path = PROJECT_ROOT / file_path
    
    with open(full_path, 'rb') as f:
        obj = pickle.load(f)
    
    print(f"Object loaded from {full_path}")
    return obj

def display_data_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Display comprehensive information about a DataFrame.
    
    Args:
        df: pandas DataFrame
        name: Name of the DataFrame for display
    """
    print(f"=== {name} Information ===")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print("\nColumn Types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    missing_percent = 100 * missing / len(df)
    missing_df = pd.DataFrame({
        'missing_count': missing,
        'missing_percent': missing_percent
    })
    print(missing_df[missing_df['missing_count'] > 0].sort_values('missing_percent', ascending=False).head(10))
    
    print(f"\nTotal missing values: {missing.sum():,}")
    print(f"Percentage of data missing: {100 * missing.sum() / (df.shape[0] * df.shape[1]):.2f}%")

def get_numeric_categorical_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical columns in DataFrame.
    
    Args:
        df: pandas DataFrame
        exclude_cols: Columns to exclude from analysis
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    all_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Identify numeric columns (int and float)
    numeric_cols = df[all_cols].select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Identify categorical columns (object and category)
    cat_cols = df[all_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Also consider columns with low cardinality as categorical
    for col in all_cols:
        if col not in numeric_cols and col not in cat_cols:
            if df[col].nunique() < 20:  # Low cardinality
                cat_cols.append(col)
            else:
                numeric_cols.append(col)
    
    return numeric_cols, cat_cols

def clean_memory() -> None:
    """
    Clean up memory by running garbage collection.
    """
    gc.collect()
    print("Memory cleaned")

def get_file_sizes() -> Dict[str, float]:
    """
    Get sizes of all data files in MB.
    
    Returns:
        Dictionary with file names and sizes in MB
    """
    config = load_config()
    data_path = PROJECT_ROOT / config['data_path']
    
    file_sizes = {}
    for file in data_path.glob('*.csv'):
        size_mb = file.stat().st_size / (1024 * 1024)
        file_sizes[file.name] = round(size_mb, 2)
    
    return file_sizes

if __name__ == "__main__":
    # Test the utility functions
    config = load_config()
    print("Configuration loaded successfully")
    print(f"Data path: {config['data_path']}")