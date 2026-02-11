"""
File:       Data preprocessing module (src/data_preprocessing.py).
Purpose:    Handles loading, cleaning, and basic preprocessing of all data files.

Documentation:
    1. load_all_data: Loads all data files specified in config.
    2. explore_data: Performs exploratory data analysis on all loaded data.
    3. handle_missing_values: Handles missing values in all datasets.
    4. handle_outliers: Handles outliers in all datasets (optional, can be skipped).
    5. encode_categorical_variables: Encodes categorical variables.
    6. run_pipeline: Runs the complete preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Importing utility functions
from utils import (
    load_config, reduce_memory_usage, load_data, 
    display_data_info, get_numeric_categorical_columns, clean_memory
)

class DataPreprocessor:
    """
    Class to handle data preprocessing (keeping data and methods together in a class).
    Implications: 
    - Data is stored in self.data and self.processed_data.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            config_path: Path to config file (optional)
            
        - Load configuration using load_config() from utils.py.
        - Prepare dictionaries to store datasets
        """
        self.config = load_config() if config_path is None else load_config()
        self.data = {}
        self.processed_data = {}
        
    def load_all_data(self, sample_size: float = 1.0) -> None:
        """
        Load all data files specified in config.
        
        Args:
            sample_size: Fraction of data to load (for testing)
        """
        print("=" * 60)
        print("Loading all data files...")
        print("=" * 60)
        
        files = self.config['files']
        data_path = Path(self.config['data_path'])
        
        for name, file_name in files.items():
            try:
                # Skip if file doesn't exist
                if not (data_path / file_name).exists():
                    print(f"Warning: {file_name} not found, skipping...")
                    continue
                
                # Load the data
                df = load_data(file_name)
                
                # Apply sampling if needed
                if sample_size < 1.0:
                    df = df.sample(frac=sample_size, random_state=42)
                    print(f"Sampled to {len(df)} rows")
                
                self.data[name] = df
                
                # Display basic info
                print(f"\n{name}:")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:5]}...")  # Show first 5 columns
                
                # Check for target in training data
                if 'train' in name and 'TARGET' in df.columns:
                    target_dist = df['TARGET'].value_counts(normalize=True)
                    print(f"  Target distribution: {target_dist.to_dict()}")
                
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"Loaded {len(self.data)} data files")
        print("=" * 60)
    
    def explore_data(self) -> None:
        """
        Perform exploratory data analysis on all loaded data.
        """
        print("\n" + "=" * 60)
        print("Exploratory Data Analysis")
        print("=" * 60)
        
        for name, df in self.data.items():
            print(f"\n{'='*40}")
            print(f"Dataset: {name}")
            print(f"{'='*40}")
            
            # Display basic information
            display_data_info(df, name)
            
            # Display first few rows
            print(f"\nFirst 3 rows:")
            print(df.head(3))
            
            # For numeric columns, show statistics
            numeric_cols, _ = get_numeric_categorical_columns(df)
            if numeric_cols:
                print(f"\nSummary statistics for numeric columns:")
                print(df[numeric_cols].describe())
    
    def handle_missing_values(self, strategy: str = 'median', 
                              categorical_strategy: str = 'mode') -> None:
        """
        Handle missing values in all datasets.
        
        Args:
            strategy: Strategy for numeric columns ('mean', 'median', 'zero')
            categorical_strategy: Strategy for categorical columns ('mode', 'missing')
        """
        print("\n" + "=" * 60)
        print("Handling Missing Values")
        print("=" * 60)
        
        for name, df in self.data.items():
            print(f"\nProcessing: {name}")
            original_shape = df.shape
            
            # Identify numeric and categorical columns
            numeric_cols, cat_cols = get_numeric_categorical_columns(df)
            
            # Handle numeric columns
            if numeric_cols:
                missing_numeric = df[numeric_cols].isnull().sum().sum()
                if missing_numeric > 0:
                    print(f"  Numeric missing values: {missing_numeric}")
                    
                    if strategy == 'mean':
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    elif strategy == 'median':
                        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    elif strategy == 'zero':
                        df[numeric_cols] = df[numeric_cols].fillna(0)
                    
                    print(f"  Filled numeric missing values using {strategy}")
            
            # Handle categorical columns
            if cat_cols:
                missing_cat = df[cat_cols].isnull().sum().sum()
                if missing_cat > 0:
                    print(f"  Categorical missing values: {missing_cat}")
                    
                    if categorical_strategy == 'mode':
                        for col in cat_cols:
                            if df[col].isnull().any():
                                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                                df[col] = df[col].fillna(mode_val)
                    elif categorical_strategy == 'missing':
                        df[cat_cols] = df[cat_cols].fillna('Missing')
                    
                    print(f"  Filled categorical missing values using {categorical_strategy}")
            
            self.data[name] = df
            print(f"  Original shape: {original_shape}, After cleaning: {df.shape}")
    
    def handle_outliers(self, method: str = 'clip', threshold: float = 3.0) -> None:
        """
        Handle outliers in numeric columns.
        
        Args:
            method: 'clip' to clip outliers, 'remove' to remove rows
            threshold: Z-score threshold for outlier detection
        """
        print("\n" + "=" * 60)
        print("Handling Outliers")
        print("=" * 60)
        
        # Only handle outliers in main application data for now
        for name in ['application_train', 'application_test']:
            if name in self.data:
                df = self.data[name]
                numeric_cols, _ = get_numeric_categorical_columns(df)
                
                outliers_count = 0
                
                for col in numeric_cols:
                    if df[col].dtype in ['float64', 'float32', 'float16']:
                        # Calculate Z-scores
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        
                        if method == 'clip':
                            # Clip outliers to threshold
                            threshold_val = df[col].mean() + threshold * df[col].std()
                            outliers = z_scores > threshold
                            if outliers.any():
                                df.loc[outliers, col] = threshold_val
                                outliers_count += outliers.sum()
                        elif method == 'remove':
                            # Mark outliers for removal (will be handled later)
                            outliers = z_scores > threshold
                            outliers_count += outliers.sum()
                
                print(f"{name}: Found {outliers_count} potential outliers")
                self.data[name] = df
    
    def encode_categorical_variables(self, encoding_type: str = 'label') -> None:
        """
        Encode categorical variables.
        
        Args:
            encoding_type: 'label' for label encoding, 'onehot' for one-hot encoding
        """
        print("\n" + "=" * 60)
        print("Encoding Categorical Variables")
        print("=" * 60)
        
        from sklearn.preprocessing import LabelEncoder
        
        for name, df in self.data.items():
            print(f"\nProcessing: {name}")
            
            # Get categorical columns
            _, cat_cols = get_numeric_categorical_columns(df)
            
            if not cat_cols:
                print("  No categorical columns found")
                continue
            
            print(f"  Found {len(cat_cols)} categorical columns")
            
            if encoding_type == 'label':
                # Label encoding for tree-based models
                label_encoders = {}
                for col in cat_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    label_encoders[col] = le
                
                print(f"  Applied label encoding to {len(cat_cols)} columns")
                
            elif encoding_type == 'onehot':
                # One-hot encoding for linear models
                # Note: This can create many columns, use with caution
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                print(f"  Applied one-hot encoding, new shape: {df.shape}")
            
            self.data[name] = df
    
    def save_processed_data(self, output_dir: str = "processed_data") -> None:
        """
        Save processed data to disk.
        
        Args:
            output_dir: Directory to save processed data
        """
        import pickle
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        for name, df in self.data.items():
            file_path = output_path / f"{name}_processed.pkl"
            df.to_pickle(file_path)
            print(f"  Saved {name}: {len(df)} rows, {len(df.columns)} columns")
        
        # Also save metadata
        metadata = {
            'datasets': list(self.data.keys()),
            'shapes': {name: df.shape for name, df in self.data.items()}
        }
        
        with open(output_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        print("\nProcessing complete!")
    
    def run_pipeline(self, sample_size: float = 1.0) -> None:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            sample_size: Fraction of data to load
        """
        print("Starting Data Preprocessing Pipeline")
        print("=" * 60)
        
        # Step 1: Load data
        self.load_all_data(sample_size=sample_size)
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Handle outliers (optional, can be skipped)
        # self.handle_outliers()
        
        # Step 5: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 6: Clean memory
        clean_memory()
        
        # Step 7: Save processed data
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("Preprocessing Pipeline Complete!")
        print("=" * 60)

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Run with 10% sample for testing
    preprocessor.run_pipeline(sample_size=0.1)
    
    # For full data, use:
    # preprocessor.run_pipeline(sample_size=1.0)