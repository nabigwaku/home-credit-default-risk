"""
Feature engineering module for Home Credit Default Risk.
Creates new features by aggregating data from multiple sources.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from utils import (
    load_config, reduce_memory_usage, load_data, 
    save_object, load_object, clean_memory, get_numeric_categorical_columns
)

class FeatureEngineer:
    """
    Class to handle feature engineering for Home Credit dataset.
    
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize FeatureEngineer.
        
        Args:
            config_path: Path to config file
        """
        self.config = load_config() if config_path is None else load_config()
        self.data = {}
        self.features = {}
        
    def load_processed_data(self) -> None:
        """
        Load processed data from previous step.
        """
        print("Loading processed data...")
        
        # Try to load processed data first
        processed_path = Path("processed_data")
        if processed_path.exists():
            for file in processed_path.glob("*_processed.pkl"):
                name = file.stem.replace("_processed", "")
                self.data[name] = pd.read_pickle(file)
                print(f"  Loaded {name}: {self.data[name].shape}")
        else:
            # If no processed data, load raw data
            print("No processed data found, loading raw data...")
            self.load_raw_data()
    
    def load_raw_data(self) -> None:
        """
        Load raw data files.
        """
        for name, file_name in self.config['files'].items():
            if name in ['application_train', 'application_test']:
                self.data[name] = load_data(file_name)
    
    def create_application_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from main application data.
        
        Args:
            df: Main application DataFrame
            
        Returns:
            DataFrame with new features
        """
        print("Creating application features...")
        
        features_df = df.copy()
        
        # 1. Income-related features
        if 'AMT_INCOME_TOTAL' in df.columns:
            # Income to credit ratio
            if 'AMT_CREDIT' in df.columns:
                features_df['INCOME_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_CREDIT'] + 1)
                features_df['INCOME_CREDIT_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Income per family member
            if 'CNT_FAM_MEMBERS' in df.columns:
                features_df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
            
            # Income to annuity ratio
            if 'AMT_ANNUITY' in df.columns:
                features_df['INCOME_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / (df['AMT_ANNUITY'] + 1)
                features_df['INCOME_ANNUITY_RATIO'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Credit-related features
        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            # Credit to goods price ratio
            features_df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
        
        # 3. Age-related features
        if 'DAYS_BIRTH' in df.columns:
            # Convert days to years (negative values)
            features_df['AGE'] = -df['DAYS_BIRTH'] / 365.25
            features_df['AGE'] = features_df['AGE'].round(0)
        
        # 4. Employment-related features
        if 'DAYS_EMPLOYED' in df.columns:
            # Flag for unrealistic employment days (1000 years!)
            features_df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
            # Replace anomaly with NaN
            features_df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            
            # Employment length in years
            features_df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365.25
        
        # 5. Document features - count of documents provided
        doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
        if doc_cols:
            features_df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
        
        # 6. Contact features
        contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 
                       'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
        contact_cols = [col for col in contact_cols if col in df.columns]
        if contact_cols:
            features_df['CONTACT_INFO_COUNT'] = df[contact_cols].sum(axis=1)
        
        print(f"  Created {len(features_df.columns) - len(df.columns)} new features")
        return features_df
    
    def create_bureau_features(self, application_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from bureau data (previous loans from other institutions).
        
        Args:
            application_df: Main application DataFrame
            
        Returns:
            DataFrame with bureau features
        """
        print("Creating bureau features...")
        
        if 'bureau' not in self.data:
            print("  Bureau data not loaded, skipping...")
            return application_df
        
        bureau = self.data['bureau'].copy()
        
        # Aggregate bureau data by SK_ID_CURR
        bureau_agg = {}
        
        # 1. Basic aggregations
        bureau_agg['BUREAU_COUNT'] = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()
        
        # 2. Credit amount aggregations
        if 'AMT_CREDIT_SUM' in bureau.columns:
            bureau_agg.update({
                'BUREAU_CREDIT_SUM_MEAN': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean(),
                'BUREAU_CREDIT_SUM_MAX': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].max(),
                'BUREAU_CREDIT_SUM_SUM': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum(),
            })
        
        # 3. Debt aggregations
        if 'AMT_CREDIT_SUM_DEBT' in bureau.columns:
            bureau_agg.update({
                'BUREAU_DEBT_MEAN': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].mean(),
                'BUREAU_DEBT_MAX': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].max(),
                'BUREAU_DEBT_SUM': bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum(),
            })
        
        # 4. Credit duration
        if 'CREDIT_DAY_OVERDUE' in bureau.columns:
            bureau_agg['BUREAU_MAX_OVERDUE'] = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max()
        
        if 'DAYS_CREDIT' in bureau.columns:
            bureau_agg.update({
                'BUREAU_CREDIT_DAYS_MEAN': bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].mean(),
                'BUREAU_CREDIT_DAYS_MAX': bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max(),
            })
        
        # 5. Credit active status
        if 'CREDIT_ACTIVE' in bureau.columns:
            # Count of active credits
            active_credits = bureau[bureau['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size()
            bureau_agg['BUREAU_ACTIVE_COUNT'] = active_credits
            
            # Ratio of active to total credits
            bureau_agg['BUREAU_ACTIVE_RATIO'] = active_credits / bureau_agg['BUREAU_COUNT']
        
        # 6. Credit type counts
        if 'CREDIT_TYPE' in bureau.columns:
            # Number of unique credit types
            bureau_agg['BUREAU_CREDIT_TYPES'] = bureau.groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique()
        
        # Create aggregated DataFrame
        bureau_features = pd.DataFrame(bureau_agg).reset_index()
        
        # Merge with application data
        result_df = application_df.merge(bureau_features, on='SK_ID_CURR', how='left')
        
        # Fill missing values (for applicants with no bureau data)
        bureau_cols = [col for col in bureau_features.columns if col != 'SK_ID_CURR']
        result_df[bureau_cols] = result_df[bureau_cols].fillna(0)
        
        print(f"  Created {len(bureau_cols)} bureau features")
        return result_df
    
    def create_previous_application_features(self, application_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from previous Home Credit applications.
        
        Args:
            application_df: Main application DataFrame
            
        Returns:
            DataFrame with previous application features
        """
        print("Creating previous application features...")
        
        if 'previous_application' not in self.data:
            print("  Previous application data not loaded, skipping...")
            return application_df
        
        prev = self.data['previous_application'].copy()
        
        # Aggregate previous application data
        prev_agg = {}
        
        # 1. Basic counts
        prev_agg['PREV_APPLICATION_COUNT'] = prev.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
        
        # 2. Credit amount aggregations
        if 'AMT_CREDIT' in prev.columns:
            prev_agg.update({
                'PREV_AMT_CREDIT_MEAN': prev.groupby('SK_ID_CURR')['AMT_CREDIT'].mean(),
                'PREV_AMT_CREDIT_MAX': prev.groupby('SK_ID_CURR')['AMT_CREDIT'].max(),
                'PREV_AMT_CREDIT_SUM': prev.groupby('SK_ID_CURR')['AMT_CREDIT'].sum(),
            })
        
        # 3. Application status
        if 'NAME_CONTRACT_STATUS' in prev.columns:
            # Count of approved applications
            approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').size()
            prev_agg['PREV_APPROVED_COUNT'] = approved
            
            # Ratio of approved applications
            prev_agg['PREV_APPROVED_RATIO'] = approved / prev_agg['PREV_APPLICATION_COUNT']
            
            # Count of refused applications
            refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').size()
            prev_agg['PREV_REFUSED_COUNT'] = refused
        
        # 4. Days decision
        if 'DAYS_DECISION' in prev.columns:
            prev_agg['PREV_DAYS_DECISION_MEAN'] = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].mean()
            prev_agg['PREV_DAYS_DECISION_MAX'] = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].max()
        
        # 5. Payment difference
        if 'AMT_APPLICATION' in prev.columns and 'AMT_CREDIT' in prev.columns:
            prev['AMT_DIFF'] = prev['AMT_CREDIT'] - prev['AMT_APPLICATION']
            prev_agg['PREV_AMT_DIFF_MEAN'] = prev.groupby('SK_ID_CURR')['AMT_DIFF'].mean()
        
        # Create aggregated DataFrame
        prev_features = pd.DataFrame(prev_agg).reset_index()
        
        # Merge with application data
        result_df = application_df.merge(prev_features, on='SK_ID_CURR', how='left')
        
        # Fill missing values
        prev_cols = [col for col in prev_features.columns if col != 'SK_ID_CURR']
        result_df[prev_cols] = result_df[prev_cols].fillna(0)
        
        print(f"  Created {len(prev_cols)} previous application features")
        return result_df
    
    def create_payment_features(self, application_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from payment history data.
        
        Args:
            application_df: Main application DataFrame
            
        Returns:
            DataFrame with payment features
        """
        print("Creating payment features...")
        
        # We'll create simple payment features
        # In practice, you would aggregate from POS_CASH_balance, installments_payments, credit_card_balance
        
        result_df = application_df.copy()
        
        # These would be filled with actual payment aggregations
        # For now, creating placeholder features
        
        payment_features = [
            'PAYMENT_LATE_COUNT',
            'PAYMENT_LATE_DAYS_MEAN',
            'PAYMENT_LATE_DAYS_MAX',
            'PAYMENT_AMOUNT_MEAN',
            'PAYMENT_AMOUNT_SUM'
        ]
        
        # Initialize with zeros (will be filled if data is available)
        for feature in payment_features:
            result_df[feature] = 0
        
        print(f"  Created {len(payment_features)} payment features (placeholders)")
        return result_df
    
    def create_all_features(self, dataset: str = 'train') -> pd.DataFrame:
        """
        Create all features for specified dataset.
        
        Args:
            dataset: 'train' or 'test'
            
        Returns:
            DataFrame with all engineered features
        """
        print(f"\n{'='*60}")
        print(f"Creating features for {dataset} dataset")
        print(f"{'='*60}")
        
        # Load appropriate dataset
        if dataset == 'train':
            df_name = 'application_train'
        else:
            df_name = 'application_test'
        
        if df_name not in self.data:
            print(f"  {df_name} not found in data, loading...")
            self.data[df_name] = load_data(self.config['files'][df_name])
        
        df = self.data[df_name].copy()
        
        # Store original shape
        original_shape = df.shape
        
        # Step 1: Application features
        df = self.create_application_features(df)
        
        # Step 2: Bureau features (external loans)
        df = self.create_bureau_features(df)
        
        # Step 3: Previous application features
        df = self.create_previous_application_features(df)
        
        # Step 4: Payment features
        df = self.create_payment_features(df)
        
        # Step 5: Feature selection (remove constant columns)
        df = self.remove_constant_columns(df)
        
        # Step 6: Reduce memory usage
        df = reduce_memory_usage(df, verbose=False)
        
        # Store features
        self.features[dataset] = df
        
        print(f"\nFeature creation complete!")
        print(f"Original columns: {original_shape[1]}")
        print(f"New columns: {df.shape[1]}")
        print(f"Total features created: {df.shape[1] - original_shape[1]}")
        
        return df
    
    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant and near-constant columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with constant columns removed
        """
        constant_columns = []
        
        for col in df.columns:
            # Skip ID and target columns
            if col in ['SK_ID_CURR', 'TARGET', 'SK_ID_PREV', 'SK_ID_BUREAU']:
                continue
            
            # Check if column is constant
            if df[col].nunique() <= 1:
                constant_columns.append(col)
            # Check if column has too many missing values (>90%)
            elif df[col].isnull().sum() / len(df) > 0.9:
                constant_columns.append(col)
        
        if constant_columns:
            print(f"  Removing {len(constant_columns)} constant/near-constant columns")
            df = df.drop(columns=constant_columns)
        
        return df
    
    def save_features(self, dataset: str = 'train') -> None:
        """
        Save engineered features to disk.
        
        Args:
            dataset: 'train' or 'test'
        """
        if dataset not in self.features:
            print(f"  No features found for {dataset}, creating them first...")
            self.create_all_features(dataset)
        
        # Create features directory if it doesn't exist
        features_path = Path(self.config['features_path'])
        features_path.mkdir(exist_ok=True)
        
        # Save features
        file_path = features_path / f"{dataset}_features.pkl"
        self.features[dataset].to_pickle(file_path)
        
        print(f"\nFeatures saved to: {file_path}")
        print(f"Shape: {self.features[dataset].shape}")
    
    def run_pipeline(self) -> None:
        """
        Run complete feature engineering pipeline.
        """
        print("Starting Feature Engineering Pipeline")
        print("=" * 60)
        
        # Load data
        self.load_processed_data()
        
        # Create features for train and test
        self.create_all_features('train')
        self.create_all_features('test')
        
        # Save features
        self.save_features('train')
        self.save_features('test')
        
        # Clean memory
        clean_memory()
        
        print("\n" + "=" * 60)
        print("Feature Engineering Pipeline Complete!")
        print("=" * 60)

if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()
    
    # Run complete pipeline
    engineer.run_pipeline()
    
    # Or create features for specific dataset
    # train_features = engineer.create_all_features('train')
    # test_features = engineer.create_all_features('test')