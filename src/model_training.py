"""
Model training module for Home Credit Default Risk.
Handles model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_config, load_object, save_object, 
    get_numeric_categorical_columns, clean_memory
)

class ModelTrainer:
    """
    Class to handle model training and evaluation.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize ModelTrainer.
        
        Args:
            config_path: Path to config file
        """
        self.config = load_config() if config_path is None else load_config()
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_features(self, dataset: str = 'train') -> pd.DataFrame:
        """
        Load engineered features.
        
        Args:
            dataset: 'train' or 'test'
            
        Returns:
            DataFrame with features
        """
        features_path = Path(self.config['features_path']) / f"{dataset}_features.pkl"
        
        if not features_path.exists():
            print(f"  Features not found for {dataset}, please run feature engineering first.")
            return None
        
        print(f"Loading {dataset} features...")
        df = pd.read_pickle(features_path)
        print(f"  Loaded {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def prepare_data(self) -> tuple:
        """
        Prepare data for model training.
        
        Returns:
            Tuple of (X_train, X_val, y_train, y_val, feature_names)
        """
        print("\n" + "="*60)
        print("Preparing data for training")
        print("="*60)
        
        # Load training features
        train_df = self.load_features('train')
        if train_df is None:
            return None
        
        # Separate features and target
        if 'TARGET' not in train_df.columns:
            print("  Error: TARGET column not found in training data")
            return None
        
        X = train_df.drop(columns=['TARGET', 'SK_ID_CURR'])
        y = train_df['TARGET']
        
        # Get feature names
        feature_names = list(X.columns)
        print(f"\nFeatures for training: {len(feature_names)}")
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        print(f"\nData split:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Positive samples in train: {y_train.sum()} ({100*y_train.mean():.2f}%)")
        print(f"  Positive samples in val: {y_val.sum()} ({100*y_val.mean():.2f}%)")
        
        return X_train, X_val, y_train, y_val, feature_names
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train logistic regression model.
        
        Returns:
            Dictionary with model and results
        """
        print("\nTraining Logistic Regression...")
        
        model = LogisticRegression(
            random_state=self.config['model']['random_state'],
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Store results
        result = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba
        }
        
        return result
    
    def train_random_forest(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train random forest model.
        
        Returns:
            Dictionary with model and results
        """
        print("\nTraining Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.config['model']['random_state'],
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        result = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'feature_importance': importance
        }
        
        return result
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train XGBoost model.
        
        Returns:
            Dictionary with model and results
        """
        print("\nTraining XGBoost...")
        
        # Get hyperparameters from config
        params = self.config['hyperparameters']['xgboost']
        
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=self.config['model']['random_state'],
            n_jobs=-1,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        result = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'feature_importance': importance
        }
        
        return result
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train LightGBM model.
        
        Returns:
            Dictionary with model and results
        """
        print("\nTraining LightGBM...")
        
        # Get hyperparameters from config
        params = self.config['hyperparameters']['lightgbm']
        
        model = lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            num_leaves=params['num_leaves'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=self.config['model']['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        result = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'feature_importance': importance
        }
        
        return result
    
    def train_catboost(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train CatBoost model.
        
        Returns:
            Dictionary with model and results
        """
        print("\nTraining CatBoost...")
        
        # Get hyperparameters from config
        params = self.config['hyperparameters']['catboost']
        
        # Identify categorical columns
        _, cat_cols = get_numeric_categorical_columns(X_train)
        
        model = cb.CatBoostClassifier(
            iterations=params['iterations'],
            learning_rate=params['learning_rate'],
            depth=params['depth'],
            l2_leaf_reg=params['l2_leaf_reg'],
            random_state=self.config['model']['random_state'],
            verbose=False,
            cat_features=cat_cols
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50
        )
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  Validation AUC: {auc:.4f}")
        print(f"  Validation Accuracy: {accuracy:.4f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        result = {
            'model': model,
            'auc': auc,
            'accuracy': accuracy,
            'predictions': y_pred,
            'predictions_proba': y_pred_proba,
            'feature_importance': importance
        }
        
        return result
    
    def evaluate_model(self, model_name: str, y_true, y_pred, y_pred_proba) -> None:
        """
        Evaluate model performance.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
        """
        print(f"\n{'='*60}")
        print(f"Evaluation for {model_name}")
        print(f"{'='*60}")
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"AUC Score: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Default', 'Default']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Store results
        self.results[model_name] = {
            'auc': auc,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def cross_validate_model(self, model, X, y, model_name: str = 'Model') -> float:
        """
        Perform cross-validation.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Target
            model_name: Name of model for printing
            
        Returns:
            Mean cross-validation score
        """
        print(f"\nCross-validating {model_name}...")
        
        cv = StratifiedKFold(
            n_splits=self.config['model']['cv_folds'],
            shuffle=True,
            random_state=self.config['model']['random_state']
        )
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        print(f"  CV AUC Scores: {scores}")
        print(f"  Mean CV AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores.mean()
    
    def train_all_models(self) -> None:
        """
        Train all models and compare performance.
        """
        print("\n" + "="*60)
        print("Training All Models")
        print("="*60)
        
        # Prepare data
        data = self.prepare_data()
        if data is None:
            return
        
        X_train, X_val, y_train, y_val, feature_names = data
        
        # Dictionary of model training functions
        model_functions = {
            'logistic_regression': self.train_logistic_regression,
            'random_forest': self.train_random_forest,
            'xgboost': self.train_xgboost,
            'lightgbm': self.train_lightgbm,
            'catboost': self.train_catboost
        }
        
        # Train each model
        for model_name, train_func in model_functions.items():
            try:
                result = train_func(X_train, y_train, X_val, y_val)
                self.models[model_name] = result['model']
                
                # Evaluate
                self.evaluate_model(
                    model_name,
                    y_val,
                    result['predictions'],
                    result['predictions_proba']
                )
                
                # Store feature importance if available
                if 'feature_importance' in result:
                    self.feature_importance[model_name] = result['feature_importance']
                
                # Clean memory between models
                clean_memory()
                
            except Exception as e:
                print(f"\nError training {model_name}: {str(e)}")
        
        # Compare model performance
        self.compare_models()
    
    def compare_models(self) -> None:
        """
        Compare performance of all trained models.
        """
        if not self.results:
            print("No models to compare")
            return
        
        print("\n" + "="*60)
        print("Model Comparison")
        print("="*60)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame(self.results).T
        
        # Select only numeric columns
        numeric_cols = ['auc', 'accuracy']
        comparison = comparison[numeric_cols]
        
        print("\nModel Performance:")
        print(comparison.sort_values('auc', ascending=False))
        
        # Visualize comparison
        self.plot_model_comparison(comparison)
        
        # Select best model based on AUC
        best_model_name = comparison['auc'].idxmax()
        best_auc = comparison.loc[best_model_name, 'auc']
        
        print(f"\nBest Model: {best_model_name} (AUC: {best_auc:.4f})")
        
        # Save best model
        self.save_best_model(best_model_name)
    
    def plot_model_comparison(self, comparison: pd.DataFrame) -> None:
        """
        Plot model comparison.
        
        Args:
            comparison: DataFrame with model comparison
        """
        plt.figure(figsize=(12, 6))
        
        # Plot AUC scores
        plt.subplot(1, 2, 1)
        comparison['auc'].sort_values().plot(kind='barh', color='skyblue')
        plt.title('Model AUC Scores')
        plt.xlabel('AUC Score')
        plt.grid(True, alpha=0.3)
        
        # Plot accuracy scores
        plt.subplot(1, 2, 2)
        comparison['accuracy'].sort_values().plot(kind='barh', color='lightgreen')
        plt.title('Model Accuracy Scores')
        plt.xlabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot feature importance for tree-based models
        self.plot_feature_importance()
    
    def plot_feature_importance(self, top_n: int = 20) -> None:
        """
        Plot feature importance for tree-based models.
        
        Args:
            top_n: Number of top features to show
        """
        tree_models = ['random_forest', 'xgboost', 'lightgbm', 'catboost']
        
        for model_name in tree_models:
            if model_name in self.feature_importance:
                importance_df = self.feature_importance[model_name]
                
                plt.figure(figsize=(10, 8))
                top_features = importance_df.head(top_n)
                
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Importance')
                plt.title(f'Top {top_n} Features - {model_name.replace("_", " ").title()}')
                plt.gca().invert_yaxis()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'feature_importance_{model_name}.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    def save_best_model(self, model_name: str) -> None:
        """
        Save the best model to disk.
        
        Args:
            model_name: Name of the best model
        """
        if model_name not in self.models:
            print(f"  Model {model_name} not found in trained models")
            return
        
        # Create models directory
        models_path = Path(self.config['models_path'])
        models_path.mkdir(exist_ok=True)
        
        # Save model
        file_path = models_path / 'best_model.pkl'
        save_object(self.models[model_name], str(file_path))
        
        print(f"\nBest model saved to: {file_path}")
        
        # Also save feature importance if available
        if model_name in self.feature_importance:
            importance_path = models_path / 'feature_importance.pkl'
            save_object(self.feature_importance[model_name], str(importance_path))
            print(f"Feature importance saved to: {importance_path}")
    
    def make_predictions(self, model_path: str = None) -> pd.DataFrame:
        """
        Make predictions on test data.
        
        Args:
            model_path: Path to saved model (optional)
            
        Returns:
            DataFrame with predictions
        """
        print("\n" + "="*60)
        print("Making Predictions on Test Data")
        print("="*60)
        
        # Load test features
        test_df = self.load_features('test')
        if test_df is None:
            return None
        
        # Store SK_ID_CURR for submission
        sk_id_curr = test_df['SK_ID_CURR']
        
        # Prepare features
        X_test = test_df.drop(columns=['SK_ID_CURR'])
        
        # Load model
        if model_path:
            model = load_object(model_path)
        elif self.models:
            # Use the best model from training
            best_model_name = max(self.results, key=lambda x: self.results[x]['auc'])
            model = self.models[best_model_name]
        else:
            print("  No model available, please train or load a model first")
            return None
        
        print(f"\nModel used: {type(model).__name__}")
        print(f"Test data shape: {X_test.shape}")
        
        # Handle missing values
        X_test = X_test.fillna(X_test.mean())
        
        # Make predictions
        print("Making predictions...")
        predictions_proba = model.predict_proba(X_test)[:, 1]
        predictions = (predictions_proba > self.config['submission']['threshold']).astype(int)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'SK_ID_CURR': sk_id_curr,
            'TARGET': predictions_proba  # Using probabilities for better scoring
        })
        
        # Save submission
        submissions_path = Path(self.config['submissions_path'])
        submissions_path.mkdir(exist_ok=True)
        
        file_name = self.config['submission']['file_name']
        submission_file = submissions_path / file_name
        submission_df.to_csv(submission_file, index=False)
        
        print(f"\nSubmission saved to: {submission_file}")
        print(f"Shape: {submission_df.shape}")
        print(f"Prediction distribution:")
        print(f"  Default probability > 0.5: {(predictions == 1).sum()} samples")
        print(f"  Default probability <= 0.5: {(predictions == 0).sum()} samples")
        
        return submission_df
    
    def run_pipeline(self) -> None:
        """
        Run complete model training pipeline.
        """
        print("Starting Model Training Pipeline")
        print("=" * 60)
        
        # Train all models
        self.train_all_models()
        
        # Make predictions
        self.make_predictions()
        
        print("\n" + "=" * 60)
        print("Model Training Pipeline Complete!")
        print("=" * 60)

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    # Run complete pipeline
    trainer.run_pipeline()
    
    # Or run specific parts
    # trainer.train_all_models()
    # submission = trainer.make_predictions()