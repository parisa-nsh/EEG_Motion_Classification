"""
Auto-Tuner Module for CNN Hyperparameter Optimization

This module implements automatic hyperparameter tuning for the EEG CNN model
using grid search and random search strategies.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import random
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import os

from .cnn_model import EEGCNNModel
from .data_processor import EEGDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGAutoTuner:
    """
    Automatic hyperparameter tuner for EEG CNN models.
    """
    
    def __init__(self, data_dict: Dict[str, np.ndarray], 
                 max_trials: int = 20, max_time: int = 3600):
        """
        Initialize the auto-tuner.
        
        Args:
            data_dict: Dictionary containing prepared data
            max_trials: Maximum number of trials for optimization
            max_time: Maximum time in seconds for optimization
        """
        self.data_dict = data_dict
        self.max_trials = max_trials
        self.max_time = max_time
        self.results = []
        self.best_params = None
        self.best_score = 0.0
        self.best_model = None
        
        # Define parameter search spaces
        self.param_grid = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'epochs': [30, 50, 100],
            'filters': [
                [32, 64],
                [32, 64, 128],
                [64, 128, 256],
                [32, 64, 128, 256]
            ],
            'kernel_size': [3, 5, 7],
            'dropout_rate': [0.3, 0.5, 0.7]
        }
        
    def grid_search(self) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter optimization.
        
        Returns:
            Dictionary containing best parameters and results
        """
        try:
            logger.info("Starting grid search hyperparameter optimization")
            
            start_time = time.time()
            trial_count = 0
            
            # Generate all parameter combinations
            param_combinations = list(ParameterGrid(self.param_grid))
            total_combinations = len(param_combinations)
            
            logger.info(f"Total parameter combinations: {total_combinations}")
            
            for params in param_combinations:
                # Check time limit
                if time.time() - start_time > self.max_time:
                    logger.info("Time limit reached, stopping grid search")
                    break
                
                # Check trial limit
                if trial_count >= self.max_trials:
                    logger.info("Trial limit reached, stopping grid search")
                    break
                
                trial_count += 1
                logger.info(f"Trial {trial_count}/{min(self.max_trials, total_combinations)}")
                logger.info(f"Testing parameters: {params}")
                
                try:
                    # Train model with current parameters
                    score, model, history = self._train_and_evaluate(params)
                    
                    # Store results
                    result = {
                        'trial': trial_count,
                        'parameters': params,
                        'score': score,
                        'timestamp': datetime.now().isoformat(),
                        'training_time': history.get('training_time', 0)
                    }
                    self.results.append(result)
                    
                    # Update best parameters if better score found
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                        self.best_model = model
                        logger.info(f"New best score: {score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Trial {trial_count} failed: {e}")
                    continue
            
            # Create summary
            summary = {
                'search_type': 'grid_search',
                'total_trials': trial_count,
                'best_score': self.best_score,
                'best_parameters': self.best_params,
                'total_time': time.time() - start_time,
                'all_results': self.results
            }
            
            logger.info(f"Grid search completed. Best score: {self.best_score:.4f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            raise
    
    def random_search(self) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter optimization.
        
        Returns:
            Dictionary containing best parameters and results
        """
        try:
            logger.info("Starting random search hyperparameter optimization")
            
            start_time = time.time()
            trial_count = 0
            
            while trial_count < self.max_trials:
                # Check time limit
                if time.time() - start_time > self.max_time:
                    logger.info("Time limit reached, stopping random search")
                    break
                
                trial_count += 1
                logger.info(f"Trial {trial_count}/{self.max_trials}")
                
                # Generate random parameters
                params = self._generate_random_params()
                logger.info(f"Testing parameters: {params}")
                
                try:
                    # Train model with current parameters
                    score, model, history = self._train_and_evaluate(params)
                    
                    # Store results
                    result = {
                        'trial': trial_count,
                        'parameters': params,
                        'score': score,
                        'timestamp': datetime.now().isoformat(),
                        'training_time': history.get('training_time', 0)
                    }
                    self.results.append(result)
                    
                    # Update best parameters if better score found
                    if score > self.best_score:
                        self.best_score = score
                        self.best_params = params.copy()
                        self.best_model = model
                        logger.info(f"New best score: {score:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Trial {trial_count} failed: {e}")
                    continue
            
            # Create summary
            summary = {
                'search_type': 'random_search',
                'total_trials': trial_count,
                'best_score': self.best_score,
                'best_parameters': self.best_params,
                'total_time': time.time() - start_time,
                'all_results': self.results
            }
            
            logger.info(f"Random search completed. Best score: {self.best_score:.4f}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in random search: {e}")
            raise
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """
        Generate random parameters for random search.
        
        Returns:
            Dictionary of random parameters
        """
        params = {}
        
        # Learning rate (log-uniform)
        params['learning_rate'] = 10 ** random.uniform(-4, -1)
        
        # Batch size
        params['batch_size'] = random.choice([16, 32, 64, 128])
        
        # Epochs
        params['epochs'] = random.choice([30, 50, 100, 150])
        
        # Filters (random number of layers with random filter sizes)
        n_layers = random.randint(2, 4)
        filter_sizes = [32, 64, 128, 256]
        params['filters'] = random.sample(filter_sizes, n_layers)
        
        # Kernel size
        params['kernel_size'] = random.choice([3, 5, 7, 9])
        
        # Dropout rate
        params['dropout_rate'] = random.uniform(0.2, 0.8)
        
        return params
    
    def _train_and_evaluate(self, params: Dict[str, Any]) -> Tuple[float, EEGCNNModel, Dict]:
        """
        Train and evaluate a model with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Tuple of (score, model, history)
        """
        try:
            # Extract data
            X_train = self.data_dict['X_train']
            y_train = self.data_dict['y_train']
            X_test = self.data_dict['X_test']
            y_test = self.data_dict['y_test']
            
            # Create model
            input_shape = X_train.shape[1:]
            model = EEGCNNModel(
                input_shape=input_shape,
                num_classes=6,
                filters=params['filters'],
                kernel_size=params['kernel_size'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate']
            )
            
            # Build model
            model.build_model()
            
            # Train model
            history = model.train(
                X_train, y_train,
                X_test, y_test,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0
            )
            
            # Evaluate model
            eval_results = model.evaluate(X_test, y_test)
            score = eval_results['test_accuracy']
            
            return score, model, history
            
        except Exception as e:
            logger.error(f"Error in train and evaluate: {e}")
            raise
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all trial results
        """
        if not self.results:
            return pd.DataFrame()
        
        # Flatten parameters for DataFrame
        df_data = []
        for result in self.results:
            row = {
                'trial': result['trial'],
                'score': result['score'],
                'timestamp': result['timestamp'],
                'training_time': result['training_time']
            }
            
            # Add parameters
            for key, value in result['parameters'].items():
                if isinstance(value, list):
                    row[f'{key}_layers'] = len(value)
                    row[f'{key}_values'] = str(value)
                else:
                    row[key] = value
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                logger.warning("No results to plot")
                return
            
            df = self.get_results_dataframe()
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Score progression
            axes[0, 0].plot(df['trial'], df['score'], 'b-', alpha=0.7)
            axes[0, 0].scatter(df['trial'], df['score'], c=df['score'], cmap='viridis')
            axes[0, 0].set_title('Optimization Progress')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Accuracy Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Learning rate vs score
            if 'learning_rate' in df.columns:
                axes[0, 1].scatter(df['learning_rate'], df['score'], alpha=0.7)
                axes[0, 1].set_xscale('log')
                axes[0, 1].set_title('Learning Rate vs Score')
                axes[0, 1].set_xlabel('Learning Rate')
                axes[0, 1].set_ylabel('Accuracy Score')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Batch size vs score
            if 'batch_size' in df.columns:
                axes[1, 0].scatter(df['batch_size'], df['score'], alpha=0.7)
                axes[1, 0].set_title('Batch Size vs Score')
                axes[1, 0].set_xlabel('Batch Size')
                axes[1, 0].set_ylabel('Accuracy Score')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Dropout rate vs score
            if 'dropout_rate' in df.columns:
                axes[1, 1].scatter(df['dropout_rate'], df['score'], alpha=0.7)
                axes[1, 1].set_title('Dropout Rate vs Score')
                axes[1, 1].set_xlabel('Dropout Rate')
                axes[1, 1].set_ylabel('Accuracy Score')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
            raise
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to file.
        
        Args:
            filepath: Path to save results
        """
        try:
            results_data = {
                'best_score': self.best_score,
                'best_parameters': self.best_params,
                'all_results': self.results,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def load_results(self, filepath: str) -> None:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to load results from
        """
        try:
            with open(filepath, 'r') as f:
                results_data = json.load(f)
            
            self.best_score = results_data['best_score']
            self.best_params = results_data['best_parameters']
            self.results = results_data['all_results']
            
            logger.info(f"Results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def get_best_model(self) -> Optional[EEGCNNModel]:
        """
        Get the best model from optimization.
        
        Returns:
            Best trained model or None
        """
        return self.best_model
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance based on correlation with score.
        
        Returns:
            Dictionary of parameter importance scores
        """
        try:
            df = self.get_results_dataframe()
            
            if df.empty:
                return {}
            
            importance = {}
            
            # Calculate correlation with score for each parameter
            for col in df.columns:
                if col not in ['trial', 'score', 'timestamp', 'training_time'] and not col.endswith('_values'):
                    if col in df.columns and df[col].dtype in ['int64', 'float64']:
                        correlation = abs(df[col].corr(df['score']))
                        importance[col] = correlation
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {} 