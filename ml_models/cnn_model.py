"""
CNN Model for EEG Time-Series Classification

This module implements a Convolutional Neural Network for classifying
6-state brain-motor activity from EEG signals.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGCNNModel:
    """
    Convolutional Neural Network for EEG time-series classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int = 6,
                 filters: List[int] = [32, 64, 128], kernel_size: int = 3,
                 dropout_rate: float = 0.5, learning_rate: float = 0.001):
        """
        Initialize the CNN model.
        
        Args:
            input_shape: Shape of input data (window_size, n_channels)
            num_classes: Number of output classes
            filters: List of filter sizes for convolutional layers
            kernel_size: Kernel size for convolutional layers
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        self.training_time = None
        
    def build_model(self) -> keras.Model:
        """
        Build the CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        try:
            logger.info("Building CNN model architecture")
            
            # Input layer
            inputs = layers.Input(shape=self.input_shape)
            
            # Add channel dimension for 1D convolution
            x = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(inputs)
            
            # Convolutional layers
            for i, filter_size in enumerate(self.filters):
                if i == 0:
                    x = layers.Conv2D(
                        filters=filter_size,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        activation='relu',
                        padding='same',
                        name=f'conv2d_{i+1}'
                    )(x)
                else:
                    x = layers.Conv2D(
                        filters=filter_size,
                        kernel_size=(self.kernel_size, self.kernel_size),
                        activation='relu',
                        padding='same',
                        name=f'conv2d_{i+1}'
                    )(x)
                
                # Batch normalization
                x = layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
                
                # Max pooling
                x = layers.MaxPooling2D(
                    pool_size=(2, 2),
                    name=f'maxpool_{i+1}'
                )(x)
                
                # Dropout for regularization
                x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            
            # Global average pooling
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
            
            # Dense layers
            x = layers.Dense(128, activation='relu', name='dense_1')(x)
            x = layers.BatchNormalization(name='batch_norm_dense')(x)
            x = layers.Dropout(self.dropout_rate, name='dropout_dense')(x)
            
            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
            
            # Create model
            self.model = models.Model(inputs=inputs, outputs=outputs, name='EEG_CNN')
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=self.learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            logger.info("CNN model built and compiled successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2, verbose: int = 1) -> Dict:
        """
        Train the CNN model.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Validation split ratio
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        try:
            logger.info("Starting CNN model training")
            
            if self.model is None:
                self.build_model()
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    filepath='best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Record training start time
            start_time = datetime.now()
            
            # Train the model
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=verbose
            )
            
            # Record training end time
            end_time = datetime.now()
            self.training_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Training completed in {self.training_time:.2f} seconds")
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test data
            y_test: Test labels
            
        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info("Evaluating CNN model")
            
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
            
            # Get predictions
            y_pred_proba = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
                X_test, y_test, verbose=0
            )
            
            # Classification report
            class_report = classification_report(
                y_true, y_pred, 
                target_names=[f'Class_{i}' for i in range(self.num_classes)],
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            results = {
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'true_labels': y_true,
                'prediction_probabilities': y_pred_proba
            }
            
            logger.info(f"Evaluation completed. Test accuracy: {test_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained yet. Call train() first.")
            
            y_pred_proba = self.model.predict(X)
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            return y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        try:
            if self.model is None:
                raise ValueError("No model to save. Train the model first.")
            
            # Save model
            self.model.save(filepath)
            
            # Save model info
            model_info = {
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'filters': self.filters,
                'kernel_size': self.kernel_size,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'training_time': self.training_time,
                'model_architecture': self.model.get_config()
            }
            
            info_filepath = filepath.replace('.h5', '_info.json')
            with open(info_filepath, 'w') as f:
                json.dump(model_info, f, indent=2, default=str)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self.model = models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot
        """
        try:
            if self.history is None:
                raise ValueError("No training history available. Train the model first.")
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy
            axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
            axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Loss
            axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
            axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Precision
            axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Recall
            axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            raise
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str], save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the plot
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
            raise
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet."
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list) 