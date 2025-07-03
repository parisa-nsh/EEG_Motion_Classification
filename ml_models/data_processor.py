"""
EEG Data Processor Module

This module handles the preprocessing of EEG signals including:
- Loading and parsing EEG data
- Filtering and normalization
- Segmentation into windows
- Label encoding
"""

import numpy as np
import pandas as pd
import mne
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDataProcessor:
    """
    A class to handle EEG data preprocessing and preparation for CNN training.
    """
    
    def __init__(self, sampling_rate: int = 1000, window_size: float = 2.0, 
                 overlap: float = 0.5, filter_low: float = 1.0, 
                 filter_high: float = 40.0):
        """
        Initialize the EEG data processor.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            window_size: Window size for segmentation in seconds
            overlap: Overlap between windows (0.0 to 1.0)
            filter_low: Low frequency cutoff for bandpass filter
            filter_high: High frequency cutoff for bandpass filter
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.window_samples = int(window_size * sampling_rate)
        self.step_samples = int(self.window_samples * (1 - overlap))
        
        # Initialize scalers
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # EEG channel names (standard 10-20 system)
        self.channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
        ]
        
    def load_eeg_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG data from file.
        
        Args:
            file_path: Path to the EEG data file
            
        Returns:
            Tuple of (eeg_data, labels)
        """
        try:
            # For now, we'll create synthetic data for demonstration
            # In a real implementation, you would load actual EEG files
            logger.info(f"Loading EEG data from {file_path}")
            
            # Create synthetic EEG data for demonstration
            # This simulates 6-state classification data
            n_samples = 10000
            n_channels = len(self.channel_names)
            
            # Generate synthetic EEG signals with different characteristics for each state
            eeg_data = np.zeros((n_samples, n_channels))
            labels = np.zeros(n_samples, dtype=int)
            
            # Define state characteristics
            state_characteristics = {
                0: {'freq': 8, 'amplitude': 1.0},    # Motor imagery onset (alpha)
                1: {'freq': 12, 'amplitude': 0.8},   # Other thoughts (beta)
                2: {'freq': 10, 'amplitude': 1.2},   # Left hand movement
                3: {'freq': 10, 'amplitude': 1.2},   # Right hand movement
                4: {'freq': 6, 'amplitude': 0.6},    # Left hand resting (theta)
                5: {'freq': 6, 'amplitude': 0.6},    # Right hand resting (theta)
            }
            
            samples_per_state = n_samples // 6
            
            for state, char in state_characteristics.items():
                start_idx = state * samples_per_state
                end_idx = start_idx + samples_per_state
                
                # Generate synthetic signal for this state
                t = np.arange(samples_per_state) / self.sampling_rate
                base_signal = char['amplitude'] * np.sin(2 * np.pi * char['freq'] * t)
                
                # Add noise and channel variations
                for ch in range(n_channels):
                    noise = np.random.normal(0, 0.1, samples_per_state)
                    channel_signal = base_signal + noise
                    eeg_data[start_idx:end_idx, ch] = channel_signal
                
                labels[start_idx:end_idx] = state
            
            logger.info(f"Loaded {n_samples} samples with {n_channels} channels")
            return eeg_data, labels
            
        except Exception as e:
            logger.error(f"Error loading EEG data: {e}")
            raise
    
    def apply_filters(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Args:
            eeg_data: Raw EEG data (samples, channels)
            
        Returns:
            Filtered EEG data
        """
        try:
            logger.info("Applying bandpass filter to EEG data")
            
            # Design bandpass filter
            nyquist = self.sampling_rate / 2
            low = self.filter_low / nyquist
            high = self.filter_high / nyquist
            
            # Create Butterworth filter
            b, a = signal.butter(4, [low, high], btype='band')
            
            # Apply filter to each channel
            filtered_data = np.zeros_like(eeg_data)
            for ch in range(eeg_data.shape[1]):
                filtered_data[:, ch] = signal.filtfilt(b, a, eeg_data[:, ch])
            
            logger.info("Bandpass filter applied successfully")
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            raise
    
    def segment_data(self, eeg_data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment EEG data into windows.
        
        Args:
            eeg_data: EEG data (samples, channels)
            labels: Corresponding labels
            
        Returns:
            Tuple of (segmented_data, segmented_labels)
        """
        try:
            logger.info("Segmenting EEG data into windows")
            
            n_samples, n_channels = eeg_data.shape
            n_windows = (n_samples - self.window_samples) // self.step_samples + 1
            
            segmented_data = np.zeros((n_windows, self.window_samples, n_channels))
            segmented_labels = np.zeros(n_windows, dtype=int)
            
            for i in range(n_windows):
                start_idx = i * self.step_samples
                end_idx = start_idx + self.window_samples
                
                if end_idx <= n_samples:
                    segmented_data[i] = eeg_data[start_idx:end_idx]
                    # Use the most common label in the window
                    window_labels = labels[start_idx:end_idx]
                    segmented_labels[i] = np.bincount(window_labels).argmax()
            
            logger.info(f"Created {n_windows} windows of {self.window_samples} samples each")
            return segmented_data, segmented_labels
            
        except Exception as e:
            logger.error(f"Error segmenting data: {e}")
            raise
    
    def normalize_data(self, eeg_data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize EEG data using StandardScaler.
        
        Args:
            eeg_data: EEG data to normalize
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized EEG data
        """
        try:
            logger.info("Normalizing EEG data")
            
            # Reshape for scaling (flatten channels for each sample)
            original_shape = eeg_data.shape
            if len(original_shape) == 3:
                # (windows, samples, channels) -> (windows * samples, channels)
                reshaped_data = eeg_data.reshape(-1, original_shape[-1])
            else:
                reshaped_data = eeg_data
            
            if fit:
                normalized_data = self.scaler.fit_transform(reshaped_data)
            else:
                normalized_data = self.scaler.transform(reshaped_data)
            
            # Reshape back to original shape
            if len(original_shape) == 3:
                normalized_data = normalized_data.reshape(original_shape)
            
            logger.info("Data normalization completed")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            raise
    
    def encode_labels(self, labels: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Encode labels using LabelEncoder.
        
        Args:
            labels: Labels to encode
            fit: Whether to fit the encoder (True for training data)
            
        Returns:
            Encoded labels
        """
        try:
            logger.info("Encoding labels")
            
            if fit:
                encoded_labels = self.label_encoder.fit_transform(labels)
            else:
                encoded_labels = self.label_encoder.transform(labels)
            
            logger.info(f"Labels encoded: {self.label_encoder.classes_}")
            return encoded_labels
            
        except Exception as e:
            logger.error(f"Error encoding labels: {e}")
            raise
    
    def prepare_data(self, file_path: str, test_size: float = 0.2, 
                    random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            file_path: Path to EEG data file
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train/test data and labels
        """
        try:
            logger.info("Starting complete data preparation pipeline")
            
            # Load data
            eeg_data, labels = self.load_eeg_data(file_path)
            
            # Apply filters
            filtered_data = self.apply_filters(eeg_data)
            
            # Segment data
            segmented_data, segmented_labels = self.segment_data(filtered_data, labels)
            
            # Split data before normalization
            X_train, X_test, y_train, y_test = train_test_split(
                segmented_data, segmented_labels, 
                test_size=test_size, random_state=random_state, 
                stratify=segmented_labels
            )
            
            # Normalize data
            X_train_norm = self.normalize_data(X_train, fit=True)
            X_test_norm = self.normalize_data(X_test, fit=False)
            
            # Encode labels
            y_train_encoded = self.encode_labels(y_train, fit=True)
            y_test_encoded = self.encode_labels(y_test, fit=False)
            
            # Convert to one-hot encoding for CNN
            from tensorflow.keras.utils import to_categorical
            y_train_onehot = to_categorical(y_train_encoded, num_classes=6)
            y_test_onehot = to_categorical(y_test_encoded, num_classes=6)
            
            logger.info("Data preparation pipeline completed successfully")
            
            return {
                'X_train': X_train_norm,
                'X_test': X_test_norm,
                'y_train': y_train_onehot,
                'y_test': y_test_onehot,
                'y_train_original': y_train_encoded,
                'y_test_original': y_test_encoded,
                'label_mapping': dict(zip(range(len(self.label_encoder.classes_)), 
                                        self.label_encoder.classes_))
            }
            
        except Exception as e:
            logger.error(f"Error in data preparation pipeline: {e}")
            raise
    
    def get_data_info(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, any]:
        """
        Get information about the prepared data.
        
        Args:
            data_dict: Dictionary containing prepared data
            
        Returns:
            Dictionary with data information
        """
        info = {
            'train_samples': data_dict['X_train'].shape[0],
            'test_samples': data_dict['X_test'].shape[0],
            'window_size': data_dict['X_train'].shape[1],
            'n_channels': data_dict['X_train'].shape[2],
            'n_classes': data_dict['y_train'].shape[1],
            'sampling_rate': self.sampling_rate,
            'window_duration': self.window_size,
            'overlap': self.overlap,
            'filter_range': f"{self.filter_low}-{self.filter_high} Hz"
        }
        
        return info 