"""
Django models for the EEG Dashboard application.
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
import json
import os
from datetime import datetime

class TrainingSession(models.Model):
    """
    Model to store training session information.
    """
    
    # Session information
    name = models.CharField(max_length=200, default="Training Session")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Training status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Model configuration
    epochs = models.IntegerField(default=50, validators=[MinValueValidator(1), MaxValueValidator(1000)])
    batch_size = models.IntegerField(default=32, validators=[MinValueValidator(1), MaxValueValidator(512)])
    learning_rate = models.FloatField(default=0.001, validators=[MinValueValidator(0.0001), MaxValueValidator(1.0)])
    kernel_size = models.IntegerField(default=3, validators=[MinValueValidator(1), MaxValueValidator(20)])
    dropout_rate = models.FloatField(default=0.5, validators=[MinValueValidator(0.0), MaxValueValidator(1.0)])
    
    # Filter configuration (stored as JSON)
    filters = models.TextField(default='[32, 64, 128]')  # JSON string
    
    # Training results
    training_accuracy = models.FloatField(null=True, blank=True)
    validation_accuracy = models.FloatField(null=True, blank=True)
    test_accuracy = models.FloatField(null=True, blank=True)
    training_loss = models.FloatField(null=True, blank=True)
    validation_loss = models.FloatField(null=True, blank=True)
    test_loss = models.FloatField(null=True, blank=True)
    
    # Training time
    training_time = models.FloatField(null=True, blank=True)  # in seconds
    
    # File paths
    model_file = models.CharField(max_length=500, null=True, blank=True)
    history_plot = models.CharField(max_length=500, null=True, blank=True)
    confusion_matrix_plot = models.CharField(max_length=500, null=True, blank=True)
    
    # Training history (stored as JSON)
    training_history = models.TextField(null=True, blank=True)  # JSON string
    
    # Error message
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.status} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_filters_list(self):
        """Get filters as a Python list."""
        try:
            return json.loads(self.filters)
        except (json.JSONDecodeError, TypeError):
            return [32, 64, 128]
    
    def set_filters_list(self, filters_list):
        """Set filters from a Python list."""
        self.filters = json.dumps(filters_list)
    
    def get_training_history_dict(self):
        """Get training history as a Python dictionary."""
        try:
            return json.loads(self.training_history) if self.training_history else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_training_history_dict(self, history_dict):
        """Set training history from a Python dictionary."""
        self.training_history = json.dumps(history_dict)
    
    def get_model_config(self):
        """Get model configuration as a dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'filters': self.get_filters_list(),
        }
    
    def is_completed(self):
        """Check if training is completed."""
        return self.status == 'completed'
    
    def is_failed(self):
        """Check if training failed."""
        return self.status == 'failed'
    
    def get_duration(self):
        """Get training duration as a formatted string."""
        if self.training_time:
            minutes = int(self.training_time // 60)
            seconds = int(self.training_time % 60)
            return f"{minutes}m {seconds}s"
        return "N/A"

class AutoTuningSession(models.Model):
    """
    Model to store auto-tuning session information.
    """
    
    # Session information
    name = models.CharField(max_length=200, default="Auto-Tuning Session")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Tuning status
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Tuning configuration
    SEARCH_TYPE_CHOICES = [
        ('grid_search', 'Grid Search'),
        ('random_search', 'Random Search'),
    ]
    search_type = models.CharField(max_length=20, choices=SEARCH_TYPE_CHOICES, default='random_search')
    max_trials = models.IntegerField(default=20, validators=[MinValueValidator(1), MaxValueValidator(1000)])
    max_time = models.IntegerField(default=3600, validators=[MinValueValidator(60), MaxValueValidator(86400)])  # seconds
    
    # Results
    best_score = models.FloatField(null=True, blank=True)
    best_parameters = models.TextField(null=True, blank=True)  # JSON string
    total_trials = models.IntegerField(null=True, blank=True)
    total_time = models.FloatField(null=True, blank=True)  # in seconds
    
    # Results file
    results_file = models.CharField(max_length=500, null=True, blank=True)
    optimization_plot = models.CharField(max_length=500, null=True, blank=True)
    
    # Error message
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.search_type} - {self.status} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_best_parameters_dict(self):
        """Get best parameters as a Python dictionary."""
        try:
            return json.loads(self.best_parameters) if self.best_parameters else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_best_parameters_dict(self, params_dict):
        """Set best parameters from a Python dictionary."""
        self.best_parameters = json.dumps(params_dict)
    
    def is_completed(self):
        """Check if tuning is completed."""
        return self.status == 'completed'
    
    def is_failed(self):
        """Check if tuning failed."""
        return self.status == 'failed'
    
    def get_duration(self):
        """Get tuning duration as a formatted string."""
        if self.total_time:
            minutes = int(self.total_time // 60)
            seconds = int(self.total_time % 60)
            return f"{minutes}m {seconds}s"
        return "N/A"

class DataFile(models.Model):
    """
    Model to store uploaded data file information.
    """
    
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='data_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # File information
    file_size = models.BigIntegerField(null=True, blank=True)  # in bytes
    file_type = models.CharField(max_length=50, null=True, blank=True)
    
    # Data information
    n_samples = models.IntegerField(null=True, blank=True)
    n_channels = models.IntegerField(null=True, blank=True)
    sampling_rate = models.IntegerField(null=True, blank=True)
    
    # Processing status
    PROCESSING_STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS_CHOICES, default='pending')
    
    # Error message
    error_message = models.TextField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.name} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_file_size_mb(self):
        """Get file size in MB."""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
    
    def is_processed(self):
        """Check if file is processed."""
        return self.processing_status == 'completed'
    
    def is_failed(self):
        """Check if processing failed."""
        return self.processing_status == 'failed'

class ModelComparison(models.Model):
    """
    Model to store model comparison results.
    """
    
    name = models.CharField(max_length=200, default="Model Comparison")
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # Models to compare
    model1 = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='comparison_model1')
    model2 = models.ForeignKey(TrainingSession, on_delete=models.CASCADE, related_name='comparison_model2')
    
    # Comparison results
    comparison_plot = models.CharField(max_length=500, null=True, blank=True)
    comparison_results = models.TextField(null=True, blank=True)  # JSON string
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_comparison_results_dict(self):
        """Get comparison results as a Python dictionary."""
        try:
            return json.loads(self.comparison_results) if self.comparison_results else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def set_comparison_results_dict(self, results_dict):
        """Set comparison results from a Python dictionary."""
        self.comparison_results = json.dumps(results_dict)
