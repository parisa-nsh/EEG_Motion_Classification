"""
Django forms for the EEG Dashboard application.
"""

from django import forms
from django.core.validators import MinValueValidator, MaxValueValidator
from .models import TrainingSession, AutoTuningSession, DataFile, ModelComparison
import json

class TrainingSessionForm(forms.ModelForm):
    """
    Form for creating and editing training sessions.
    """
    
    # Custom fields for better UX
    filters_input = forms.CharField(
        label='Filters (comma-separated)',
        help_text='Enter filter sizes separated by commas (e.g., 32,64,128)',
        initial='32,64,128',
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    class Meta:
        model = TrainingSession
        fields = [
            'name', 'epochs', 'batch_size', 'learning_rate', 
            'kernel_size', 'dropout_rate'
        ]
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'epochs': forms.NumberInput(attrs={'class': 'form-control'}),
            'batch_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'learning_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'}),
            'kernel_size': forms.NumberInput(attrs={'class': 'form-control'}),
            'dropout_rate': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        }
    
    def clean_filters_input(self):
        """Clean and validate filters input."""
        filters_input = self.cleaned_data.get('filters_input')
        try:
            # Parse comma-separated values
            filters = [int(x.strip()) for x in filters_input.split(',') if x.strip()]
            if not filters:
                raise forms.ValidationError("At least one filter size is required.")
            
            # Validate filter sizes
            for filter_size in filters:
                if filter_size <= 0:
                    raise forms.ValidationError("Filter sizes must be positive integers.")
            
            return filters
        except ValueError:
            raise forms.ValidationError("Please enter valid integers separated by commas.")
    
    def save(self, commit=True):
        """Save the form and set filters."""
        instance = super().save(commit=False)
        
        # Set filters from the cleaned input
        filters = self.cleaned_data.get('filters_input')
        if isinstance(filters, list):
            instance.set_filters_list(filters)
        
        if commit:
            instance.save()
        return instance

class AutoTuningSessionForm(forms.ModelForm):
    """
    Form for creating auto-tuning sessions.
    """
    
    class Meta:
        model = AutoTuningSession
        fields = ['name', 'search_type', 'max_trials', 'max_time']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'search_type': forms.Select(attrs={'class': 'form-control'}),
            'max_trials': forms.NumberInput(attrs={'class': 'form-control'}),
            'max_time': forms.NumberInput(attrs={'class': 'form-control'}),
        }
    
    def clean_max_time(self):
        """Convert minutes to seconds for max_time."""
        max_time = self.cleaned_data.get('max_time')
        # Assuming input is in minutes, convert to seconds
        return max_time * 60 if max_time else 3600

class DataFileUploadForm(forms.ModelForm):
    """
    Form for uploading data files.
    """
    
    class Meta:
        model = DataFile
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'file': forms.FileInput(attrs={'class': 'form-control'}),
        }
    
    def clean_file(self):
        """Validate uploaded file."""
        file = self.cleaned_data.get('file')
        if file:
            # Check file size (max 100MB)
            if file.size > 100 * 1024 * 1024:
                raise forms.ValidationError("File size must be less than 100MB.")
            
            # Check file extension
            allowed_extensions = ['.csv', '.txt', '.mat', '.edf', '.set', '.fif']
            file_extension = file.name.lower()
            if not any(file_extension.endswith(ext) for ext in allowed_extensions):
                raise forms.ValidationError(
                    f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
                )
        
        return file

class ModelComparisonForm(forms.ModelForm):
    """
    Form for creating model comparisons.
    """
    
    class Meta:
        model = ModelComparison
        fields = ['name', 'model1', 'model2']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'model1': forms.Select(attrs={'class': 'form-control'}),
            'model2': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Only show completed training sessions
        self.fields['model1'].queryset = TrainingSession.objects.filter(status='completed')
        self.fields['model2'].queryset = TrainingSession.objects.filter(status='completed')
    
    def clean(self):
        """Ensure model1 and model2 are different."""
        cleaned_data = super().clean()
        model1 = cleaned_data.get('model1')
        model2 = cleaned_data.get('model2')
        
        if model1 and model2 and model1 == model2:
            raise forms.ValidationError("Please select two different models for comparison.")
        
        return cleaned_data

class QuickTrainingForm(forms.Form):
    """
    Form for quick training with minimal parameters.
    """
    
    epochs = forms.IntegerField(
        label='Epochs',
        initial=50,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    batch_size = forms.ChoiceField(
        label='Batch Size',
        choices=[(16, '16'), (32, '32'), (64, '64'), (128, '128')],
        initial=32,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    learning_rate = forms.ChoiceField(
        label='Learning Rate',
        choices=[
            (0.0001, '0.0001'),
            (0.001, '0.001'),
            (0.01, '0.01')
        ],
        initial=0.001,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class AdvancedTrainingForm(forms.Form):
    """
    Form for advanced training with all parameters.
    """
    
    name = forms.CharField(
        label='Session Name',
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    epochs = forms.IntegerField(
        label='Epochs',
        initial=50,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    batch_size = forms.IntegerField(
        label='Batch Size',
        initial=32,
        min_value=1,
        max_value=512,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    learning_rate = forms.FloatField(
        label='Learning Rate',
        initial=0.001,
        min_value=0.0001,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'})
    )
    
    kernel_size = forms.IntegerField(
        label='Kernel Size',
        initial=3,
        min_value=1,
        max_value=20,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    dropout_rate = forms.FloatField(
        label='Dropout Rate',
        initial=0.5,
        min_value=0.0,
        max_value=1.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    filters = forms.CharField(
        label='Filters',
        initial='32,64,128',
        help_text='Enter filter sizes separated by commas',
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    def clean_filters(self):
        """Clean and validate filters input."""
        filters_input = self.cleaned_data.get('filters')
        try:
            filters = [int(x.strip()) for x in filters_input.split(',') if x.strip()]
            if not filters:
                raise forms.ValidationError("At least one filter size is required.")
            
            for filter_size in filters:
                if filter_size <= 0:
                    raise forms.ValidationError("Filter sizes must be positive integers.")
            
            return filters
        except ValueError:
            raise forms.ValidationError("Please enter valid integers separated by commas.")

class AutoTuningForm(forms.Form):
    """
    Form for auto-tuning configuration.
    """
    
    name = forms.CharField(
        label='Session Name',
        max_length=200,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    search_type = forms.ChoiceField(
        label='Search Type',
        choices=[
            ('grid_search', 'Grid Search'),
            ('random_search', 'Random Search')
        ],
        initial='random_search',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    
    max_trials = forms.IntegerField(
        label='Maximum Trials',
        initial=20,
        min_value=1,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    max_time = forms.IntegerField(
        label='Maximum Time (minutes)',
        initial=60,
        min_value=1,
        max_value=1440,  # 24 hours
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )

class DataProcessingForm(forms.Form):
    """
    Form for data processing parameters.
    """
    
    window_size = forms.FloatField(
        label='Window Size (seconds)',
        initial=2.0,
        min_value=0.5,
        max_value=10.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    overlap = forms.FloatField(
        label='Overlap',
        initial=0.5,
        min_value=0.0,
        max_value=0.9,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    filter_low = forms.FloatField(
        label='Low Frequency Cutoff (Hz)',
        initial=1.0,
        min_value=0.1,
        max_value=50.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    filter_high = forms.FloatField(
        label='High Frequency Cutoff (Hz)',
        initial=40.0,
        min_value=1.0,
        max_value=100.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    sampling_rate = forms.IntegerField(
        label='Sampling Rate (Hz)',
        initial=1000,
        min_value=100,
        max_value=10000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    ) 
