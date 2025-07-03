"""
Django views for the EEG Dashboard application.
"""

import os
import json
import threading
import time
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.conf import settings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

from .models import TrainingSession, AutoTuningSession, DataFile, ModelComparison
from .forms import (
    TrainingSessionForm, AutoTuningSessionForm, DataFileUploadForm,
    ModelComparisonForm, QuickTrainingForm, AdvancedTrainingForm,
    AutoTuningForm, DataProcessingForm
)

# Import ML modules
from ml_models.data_processor import EEGDataProcessor
from ml_models.cnn_model import EEGCNNModel
from ml_models.auto_tuner import EEGAutoTuner

def index(request):
    """Home page view."""
    # Get recent training sessions
    recent_sessions = TrainingSession.objects.all()[:5]
    recent_tuning = AutoTuningSession.objects.all()[:5]
    
    # Get statistics
    total_sessions = TrainingSession.objects.count()
    completed_sessions = TrainingSession.objects.filter(status='completed').count()
    total_tuning = AutoTuningSession.objects.count()
    completed_tuning = AutoTuningSession.objects.filter(status='completed').count()
    
    context = {
        'recent_sessions': recent_sessions,
        'recent_tuning': recent_tuning,
        'total_sessions': total_sessions,
        'completed_sessions': completed_sessions,
        'total_tuning': total_tuning,
        'completed_tuning': completed_tuning,
    }
    
    return render(request, 'dashboard/index.html', context)

def training_list(request):
    """List all training sessions."""
    sessions = TrainingSession.objects.all()
    
    # Pagination
    paginator = Paginator(sessions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'sessions': page_obj,
    }
    
    return render(request, 'dashboard/training_list.html', context)

def training_detail(request, session_id):
    """Detail view for a training session."""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    # Get training history for plotting
    history_data = session.get_training_history_dict()
    
    # Create plots if data exists
    plots = {}
    if history_data:
        plots = create_training_plots(history_data, session)
    
    context = {
        'session': session,
        'history_data': history_data,
        'plots': plots,
    }
    
    return render(request, 'dashboard/training_detail.html', context)

def training_create(request):
    """Create a new training session."""
    if request.method == 'POST':
        form = TrainingSessionForm(request.POST)
        if form.is_valid():
            session = form.save(commit=False)
            session.status = 'pending'
            session.save()
            
            # Start training in background
            start_training_thread(session.id)
            
            messages.success(request, 'Training session created successfully!')
            return redirect('training_detail', session_id=session.id)
    else:
        form = TrainingSessionForm()
    
    context = {
        'form': form,
        'title': 'Create Training Session',
    }
    
    return render(request, 'dashboard/training_form.html', context)

def quick_training(request):
    """Quick training with minimal parameters."""
    if request.method == 'POST':
        form = QuickTrainingForm(request.POST)
        if form.is_valid():
            # Create training session with default parameters
            session = TrainingSession.objects.create(
                name=f"Quick Training {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                epochs=form.cleaned_data['epochs'],
                batch_size=form.cleaned_data['batch_size'],
                learning_rate=form.cleaned_data['learning_rate'],
                status='pending'
            )
            
            # Start training in background
            start_training_thread(session.id)
            
            messages.success(request, 'Quick training started!')
            return redirect('training_detail', session_id=session.id)
    else:
        form = QuickTrainingForm()
    
    context = {
        'form': form,
        'title': 'Quick Training',
    }
    
    return render(request, 'dashboard/quick_training.html', context)

def auto_tuning_list(request):
    """List all auto-tuning sessions."""
    sessions = AutoTuningSession.objects.all()
    
    # Pagination
    paginator = Paginator(sessions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'sessions': page_obj,
    }
    
    return render(request, 'dashboard/auto_tuning_list.html', context)

def auto_tuning_detail(request, session_id):
    """Detail view for an auto-tuning session."""
    session = get_object_or_404(AutoTuningSession, id=session_id)
    
    context = {
        'session': session,
    }
    
    return render(request, 'dashboard/auto_tuning_detail.html', context)

def auto_tuning_create(request):
    """Create a new auto-tuning session."""
    if request.method == 'POST':
        form = AutoTuningForm(request.POST)
        if form.is_valid():
            session = AutoTuningSession.objects.create(
                name=form.cleaned_data['name'],
                search_type=form.cleaned_data['search_type'],
                max_trials=form.cleaned_data['max_trials'],
                max_time=form.cleaned_data['max_time'],
                status='pending'
            )
            
            # Start auto-tuning in background
            start_auto_tuning_thread(session.id)
            
            messages.success(request, 'Auto-tuning session created successfully!')
            return redirect('auto_tuning_detail', session_id=session.id)
    else:
        form = AutoTuningForm()
    
    context = {
        'form': form,
        'title': 'Create Auto-Tuning Session',
    }
    
    return render(request, 'dashboard/auto_tuning_form.html', context)

def data_upload(request):
    """Upload data files."""
    if request.method == 'POST':
        form = DataFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            data_file = form.save(commit=False)
            data_file.file_size = request.FILES['file'].size
            data_file.file_type = os.path.splitext(request.FILES['file'].name)[1]
            data_file.save()
            
            messages.success(request, 'Data file uploaded successfully!')
            return redirect('data_list')
    else:
        form = DataFileUploadForm()
    
    context = {
        'form': form,
        'title': 'Upload Data File',
    }
    
    return render(request, 'dashboard/data_upload.html', context)

def data_list(request):
    """List uploaded data files."""
    files = DataFile.objects.all()
    
    context = {
        'files': files,
    }
    
    return render(request, 'dashboard/data_list.html', context)

def model_comparison(request):
    """Compare two models."""
    if request.method == 'POST':
        form = ModelComparisonForm(request.POST)
        if form.is_valid():
            comparison = form.save()
            
            # Generate comparison plots
            create_comparison_plots(comparison)
            
            messages.success(request, 'Model comparison created successfully!')
            return redirect('comparison_detail', comparison_id=comparison.id)
    else:
        form = ModelComparisonForm()
    
    context = {
        'form': form,
        'title': 'Compare Models',
    }
    
    return render(request, 'dashboard/model_comparison.html', context)

def comparison_detail(request, comparison_id):
    """Detail view for model comparison."""
    comparison = get_object_or_404(ModelComparison, id=comparison_id)
    
    context = {
        'comparison': comparison,
    }
    
    return render(request, 'dashboard/comparison_detail.html', context)

@csrf_exempt
def training_status(request, session_id):
    """Get training status via AJAX."""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    return JsonResponse({
        'status': session.status,
        'progress': get_training_progress(session),
        'accuracy': session.test_accuracy,
        'loss': session.test_loss,
    })

@csrf_exempt
def auto_tuning_status(request, session_id):
    """Get auto-tuning status via AJAX."""
    session = get_object_or_404(AutoTuningSession, id=session_id)
    
    return JsonResponse({
        'status': session.status,
        'best_score': session.best_score,
        'total_trials': session.total_trials,
        'total_time': session.total_time,
    })

def download_model(request, session_id):
    """Download trained model."""
    session = get_object_or_404(TrainingSession, id=session_id)
    
    if not session.model_file or not os.path.exists(session.model_file):
        messages.error(request, 'Model file not found.')
        return redirect('training_detail', session_id=session_id)
    
    with open(session.model_file, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename="model_{session_id}.h5"'
        return response

def download_results(request, session_id):
    """Download auto-tuning results."""
    session = get_object_or_404(AutoTuningSession, id=session_id)
    
    if not session.results_file or not os.path.exists(session.results_file):
        messages.error(request, 'Results file not found.')
        return redirect('auto_tuning_detail', session_id=session_id)
    
    with open(session.results_file, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/json')
        response['Content-Disposition'] = f'attachment; filename="results_{session_id}.json"'
        return response

# Background task functions
def start_training_thread(session_id):
    """Start training in a background thread."""
    def train_model():
        try:
            session = TrainingSession.objects.get(id=session_id)
            session.status = 'running'
            session.save()
            
            # Prepare data
            data_processor = EEGDataProcessor()
            data_dict = data_processor.prepare_data('dummy_data.csv')  # Use dummy data for now
            
            # Create and train model
            input_shape = data_dict['X_train'].shape[1:]
            model = EEGCNNModel(
                input_shape=input_shape,
                num_classes=6,
                filters=session.get_filters_list(),
                kernel_size=session.kernel_size,
                dropout_rate=session.dropout_rate,
                learning_rate=session.learning_rate
            )
            
            # Train model
            history = model.train(
                data_dict['X_train'], data_dict['y_train'],
                data_dict['X_test'], data_dict['y_test'],
                epochs=session.epochs,
                batch_size=session.batch_size
            )
            
            # Evaluate model
            results = model.evaluate(data_dict['X_test'], data_dict['y_test'])
            
            # Save model
            model_path = os.path.join(settings.MODEL_SAVE_PATH, f'model_{session_id}.h5')
            model.save_model(model_path)
            
            # Create plots
            history_plot_path = os.path.join(settings.PLOT_SAVE_PATH, f'history_{session_id}.png')
            model.plot_training_history(save_path=history_plot_path)
            
            conf_matrix_path = os.path.join(settings.PLOT_SAVE_PATH, f'confusion_{session_id}.png')
            model.plot_confusion_matrix(
                results['confusion_matrix'],
                list(settings.EEG_LABELS.values()),
                save_path=conf_matrix_path
            )
            
            # Update session
            session.status = 'completed'
            session.training_accuracy = history['accuracy'][-1]
            session.validation_accuracy = history['val_accuracy'][-1]
            session.test_accuracy = results['test_accuracy']
            session.training_loss = history['loss'][-1]
            session.validation_loss = history['val_loss'][-1]
            session.test_loss = results['test_loss']
            session.training_time = model.training_time
            session.model_file = model_path
            session.history_plot = history_plot_path
            session.confusion_matrix_plot = conf_matrix_path
            session.set_training_history_dict(history)
            session.save()
            
        except Exception as e:
            session = TrainingSession.objects.get(id=session_id)
            session.status = 'failed'
            session.error_message = str(e)
            session.save()
    
    thread = threading.Thread(target=train_model)
    thread.daemon = True
    thread.start()

def start_auto_tuning_thread(session_id):
    """Start auto-tuning in a background thread."""
    def auto_tune():
        try:
            session = AutoTuningSession.objects.get(id=session_id)
            session.status = 'running'
            session.save()
            
            # Prepare data
            data_processor = EEGDataProcessor()
            data_dict = data_processor.prepare_data('dummy_data.csv')
            
            # Create auto-tuner
            tuner = EEGAutoTuner(data_dict, session.max_trials, session.max_time)
            
            # Run optimization
            if session.search_type == 'grid_search':
                results = tuner.grid_search()
            else:
                results = tuner.random_search()
            
            # Save results
            results_path = os.path.join(settings.PLOT_SAVE_PATH, f'results_{session_id}.json')
            tuner.save_results(results_path)
            
            # Create optimization plot
            plot_path = os.path.join(settings.PLOT_SAVE_PATH, f'optimization_{session_id}.png')
            tuner.plot_optimization_history(save_path=plot_path)
            
            # Update session
            session.status = 'completed'
            session.best_score = results['best_score']
            session.set_best_parameters_dict(results['best_parameters'])
            session.total_trials = results['total_trials']
            session.total_time = results['total_time']
            session.results_file = results_path
            session.optimization_plot = plot_path
            session.save()
            
        except Exception as e:
            session = AutoTuningSession.objects.get(id=session_id)
            session.status = 'failed'
            session.error_message = str(e)
            session.save()
    
    thread = threading.Thread(target=auto_tune)
    thread.daemon = True
    thread.start()

# Utility functions
def create_training_plots(history_data, session):
    """Create training plots."""
    plots = {}
    
    try:
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        plt.plot(history_data.get('accuracy', []), label='Training Accuracy')
        plt.plot(history_data.get('val_accuracy', []), label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plots['accuracy'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(history_data.get('loss', []), label='Training Loss')
        plt.plot(history_data.get('val_loss', []), label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plots['loss'] = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
    except Exception as e:
        print(f"Error creating plots: {e}")
    
    return plots

def create_comparison_plots(comparison):
    """Create comparison plots."""
    try:
        model1 = comparison.model1
        model2 = comparison.model2
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Metrics comparison
        metrics = ['Accuracy', 'Loss']
        model1_scores = [model1.test_accuracy, model1.test_loss]
        model2_scores = [model2.test_accuracy, model2.test_loss]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, model1_scores, width, label=model1.name, alpha=0.8)
        plt.bar(x + width/2, model2_scores, width, label=model2.name, alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(settings.PLOT_SAVE_PATH, f'comparison_{comparison.id}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Update comparison
        comparison.comparison_plot = plot_path
        comparison.save()
        
    except Exception as e:
        print(f"Error creating comparison plots: {e}")

def get_training_progress(session):
    """Get training progress percentage."""
    if session.status == 'completed':
        return 100
    elif session.status == 'running':
        # Estimate progress based on time (rough estimate)
        return 50  # Placeholder
    else:
        return 0
