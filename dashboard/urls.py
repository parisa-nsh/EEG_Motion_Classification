"""
URL configuration for the dashboard app.
"""

from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    # Home page
    path('', views.index, name='index'),
    
    # Training sessions
    path('training/', views.training_list, name='training_list'),
    path('training/create/', views.training_create, name='training_create'),
    path('training/quick/', views.quick_training, name='quick_training'),
    path('training/<int:session_id>/', views.training_detail, name='training_detail'),
    path('training/<int:session_id>/status/', views.training_status, name='training_status'),
    path('training/<int:session_id>/download/', views.download_model, name='download_model'),
    
    # Auto-tuning sessions
    path('auto-tuning/', views.auto_tuning_list, name='auto_tuning_list'),
    path('auto-tuning/create/', views.auto_tuning_create, name='auto_tuning_create'),
    path('auto-tuning/<int:session_id>/', views.auto_tuning_detail, name='auto_tuning_detail'),
    path('auto-tuning/<int:session_id>/status/', views.auto_tuning_status, name='auto_tuning_status'),
    path('auto-tuning/<int:session_id>/download/', views.download_results, name='download_results'),
    
    # Data management
    path('data/', views.data_list, name='data_list'),
    path('data/upload/', views.data_upload, name='data_upload'),
    
    # Model comparison
    path('comparison/', views.model_comparison, name='model_comparison'),
    path('comparison/<int:comparison_id>/', views.comparison_detail, name='comparison_detail'),
] 