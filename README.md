# 🧠 Real-Time EEG Classification and Visualization Dashboard

A Django-based dashboard for classifying 6-state brain-motor activity using Convolutional Neural Networks (CNNs) with real-time visualization capabilities.

## 🎯 Project Overview

This system classifies EEG signals into 6 mental/physical states:
- Motor imagery onset
- Other thoughts (non-motor)
- Left hand movement
- Right hand movement
- Left hand resting
- Right hand resting

## 🚀 Features

- **Interactive CNN Training**: Adjust hyperparameters through the dashboard
- **Real-time Visualization**: Dynamic plots for accuracy, loss, and confusion matrices
- **Auto-tuning**: Automatic hyperparameter optimization
- **Data Preprocessing**: Built-in EEG signal filtering and normalization
- **Model Comparison**: Compare manual vs auto-tuned results

## 🛠️ Technology Stack

- **Backend**: Django 4.2.7, Python 3
- **ML/AI**: TensorFlow 2.15.0, Keras
- **Frontend**: Django Templates, Bootstrap 5, Plotly
- **Data Processing**: MNE, NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "Brain signal Classification Dashboard"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

6. **Access the dashboard**:
   Open your browser and go to `http://127.0.0.1:8000`

## 📊 Dataset

The system is designed to work with EEG datasets from OpenNeuro. The current implementation includes:
- Support for the ds002338 dataset
- Automatic data preprocessing and segmentation
- 6-state classification labels

## 🎮 Usage

1. **Upload Data**: Use the data upload interface or place your EEG data in the `data/` directory
2. **Configure Parameters**: Adjust CNN hyperparameters through the dashboard
3. **Train Model**: Click "Train Model" to start training with your parameters
4. **Auto-tune**: Use the "Auto-tune" button for automatic hyperparameter optimization
5. **Visualize Results**: View training progress, accuracy plots, and confusion matrices

## 📁 Project Structure

```
Brain signal Classification Dashboard/
├── eeg_dashboard/          # Main Django project
├── dashboard/              # Main app
│   ├── models.py          # Database models
│   ├── views.py           # View logic
│   ├── forms.py           # Forms for parameter input
│   └── templates/         # HTML templates
├── ml_models/             # Machine learning modules
│   ├── cnn_model.py       # CNN architecture
│   ├── data_processor.py  # Data preprocessing
│   └── auto_tuner.py      # Hyperparameter optimization
├── static/                # Static files (CSS, JS, images)
├── media/                 # Uploaded files and generated plots
├── data/                  # EEG dataset storage
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

Key configuration options in `settings.py`:
- `DATA_DIR`: Directory for EEG datasets
- `MODEL_SAVE_PATH`: Path for saving trained models
- `PLOT_SAVE_PATH`: Directory for generated visualizations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenNeuro for providing the EEG dataset
- TensorFlow/Keras for the deep learning framework
- Django community for the web framework 