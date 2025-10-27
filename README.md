# Federated Learning for Intrusion Detection System (FLIDS)

A comprehensive federated learning system for IoT intrusion detection with SGDClassifier (SVM-like) lightweight models and MLP heavyweight models.

## Project Structure

```
flids/
├── device.py                   # IoT device simulation and log generation
├── generate_datapoints.py      # Convert logs to feature datapoints
├── shuffle_peerdata.py         # Shuffle datapoints for better training
├── models.py                   # Lightweight (SGD) and Heavyweight (MLP) models
├── federated_learning.py       # Federated learning orchestration
├── utils.py                    # Data loading and preprocessing utilities
├── main.py                     # Main training script
├── eda_analysis.py            # Exploratory Data Analysis (NEW)
├── plot_training.py           # Training results visualization (NEW)
├── makefile                    # Build automation
└── README.md                   # This file
```

## Features

### Core Federated Learning
- **Lightweight Models**: SGDClassifier with hinge loss (SVM-like) for resource-constrained devices
- **Heavyweight Models**: Multi-layer Perceptron (MLP) for more capable devices
- **True Incremental Learning**: Both model types support partial_fit for online learning
- **Federated Averaging**: Weight aggregation across local models
- **Bidirectional Knowledge Transfer**: Teacher-student learning between lightweight and heavyweight models

### Analysis & Visualization (NEW)
- **Comprehensive EDA**: Analyze raw device logs and pre-shuffled datapoints
- **Training Visualization**: Loss curves, accuracy plots, and performance metrics
- **Separate Workflow**: Run analysis independently without regenerating data

## Installation

```bash
# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn

# Or use requirements.txt if available
pip install -r requirements.txt
```

## Usage

### Complete Workflow (Data Generation + Training + Analysis)

```bash
# Run everything: generate data, shuffle, train, and analyze
make all
```

### Step-by-Step Workflow

```bash
# 1. Generate device logs and datapoints
make data

# 2. Shuffle datapoints for better training
make shuffle

# 3. Run federated learning training
make run

# 4. Generate training visualizations
make plot

# 5. Perform exploratory data analysis
make eda
```

### Individual Commands

```bash
# Generate logs and datapoints only
python device.py
python generate_datapoints.py

# Shuffle data
python shuffle_peerdata.py

# Run training
python main.py

# Visualize training results (requires training_results.csv)
python plot_training.py

# Perform EDA on logs and datapoints
python eda_analysis.py
```

## Output Files

### Data Files
- `logs/`: Raw device logs (CSV format)
- `data/`: Feature datapoints extracted from logs
- `shuffled_data/`: Shuffled datapoints for training
- `training_results.csv`: Training history and metrics

### Visualization Outputs
- `eda_plots/`:
  - `device_logs_eda.png`: Comprehensive EDA of device logs
  - `shuffled_datapoints_eda.png`: Feature analysis of datapoints
  - `device_logs_summary.txt`: Statistical summary of logs
  - `datapoints_summary.txt`: Statistical summary of datapoints

- `training_plots/`:
  - `comprehensive_training_results.png`: All training metrics
  - `loss_analysis.png`: Loss curves for all models
  - `accuracy_metrics.png`: Accuracy, precision, recall, F1 scores
  - `training_summary.txt`: Text summary of training results

## Key Metrics Tracked

### Model Performance
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Training Metrics
- **Loss**: Approximated as (1 - Accuracy)
- **Knowledge Transfer Agreement**: How well models agree during knowledge transfer
- **Training Samples**: Number of samples processed per round
- **Improvement Rate**: Rate of accuracy/loss change

## Model Architecture

### Lightweight Model (SGDClassifier)
- **Algorithm**: Stochastic Gradient Descent with hinge loss (SVM-like)
- **Parameters**: 
  - Loss: 'hinge' (SVM loss)
  - Penalty: L2 regularization
  - warm_start: True (enables incremental learning)
- **Features**: Supports true incremental learning via partial_fit

### Heavyweight Model (MLPClassifier)
- **Architecture**: (32, 16) hidden layers
- **Activation**: ReLU (default)
- **Optimizer**: Adam
- **Features**: Supports incremental learning via partial_fit

### Global Models
- Mirror architecture of local models for federated averaging
- Participate in bidirectional knowledge transfer

## Federated Learning Process

1. **Local Training**: Each device trains on its local data
2. **Federated Averaging**: Local model weights are aggregated to global model
3. **Knowledge Transfer**: Bidirectional transfer between lightweight and heavyweight global models
4. **Local Update**: Local models updated with weighted combination of local and global weights (15% local + 85% global)
5. **Evaluation**: All models evaluated on validation set

## EDA Visualizations

### Device Logs Analysis
- Total packets per device
- Attack vs normal traffic distribution
- Packet types distribution
- Average packet sizes
- Traffic timeline
- Queue utilization
- Port usage statistics
- Top source-destination pairs

### Datapoints Analysis
- Label distribution per device
- Feature correlation heatmap
- Attack vs normal feature comparison
- Feature distributions over time
- Queue statistics
- TCP vs UDP traffic
- Packet size variance
- Class imbalance analysis

## Training Visualizations

### Loss Analysis
- Lightweight model loss (local & global)
- Heavyweight model loss (local & global)
- Combined loss comparison
- Loss reduction rate

### Accuracy Metrics
- Local models accuracy over time
- Global models accuracy over time
- Precision, recall, F1 scores
- All models comparison
- Cumulative performance

### Additional Plots
- Training samples per round
- Knowledge transfer effectiveness
- Accuracy improvement rate
- Final metrics comparison (bar charts)
- Performance summary tables
- Error distribution (box plots)

## Customization

### Training Parameters
Edit `main.py` to adjust:
- Training duration (total_seconds)
- Number of devices
- Local/global weight mixing ratios

### Model Architecture
Edit `models.py` to modify:
- Hidden layer sizes
- Learning rates
- Regularization parameters

### Data Generation
Edit `device.py` to customize:
- Attack patterns
- Network topology
- Traffic behaviors
- Queue sizes

## Performance Tips

1. **Parallel Processing**: `generate_datapoints.py` uses multiprocessing for faster data generation
2. **Incremental Learning**: Both model types support true incremental learning without retraining from scratch
3. **Cached Data**: After initial data generation, you can run training multiple times without regenerating
4. **Separate Analysis**: EDA and visualization can run independently after training

## Troubleshooting

### Missing Data Files
```bash
# If you see "data files not found", run:
make data
make shuffle
```

### Training Results Not Found
```bash
# If plot_training.py fails, ensure you've run training first:
python main.py
```

### Import Errors
```bash
# Install missing packages:
pip install numpy pandas matplotlib seaborn scikit-learn
```

## License

See LICENSE file for details.

## Citation

If you use this code in your research, please cite appropriately.
