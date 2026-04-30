# ECG Heartbeat Classification

This project implements and compares multiple deep learning architectures for automated ECG heartbeat classification on the MIT-BIH arrhythmia database.

## Overview

The project trains and evaluates five different neural network models for classifying ECG heartbeats into five categories: Normal (N), Atrial premature (A), Ventricular premature (V), Left bundle branch block (L), and Right bundle branch block (R).

**Models Implemented:**
- **LSTM**: Recurrent neural network for sequence processing
- **LightCNN**: Lightweight convolutional neural network
- **NM2019**: A reference ECG classification model
- **Transformer**: Transformer-based architecture for ECG signals
- **TransformerCNN**: Hybrid architecture combining Transformer and CNN

## Dataset

The project uses the **MIT-BIH Arrhythmia Database**:
- **Total samples**: 92,192 heartbeat windows
- **Input shape**: (92,192, 300) - 300-point ECG windows  
- **Classes**: 5 heartbeat types
  - Class 0: N - Normal beat
  - Class 1: A - Atrial premature beat
  - Class 2: V - Ventricular premature beat
  - Class 3: L - Left bundle branch block beat
  - Class 4: R - Right bundle branch block beat

### Data Structure

Two NumPy files required in `data/`:
- `dataset_raw.npy`: Shape (92,192, 300) - ECG signal samples
- `labelset_raw.npy`: Shape (92,192,) - Integer class labels {0, 1, 2, 3, 4}

## Project Structure

```
├── train.py                 # Main training script
├── run_parallel.sh         # Bash script for parallel training
├── configs/                 # Model configuration files (YAML)
│   ├── LSTM.yaml
│   ├── LightCNN.yaml
│   ├── NM2019.yaml
│   ├── Transformer.yaml
│   └── TransformerCNN.yaml
├── models/                  # Model architecture implementations
│   ├── LSTM.py
│   ├── LightCNN.py
│   ├── NM2019.py
│   ├── Transformer.py
│   └── TransformerCNN.py
├── util/                    # Utility modules
│   ├── load_data.py        # Data loading and preprocessing
│   ├── load_model.py       # Model utilities
│   └── aug.py              # Data augmentation
├── saved/                   # Saved model checkpoints (organized by model and seed)
├── logs/                    # Training logs and metrics
├── logs_parallel/           # Logs from parallel training
├── best_model/             # Best trained models for each architecture
└── visualization/           # Analysis and visualization scripts
    ├── umap_visualization.py
    ├── classwise_average_saliency.py
    ├── param_mmacs.py
    └── complexity.py
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- scikit-learn
- PyYAML
- umap-learn (for visualization)

### Setup

1. Clone or download the project

2. Install dependencies:
```bash
pip install torch numpy scikit-learn pyyaml umap-learn
```

3. Prepare data:
   - Place `dataset_raw.npy` and `labelset_raw.npy` in the `data/` directory
   - Files should have shapes (92,192, 300) and (92,192,) respectively

## Usage

### Training a Single Model

Train a model using a configuration file:

```bash
python train.py --config configs/Transformer.yaml --seed 0
```

**Command-line Arguments:**
- `--config`: Path to configuration file (default: `configs/NM2019.yaml`)
- `--seed`: Random seed for reproducibility (overrides config)
- `--device`: Device to use, e.g., `cuda:0` or `cpu` (overrides config)
- `--data-dir`: Path to data directory (overrides config)
- `--epochs`: Number of training epochs (overrides config)
- `--batch-size`: Batch size (overrides config)
- `--fold`: Run specific stratified k-fold split (default: all folds)
- `--n-splits`: Number of k-fold splits (overrides config)
- `--all-folds`: Run all folds sequentially in one process

### Training Multiple Models in Parallel

Run all models across multiple seeds:

```bash
bash run_parallel.sh
```

This executes training for all architectures with different random seeds in parallel.

### Configuration Files

Each model has a YAML configuration file specifying:

**Experiment settings:**
- Model name
- Random seed
- Save and log directories

**Data settings:**
- Data directory and file names
- Input dimensions (300 samples, 1 channel)
- Split ratios (80/10/10 train/val/test)
- Normalization method (z-score per sample)
- Number of stratified k-fold splits (default: 5)

**Model settings:**
- Architecture-specific hyperparameters (e.g., hidden size, attention heads)
- Number of classes (5)

**Training settings:**
- Device (cuda or cpu)
- Number of epochs
- Batch size
- Learning rate
- Optimizer
- Loss function

## Training Output

Training saves:
- **Model checkpoints**: `saved/<model_experiment_name>/seed_<seed>/`
- **Training logs**: `logs/<model_experiment_name>/seed_<seed>/`
- **Metrics files**:
  - `train_metrics.json`: Per-epoch training metrics
  - `test_metrics.json`: Evaluation metrics on test set
  - `val_metrics.json`: Validation metrics

**Evaluation Metrics:**
- Accuracy
- Balanced Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Per-class classification report

## Visualization & Analysis

### UMAP Embeddings
Visualize learned feature representations using UMAP:

```bash
python visualization/umap_visualization.py
```

Generates:
- UMAP embeddings for each model
- Low-dimensional representations in `figures/umap_embeddings/`

### Saliency Maps
Analyze which parts of ECG signals contribute most to predictions:

```bash
python visualization/classwise_average_saliency.py
```

Generates class-wise average saliency maps showing important signal regions.

### Model Complexity Analysis

Analyze computational complexity:

```bash
python visualization/param_mmacs.py
python visualization/complexity.py
```

Provides:
- Parameter counts
- FLOPs and MACs
- Runtime analysis
- Model size comparisons

## Model Details

### LSTM
Sequential model using LSTM cells for capturing temporal dependencies in ECG signals.

### LightCNN
Lightweight convolutional architecture optimized for efficiency and performance.

### NM2019
Reference ECG classification model based on published research.

### Transformer
Attention-based architecture for learning long-range dependencies in ECG signals.

### TransformerCNN
Hybrid model combining CNN for local feature extraction and Transformer for global pattern learning.

## Results

Model checkpoints and training results are stored in:
- `best_model/`: Best performing model for each architecture
- `saved/`: All model checkpoints by seed and fold
- `logs/`: Complete training logs and metrics

## Utilities

### Data Loading (`util/load_data.py`)
- Loads and preprocesses ECG data
- Implements stratified k-fold cross-validation
- Applies data normalization (z-score per sample)
- Creates PyTorch DataLoaders

### Model Loading (`util/load_model.py`)
- Instantiates models from configuration
- Counts parameters
- Handles model checkpointing

### Data Augmentation (`util/aug.py`)
- Augmentation strategies for ECG data

## References

This project uses the **MIT-BIH Arrhythmia Database**:
- Moody GB, Mark RG. The impact of the MIT-BIH arrhythmia database. IEEE Engineering in Medicine and Biology Magazine 2001;20(3):45-50.

## License

[Specify your license here]

## Contact

For questions or issues, please contact [your contact information].
