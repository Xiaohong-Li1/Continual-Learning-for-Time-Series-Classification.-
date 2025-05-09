<!-- Automatically generated markdown file -->
# Time Series Classification with Continual Learning

This project focuses on Continual Learning (CL) for Time Series Classification (TSC) using a custom deep learning model, `CNN_BiGRU_Attention`. It provides scripts for standard training, evaluation, and dedicated continual learning experiments on UCR time series datasets, along with several baseline models.

## Features

### 1. Core Model: `CNN_BiGRU_Attention`
   - Located in `model.py`.
   - **Hybrid Architecture**: CNNs + BiGRU + Attention.
   - **Multi-Scale Feature Extraction**: Configurable parallel CNN paths.
   - **Attention Mechanism**: CBAM-like channel and spatial attention.
       - **Attention Weight Export**: Optional output for interpretability (`visualize_attention.py`).
   - **Adaptive Pooling**: Handles variable-length sequences.
   - **Bidirectional GRU**: Captures temporal context.
   - **Configurable Architecture**: Key parameters adjustable via `train.py` arguments.
   - **Multi-Head Classifier**: Dynamic head addition for CL tasks.

### 2. Other Models in `model.py`
   - Includes standard deep learning models for TSC:
     - `StandardLSTM`, `StandardGRU`
     - `CNN`, `MCNN`, `FCN`, `ResNet`
   - Can be trained in standard supervised mode using `train_standard.py`.

### 3. Baseline Implementations
   - **`baseline_arima_features.py`**: Implements classification based on features extracted from ARIMA model **residuals**. 
     - Fits an ARIMA model (default order (1,1,0)) to each time series.
     - Extracts statistical features (mean, std) from the residuals.
     - Uses these features to train a simple classifier (e.g., Logistic Regression).
     - Depends on `statsmodels` and `scikit-learn`.
   - **`baseline_random_forest.py`**: Implements classification using Random Forest on **statistical features**.
     - Extracts features like mean, std, min, max, skewness, kurtosis from each time series.
     - Trains a `sklearn.ensemble.RandomForestClassifier` on these features.
     - Depends on `scikit-learn` and `scipy`.

### 4. Data Handling (`load_timeseries_data.py`)
   - **UCR Dataset Loading**: Loads and preprocesses UCR `.arff` files.
   - **Preprocessing**: Z-score normalization, label encoding.
   - **Model Input Formatting**: Handles univariate/multivariate series and ensures `(batch, channel, sequence_length)` format for models.

### 5. Continual Learning Framework (`train.py`)
   - **Purpose**: Runs **continual learning** experiments, primarily for `CNN_BiGRU_Attention`.
   - **Benchmarks**: UCR Time Series (10 datasets) or SplitMNIST.
   - **CL Strategies**: `Naive`, `EWC`, `Replay`, `LwF`.
   - **Functionality**: Dynamic head addition, comprehensive CL metrics, logging.

### 6. Standard Training Framework (`train_standard.py`)
   - **Purpose**: Standard supervised training/evaluation of **any model** in `model.py` on a **single dataset**.
   - **Use Case**: Obtain single-task baseline performance.
   - **Functionality**: Loads data, trains model, evaluates, saves results (model weights, history CSV, summary JSON).

### 7. Baseline Model Comparisons Strategy
   - The project allows comparing `CNN_BiGRU_Attention` (in CL setting via `train.py`, or standard setting via `train_standard.py`) against:
     - **Implemented Deep Learning Baselines**: `StandardLSTM`, `StandardGRU`, `CNN`, `MCNN`, `FCN`, `ResNet` (trained using `train_standard.py`).
     - **Implemented Feature-Based Baselines**: 
       - ARIMA Residual Features + Classifier (run via `baseline_arima_features.py` - requires script modification for specific datasets).
       - Random Forest + Statistical Features (run via `baseline_random_forest.py` - requires script modification for specific datasets).
     - **Future Baselines**: Extending comparisons to direct ARIMA forecasting for classification (if applicable), other ML models with different feature sets, etc.

### 8. Model Interpretability (`visualize_attention.py`)
   - **Attention Weight Visualization**: For `CNN_BiGRU_Attention`.

### 9. Results Visualization (`visualization_all.py`) - In Progress
   - Generates plots from experiment results.
   - Currently supports loading/plotting training history from CSV.

## Project Structure

```
.
├── dataset/                  # Directory to store UCR datasets
├── model.py                  # Contains CNN_BiGRU_Attention, LSTM, GRU, CNN, FCN, ResNet, MCNN model definitions.
├── load_timeseries_data.py   # Utilities for loading UCR datasets.
├── train.py                  # Main script for CONTINUAL LEARNING experiments.
├── train_standard.py         # Script for STANDARD SUPERVISED training of models in model.py.
├── baseline_arima_features.py # Script for ARIMA Residual Feature + Classifier baseline.
├── baseline_random_forest.py # Script for Random Forest + Statistical Feature baseline.
├── visualize_attention.py    # Script to visualize attention weights of CNN_BiGRU_Attention.
├── visualization_all.py      # Script to generate various plots from experiment results.
├── plots/                    # Default directory where generated plots are saved by visualization_all.py.
├── results_standard/         # Default directory where train_standard.py saves results.
├── README.md                 # This file.
└── (other potential files: requirements.txt, saved_models/, logs/, tb_logs/)
```

## Setup

1.  **Clone the repository (if applicable).**
2.  **Install dependencies.** It's recommended to use a virtual environment.
    ```bash
    pip install torch torchvision torchaudio avalanche-lib numpy pandas scikit-learn matplotlib seaborn arff statsmodels scipy
    ```
    (Added `statsmodels`, `scipy`. Ensure `avalanche-lib` is compatible, e.g., >=0.5.0).
3.  **Download UCR Datasets**: Place the UCR datasets into the `dataset/` directory.

## Usage

### 1. Training Continual Learning Models (`train.py`)
   Use `train.py` for **continual learning** experiments with `CNN_BiGRU_Attention`.
   ```bash
   python train.py --benchmark_name ucr_timeseries --strategies EWC LwF # ... other args ...
   ```

### 2. Training Standard Models (`train_standard.py`)
   Use `train_standard.py` for **standard supervised training** of any model from `model.py` on a single dataset.
   ```bash
   python train_standard.py --model_name lstm --dataset_name ArrowHead # ... other args ...
   ```

### 3. Running Feature-Based Baselines (`baseline_*.py`)
   - **`baseline_arima_features.py`** and **`baseline_random_forest.py`** implement feature-based classification methods.
   - **Current Status**: These scripts currently contain example usage with dummy data within their `if __name__ == '__main__':` blocks.
   - **To use on real data**: You need to modify these scripts:
     1. Add code to load a specific UCR dataset (using `load_timeseries_data.load_ucr_dataset`).
     2. Replace the dummy data (`X_train_dummy`, `y_train_dummy`, etc.) with the loaded real data.
     3. Run the modified script: `python baseline_arima_features.py` or `python baseline_random_forest.py`.
   - **Recommendation**: Consider adding command-line argument parsing (like in `train_standard.py`) to these scripts for easier dataset selection and result saving (see Future Work).

### 4. Visualizing Attention Weights (`visualize_attention.py`)
   Visualize attention for a trained `CNN_BiGRU_Attention` model.
   ```bash
   python visualize_attention.py --model_path <path_to_model.pth> --dataset_name ArrowHead # ... other args ...
   ```

### 5. Visualizing Experiment Results (`visualization_all.py`)
   Generate plots from saved experiment results.
   ```bash
   python visualization_all.py --history_csv <path_to_history.csv> --dataset_name Coffee # ... other args ...
   ```

## Future Work 

*   **Add Command-Line Interface to Baseline Scripts**: Enhance `baseline_arima_features.py` and `baseline_random_forest.py` with `argparse` for dataset selection and result saving.
*   **Implement Data Unification**: Add padding/truncation for varying sequence lengths.
*   **Save/Load Model Configuration**: Store architecture parameters with model weights.
*   **Expand `visualization_all.py`**: Implement loading for all result types (accuracies, CL metrics, etc.) possibly from JSON summaries or logs.
*   **Comprehensive Baseline Comparisons**: 
    *   Systematically run `train_standard.py` for all relevant DL models across datasets.
    *   Run the modified `baseline_*.py` scripts across datasets.
    *   Implement and evaluate direct ARIMA forecasting for classification (if feasible).
    *   Implement and evaluate other ML classifiers with potentially more sophisticated feature engineering (e.g., using `tsfresh`).
*   **Hyperparameter Optimization**: Tune hyperparameters for all models and training modes.
*   **Explore More Advanced CL Algorithms**.
*   **Quantitative CL Metrics**: Enhance CL logging/reporting.
*   **Refined Attention Visualization**. 
