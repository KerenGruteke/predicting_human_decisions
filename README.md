# Predicting Human Decisions

A machine learning project for predicting human decision-making behavior in risky choice tasks.

## Dataset

The project uses **900 training problems** and **100 test problems**:

- **Training Set**: 900 decision problems with human choice data
- **Test Set**: 100 decision problems for final evaluation

Each problem presents participants with two options (A and B) described in natural language, and the task is to predict the probability that humans will choose option B.

## Key Features

### 1. Multiple Model Architectures
- **Ridge Regression**: Linear regression with L2 regularization
- **XGBoost**: Gradient boosting framework
- **Random Forest**: Ensemble method using multiple decision trees
- **TabPFN**: Prior-data fitted networks for tabular data
- **TabStar**: State-of-the-art tabular deep learning model

### 2. Features Approaches
- **TF-IDF Vectorization**: Traditional text feature extraction
- **Sentence Embeddings**: Using pre-trained models (all-MiniLM-L6-v2, all-mpnet-base-v2, all-distilroberta-v1)
- **Enriched Data using Cosine Similarity**: Enhancing features by measuring similarity between text embeddings of key words and the two options (A and B).
- **LLM Annotations**: Features generated using Gemini and Llama models
- **Ensemble Models**: Combining predictions from multiple models to improve performance.


## Project Structure

```
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── text_task/          # Main dataset (900 train + 100 test problems)
│   ├── models/                 # Model implementations
│   ├── run/                   # Pipeline execution scripts
│   ├── utils/                 # Utility functions
│   ├── annotate/              # LLM annotation and enrichment
│   └── results/               # Experimental results and plots
├── environment.yml            # Conda environment specification
├── pyproject.toml            # Python project configuration
└── README.md                 # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KerenGruteke/predicting_human_decisions.git
   cd predicting_human_decisions
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate human_decisions
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Pipeline Execution

Run the main pipeline with different model configurations:

```python
from src.run.run_pipeline import run_pipeline

# Basic random forest with embeddings
run_pipeline(
    exp_name="rf_embeddings", 
    model_name="random_forest", 
    embedding_type="embed_A_minus_B",
    embed_model="all-mpnet-base-v2"
)

# TabStar with LLM annotations
run_pipeline(
    exp_name="tabstar_llm", 
    model_name="tabstar", 
    enrich_type="llm_annotations_values_and_prob"
)
```

## Model Performance

The project evaluates models using Mean Squared Error (MSE) between predicted and actual human choice probabilities. Results are logged and visualized automatically using the `src/utils/mse_logger.py` script.

## Data Format

Each decision problem consists of:
- `problem_num`: Unique identifier
- `A`: Textual description of option A
- `B`: Textual description of option B  
- `A_rates`: Empirical probability of choosing option A (ground truth)

Example:
```csv
problem_num,A,B,A_rates
M568,"A safe bet that champions gains over losses","Navigate a mostly downward path",0.92
```

## Results and Visualization

Results are automatically saved to `src/results/` including:
- MSE logs across all experiments
- Prediction vs. true value scatter plots
- Embedding visualizations (PCA, t-SNE, UMAP)
