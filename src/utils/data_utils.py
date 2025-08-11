import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from loguru import logger

LABEL_COL = 'A_rates'

def load_and_prepare_data(data_path):
    df = pd.read_csv(data_path)
    df["combined_text"] = df["A"] + " " + df["B"]
    
    # Assert based on filename
    if "test" in data_path.lower():
        assert len(df) == 100, f"Expected 100 test problems, but got {len(df)}"
    else:
        assert len(df) == 900, f"Expected 900 training problems, but got {len(df)}"
    
    return df


def split_data(df, val_size=0, test_size=0.1, random_state=42, exp_name=None):
    # First split: test set
    train_val_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    # # Calculate adjusted validation size relative to remaining data
    # val_ratio_adjusted = val_size / (1 - test_size)
    
    # # Second split: validation from remaining data
    # train_data, val_data = train_test_split(train_val_data, test_size=val_ratio_adjusted, random_state=random_state)

    if "tabstar_final" in exp_name:
        # For final training, we use all data as train_data
        train_val_data = df.copy()
        logger.warning(f"Using all data for training in {exp_name}. No validation set created.")
        # log n rows in train_val_data
        logger.info(f"Total training data rows: {len(train_val_data)}")
        return train_val_data, test_data

    # log n rows in train_val_data
    logger.info(f"Total training/validation data rows: {len(train_val_data)}")
    return train_val_data, test_data

def save_preds_to_csv(final_test_data, final_preds, output_dir, exp_name, model_name):
    # if shape is (100, 1), convert to (100,)
    if final_preds.ndim == 2 and final_preds.shape[1] == 1:
        final_preds = final_preds.flatten()
    
    pd.DataFrame({
        "problem_num": final_test_data["problem_num"],
        "prediction": final_preds
    }).to_csv(f"{output_dir}/{exp_name}/{model_name}_predictions.csv", index=False)

def save_model_and_vectorizer(model, vectorizer, output_dir, exp_name, model_name):
    joblib.dump(model, f"{output_dir}/{exp_name}/{model_name}.joblib")
    
    if vectorizer is not None:
        joblib.dump(vectorizer, f"{output_dir}/{exp_name}/{model_name}_vectorizer.joblib")