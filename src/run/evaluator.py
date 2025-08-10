from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X, y, model_name, split_name, output_dir, exp_name, data):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    
    # if dim preds is 2D, flatten it
    if preds.ndim > 1:
        preds = preds.flatten()
    errors = y - preds
    
    # plot predictions vs true values
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(
        y, preds,
        c=errors,
        cmap="coolwarm",
        alpha=0.7,
        edgecolors='k',
        linewidths=0.2
    )
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted (colored by error)")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Prediction Error (y - pred)")

    plt.tight_layout()
    
    # save the plot
    os.makedirs(f"{output_dir}/{exp_name}", exist_ok=True)
    plt.savefig(f"{output_dir}/{exp_name}/{model_name}_{split_name}_pred_vs_true.png")
    plt.close()
    
    # save csv with true, predicted, error, and original data columns from the data ("A", "B" and "problem_num")
    results_df = data.copy()
    results_df['predictions'] = preds
    results_df['error'] = errors
    # select only relevant columns
    results_df = results_df[['problem_num', 'A', 'B', 'A_rates', 'predictions', 'error']].sort_values(by='error', ascending=False)
    results_df.to_csv(f"{output_dir}/{exp_name}/{model_name}_{split_name}_results.csv", index=False)
    
    return mse, preds
