from src.utils.vectorizer import get_vectorizer, embed_column
from src.utils.data_utils import LABEL_COL
import pandas as pd
from loguru import logger
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap # pip install umap-learn
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def _embed_A_minus_B(train_data, test_data, final_test_data, embed_model):
    # embed A
    A_train = embed_column(train_data, "A", embed_model)
    A_test = embed_column(test_data, "A", embed_model)
    A_final_test = embed_column(final_test_data, "A", embed_model)
    # embed B
    B_train = embed_column(train_data, "B", embed_model)
    B_test = embed_column(test_data, "B", embed_model)
    B_final_test = embed_column(final_test_data, "B", embed_model)

    # create new col that is A minus B
    A_minus_B_train = A_train - B_train
    A_minus_B_test = A_test - B_test
    A_minus_B_final_test = A_final_test - B_final_test
    return A_minus_B_train, A_minus_B_test, A_minus_B_final_test
    

def process_data(train_data, test_data, final_test_data, vectorize_type: str, enrich_type: str, embedding_type: str, baseline_type: str, ensemble_type: str, embed_model='all-MiniLM-L6-v2'):
    
    if baseline_type == "length":
        # add length of A and B as features
        train_data["A_length"] = train_data["A"].apply(lambda x: len(x.split()))
        train_data["B_length"] = train_data["B"].apply(lambda x: len(x.split()))
        train_data["length_diff"] = train_data["A_length"] - train_data["B_length"]
        test_data["A_length"] = test_data["A"].apply(lambda x: len(x.split()))
        test_data["B_length"] = test_data["B"].apply(lambda x: len(x.split()))
        test_data["length_diff"] = test_data["A_length"] - test_data["B_length"]
        final_test_data["A_length"] = final_test_data["A"].apply(lambda x: len(x.split()))
        final_test_data["B_length"] = final_test_data["B"].apply(lambda x: len(x.split()))
        final_test_data["length_diff"] = final_test_data["A_length"] - final_test_data["B_length"]
        
        # select cols
        selected_columns = ["A_length", "B_length", "length_diff"]
        X_train = train_data[selected_columns]
        X_test = test_data[selected_columns]
        X_final_test = final_test_data[selected_columns]
        
        # log n columns
        logger.info(f"Number of columns after length enrichment: {X_train.shape[1]}")
        
        return X_train, X_test, X_final_test, None
    
    if vectorize_type is not None:
        if vectorize_type == "tfidf_by_A":
            vectorizer = get_vectorizer()
            col_to_vectorize = "A"
        elif vectorize_type == "tfidf_by_B":
            vectorizer = get_vectorizer()
            col_to_vectorize = "B"
        elif vectorize_type == "tfidf_by_A_and_B":
            vectorizer = get_vectorizer()
            col_to_vectorize = "combined_text"
            
        X_train = vectorizer.fit_transform(train_data[col_to_vectorize])
        X_test = vectorizer.transform(test_data[col_to_vectorize])
        X_final_test = vectorizer.transform(final_test_data[col_to_vectorize])
            
        return X_train, X_test, X_final_test, vectorizer
    
    elif embedding_type == "A_minus_B" and enrich_type is None:
        A_minus_B_train, A_minus_B_test, A_minus_B_final_test = _embed_A_minus_B(train_data, test_data, final_test_data, embed_model=embed_model)
        return A_minus_B_train, A_minus_B_test, A_minus_B_final_test, None

    elif embedding_type is not None and embedding_type != "A_minus_B":
        if embedding_type == "by_A":
            col_to_embed = "A"
        elif embedding_type == "by_B":
            col_to_embed = "B"
        elif embedding_type == "by_A_and_B":
            col_to_embed = "combined_text"

        X_train = embed_column(train_data, col_to_embed, embed_model=embed_model)
        X_test = embed_column(test_data, col_to_embed, embed_model=embed_model)
        X_final_test = embed_column(final_test_data, col_to_embed, embed_model=embed_model)

        return X_train, X_test, X_final_test, None
    
    elif enrich_type == "A_B_only":
        # keep only A and B columns
        selected_columns = ["A", "B"]
        train_data = train_data[selected_columns]
        test_data = test_data[selected_columns]
        final_test_data = final_test_data[selected_columns]
        
        # log num of columns
        logger.info(f"Number of columns after enrichment: {len(train_data.columns)}")
        
        return train_data, test_data, final_test_data, None
    
    elif enrich_type is not None and "key_words" in enrich_type:
        # embed A and B, and measure for each embedding the cosine similiarity with the key words embeddings
        
        # define key words
        key_words = [
            # safety & certainty
            "safe", "risk-free", "guaranteed", "certain", "assured", "predictable", "secure",
            "no surprises", "reliable", "consistent", "stable", "unchanged", "no loss", "steady",
            "precise planning", "free from worry", "balanced", "peace of mind", "oasis of calm",

            # risk & uncertainty
            "risky", "dangerous", "unpredictable", "gamble", "chance", "volatile", "uncertain",
            "fluctuating", "may result", "could lead", "if unlucky", "rare visitor", "embrace risk",
            "not insignificant chance", "glimpse of fortune", "no reprieve", "depletion",

            # loss-oriented
            "loss", "negative consequence", "downward path", "worse outcome", "minor setbacks",
            "drain", "certain negative result", "bitter taste", "no gain", "small loss", 
            "disappointment", "depleting", "accepting no change", "assured loss",

            # gain-oriented
            "gain", "win", "reward", "positive", "victory", "high reward", "boost", "rich rewards",
            "upper hand", "taste of victory", "significant boost", "holds appeal", "outperforms", "success",

            # comparison & framing
            "compared to", "than its counterpart", "more often than not", "best", "better outcome",
            "wins", "flaunts", "outshines",

            # optimism
            "optimistic", "hopeful", "promising", "encouraging", "uplifting", "bright future",
            "positive outlook", "silver lining", "good fortune", "lucky", "opportunity",
            "fortunate", "win big", "rewarding", "confidence", "high hopes",

            # pessimism
            "pessimistic", "hopeless", "unfavorable", "grim", "discouraging", "bad outcome",
            "bleak", "letdown", "bound to fail", "inevitable loss", "doom", "dark path",

            # fear
            "fear", "anxious", "worried", "alarming", "panic", "too risky", "threat", 
            "afraid", "frightening", "dread", "uncertainty looms", "not worth the risk",

            # bravery
            "brave", "bold", "courageous", "daring", "adventurous", "heroic", 
            "go for it", "step into the unknown", "fearless", "no guts no glory",
            "fortune favors the bold", "embrace the unknown", "stand tall"
        ]
        
        if "10_key_words" in enrich_type:
            # use only 10 key words
            key_words = ["safe", "risky", "loss", "gain", "win", "optimistic", "pessimistic", "fear", "brave", "hopeful"]
        
        if "risk_key_words" in enrich_type:
            key_words = [# safety & certainty
            "safe", "risk-free", "guaranteed", "certain", "assured", "predictable", "secure",
            "no surprises", "reliable", "consistent", "stable", "unchanged", "no loss", "steady",
            "precise planning", "free from worry", "balanced", "peace of mind", "oasis of calm",

            # risk & uncertainty
            "risky", "dangerous", "unpredictable", "gamble", "chance", "volatile", "uncertain",
            "fluctuating", "may result", "could lead", "if unlucky", "rare visitor", "embrace risk",
            "not insignificant chance", "glimpse of fortune", "no reprieve", "depletion",

            # loss-oriented
            "loss", "negative consequence", "downward path", "worse outcome", "minor setbacks",
            "drain", "certain negative result", "bitter taste", "no gain", "small loss", 
            "disappointment", "depleting", "accepting no change", "assured loss",

            # gain-oriented
            "gain", "win", "reward", "positive", "victory", "high reward", "boost", "rich rewards",
            "upper hand", "taste of victory", "significant boost", "holds appeal", "outperforms", "success",

            # comparison & framing
            "compared to", "than its counterpart", "more often than not", "best", "better outcome",
            "wins", "flaunts", "outshines"
            ]
            
        if "fear_bravery_key_words" in enrich_type:
            key_words = [
            # fear
            "fear", "anxious", "worried", "alarming", "panic", "too risky", "threat", 
            "afraid", "frightening", "dread", "uncertainty looms", "not worth the risk",

            # bravery
            "brave", "bold", "courageous", "daring", "adventurous", "heroic", 
            "go for it", "step into the unknown", "fearless", "no guts no glory",
            "fortune favors the bold", "embrace the unknown", "stand tall"
            ]
            
        if "optimism_key_words" in enrich_type:
            key_words = [
            # optimism
            "optimistic", "hopeful", "promising", "encouraging", "uplifting", "bright future",
            "positive outlook", "silver lining", "good fortune", "lucky", "opportunity",
            "fortunate", "win big", "rewarding", "confidence", "high hopes",

            # pessimism
            "pessimistic", "hopeless", "unfavorable", "grim", "discouraging", "bad outcome",
            "bleak", "letdown", "bound to fail", "inevitable loss", "doom", "dark path"
            ]

        if "not_sure_key_words" in enrich_type:
            key_words = [
            # not sure
            "not sure", "undecided", "uncertain", "hesitant", "doubtful", "conflicted",
            "on the fence", "in two minds", "wavering", "unsure", "ambivalent", "mixed feelings"
            ]
            
        logger.info(f"Enriching data with {len(key_words)} key words: {key_words}")
        
        # embed key words
        key_words_embeddings = embed_column(pd.DataFrame({"text": key_words}), "text", embed_model=embed_model)

        # embed A
        A_train = embed_column(train_data, "A", embed_model=embed_model)
        A_test = embed_column(test_data, "A", embed_model=embed_model)
        A_final_test = embed_column(final_test_data, "A", embed_model=embed_model)

        # embed B
        B_train = embed_column(train_data, "B", embed_model=embed_model)
        B_test = embed_column(test_data, "B", embed_model=embed_model)
        B_final_test = embed_column(final_test_data, "B", embed_model=embed_model)

        # calculate cosine similarity with key words embeddings
        from sklearn.metrics.pairwise import cosine_similarity
        
        A_train_similarities = cosine_similarity(A_train, key_words_embeddings)
        A_test_similarities = cosine_similarity(A_test, key_words_embeddings)
        A_final_test_similarities = cosine_similarity(A_final_test, key_words_embeddings)

        B_train_similarities = cosine_similarity(B_train, key_words_embeddings)
        B_test_similarities = cosine_similarity(B_test, key_words_embeddings)
        B_final_test_similarities = cosine_similarity(B_final_test, key_words_embeddings)

        # create new columns for each key word
        for i, word in enumerate(key_words):
            train_data[f"A_{word}"] = A_train_similarities[:, i]
            test_data[f"A_{word}"] = A_test_similarities[:, i]
            final_test_data[f"A_{word}"] = A_final_test_similarities[:, i]

            train_data[f"B_{word}"] = B_train_similarities[:, i]
            test_data[f"B_{word}"] = B_test_similarities[:, i]
            final_test_data[f"B_{word}"] = B_final_test_similarities[:, i]

        # keep only similarity columns
        selected_columns = [col for col in train_data.columns if col.startswith(("A_", "B_"))]
        if "include_A_B" in enrich_type:
            selected_columns += ["A", "B"]
        train_data = train_data[selected_columns]
        test_data = test_data[selected_columns]
        final_test_data = final_test_data[selected_columns]
        
        train_data = _add_diff_columns(train_data, key_words)
        test_data = _add_diff_columns(test_data, key_words)
        final_test_data = _add_diff_columns(final_test_data, key_words)
        
        if enrich_type == "key_words_only_diff":
            selected_columns = [col for col in train_data.columns if col.endswith("_diff")]
            train_data = train_data[selected_columns]
            test_data = test_data[selected_columns]
            final_test_data = final_test_data[selected_columns]
            
        if enrich_type == "key_words_diff_include_A_B":
            selected_columns = [col for col in train_data.columns if col.endswith("_diff")] + ["A", "B"]
            train_data = train_data[selected_columns]
            test_data = test_data[selected_columns]
            final_test_data = final_test_data[selected_columns]
            
        if embedding_type is not None:
            if embedding_type == "A_minus_B":
                A_minus_B_train, A_minus_B_test, A_minus_B_final_test = _embed_A_minus_B(train_data, test_data, final_test_data, embed_model=embed_model) 
                # convert to pd
                A_minus_B_train = pd.DataFrame(A_minus_B_train)
                A_minus_B_test = pd.DataFrame(A_minus_B_test)
                A_minus_B_final_test = pd.DataFrame(A_minus_B_final_test)
                # set columns names to "dim_0", "dim_1", ...
                columns = [f"dim_{i}" for i in range(A_minus_B_train.shape[1])]
                A_minus_B_train.columns = columns
                A_minus_B_test.columns = columns
                A_minus_B_final_test.columns = columns
                # add A_minus_B columns
                train_data = pd.concat([train_data.reset_index(drop=True), A_minus_B_train], axis=1)
                test_data = pd.concat([test_data.reset_index(drop=True), A_minus_B_test], axis=1)
                final_test_data = pd.concat([final_test_data.reset_index(drop=True), A_minus_B_final_test], axis=1)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
            
        if "A" in train_data.columns and "B" in train_data.columns:
            # rename A and B columns to "Offer A" and "Offer B"
            train_data.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)
            test_data.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)
            final_test_data.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)
        
        # log num of columns
        logger.info(f"Number of columns after enrichment: {len(train_data.columns)}")
        
        # return the modified dataframes
        return train_data, test_data, final_test_data, None
        
        
    elif "key_words" in ensemble_type:
        # take the predictions of a lot of models as features
        
        # list of experiments to include
        enrich_types = [
            "enrich_key_words",
            "enrich_risk_key_words"
        ]
        models = [
            "ridge",
            "xgboost",
            "random_forest",
            "tabpfn",
            "tabstar",
        ]
        
        # create a list to hold the dataframes
        train_dfs = []
        test_dfs = []
        final_test_dfs = []
        
        for enrich_type in enrich_types:
            for model in models:
                # load the predictions
                train_preds_path = f"src/results/{model}_{enrich_type} | {embed_model}/{model}_train_results.csv"
                test_preds_path = f"src/results/{model}_{enrich_type} | {embed_model}/{model}_test_results.csv"
                final_test_preds_path = f"src/results/{model}_{enrich_type} | {embed_model}/{model}_predictions.csv"

                train_preds_df = get_exp_results_file(train_preds_path)
                test_preds_df = get_exp_results_file(test_preds_path)
                final_test_df = get_exp_results_file(final_test_preds_path)
                
                if not train_preds_df.empty:
                    train_dfs.append(train_preds_df['predictions'])
                if not test_preds_df.empty:
                    test_dfs.append(test_preds_df['predictions'])
                if not final_test_df.empty:
                    final_test_dfs.append(final_test_df['prediction'])
                    
        if "embed_A_minus_B" in ensemble_type:
            for model in models:
                # load the predictions
                train_preds_path = f"src/results/{model}_embed_A_minus_B | {embed_model}/{model}_train_results.csv"
                test_preds_path = f"src/results/{model}_embed_A_minus_B | {embed_model}/{model}_test_results.csv"
                final_test_preds_path = f"src/results/{model}_embed_A_minus_B | {embed_model}/{model}_predictions.csv"

                train_preds_df = get_exp_results_file(train_preds_path)
                test_preds_df = get_exp_results_file(test_preds_path)
                final_test_df = get_exp_results_file(final_test_preds_path)

                if not train_preds_df.empty:
                    train_dfs.append(train_preds_df['predictions'])
                if not test_preds_df.empty:
                    test_dfs.append(test_preds_df['predictions'])
                if not final_test_df.empty:
                    final_test_dfs.append(final_test_df['prediction'])
                    
        if "include_A_and_B" in ensemble_type:
            model = "tabstar"
            # load the predictions
            train_preds_path = f"src/results/{model}_A_B_only/{model}_train_results.csv"
            test_preds_path = f"src/results/{model}_A_B_only/{model}_test_results.csv"
            final_test_preds_path = f"src/results/{model}_A_B_only/{model}_predictions.csv"
            
            train_preds_df = get_exp_results_file(train_preds_path)
            test_preds_df = get_exp_results_file(test_preds_path)
            final_test_df = get_exp_results_file(final_test_preds_path)

            if not train_preds_df.empty:
                train_dfs.append(train_preds_df['predictions'])
            if not test_preds_df.empty:
                test_dfs.append(test_preds_df['predictions'])
            if not final_test_df.empty:
                final_test_dfs.append(final_test_df['prediction'])

        # concatenate the predictions
        if train_dfs:
            X_train = pd.concat(train_dfs, axis=1)
            # change columns names to f"pred_{i}" where i is the index of the model
            X_train.columns = [f"pred_{i}" for i in range(X_train.shape[1])]
        else:
            X_train = pd.DataFrame()       
        
        if test_dfs:
            X_test = pd.concat(test_dfs, axis=1)
            # change columns names to f"pred_{i}" where i is the index of the model
            X_test.columns = [f"pred_{i}" for i in range(X_test.shape[1])]
        else:
            X_test = pd.DataFrame()
        
        if final_test_dfs:
            X_final_test = pd.concat(final_test_dfs, axis=1)
            # change columns names to f"pred_{i}" where i is the index of the model
            X_final_test.columns = [f"pred_{i}" for i in range(X_final_test.shape[1])]
        else:
            X_final_test = pd.DataFrame()
            
        if "raw_A_B" in ensemble_type:
            A_B_train = train_data[["A", "B"]]
            A_B_test = test_data[["A", "B"]]
            A_B_final_test = final_test_data[["A", "B"]]
            # concatenate the raw A and B columns
            X_train = pd.concat([X_train, A_B_train.reset_index(drop=True)], axis=1)
            X_test = pd.concat([X_test, A_B_test.reset_index(drop=True)], axis=1)
            X_final_test = pd.concat([X_final_test, A_B_final_test.reset_index(drop=True)], axis=1)
            
            # rename A and B columns to "Offer A" and "Offer B"
            X_train.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)
            X_test.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)
            X_final_test.rename(columns={"A": "Offer A", "B": "Offer B"}, inplace=True)

        # log num of columns
        logger.info(f"Number of columns after ensemble for train: {X_train.shape[1]}")
        logger.info(f"Number of columns after ensemble for test: {X_test.shape[1]}")
        logger.info(f"Number of columns after ensemble for final test: {X_final_test.shape[1]}")
        
        return X_train, X_test, X_final_test, None
    else:
        raise ValueError("No valid processing type provided. Please specify vectorize_type, embedding_type, or enrich_type.")
    
def get_exp_results_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if not df.empty:
            return df
    logger.warning(f"File {file_path} does not exist or is empty. Skipping.")
    return pd.DataFrame()


def _add_diff_columns(df, key_words):
    # Compute all diffs at once into a new DataFrame
    diff_cols = {
        f"{word}_diff": df[f"A_{word}"] - df[f"B_{word}"]
        for word in key_words
        if f"A_{word}" in df.columns and f"B_{word}" in df.columns
    }
    return pd.concat([df, pd.DataFrame(diff_cols, index=df.index)], axis=1)


def _process_and_plot_embeddings(train_data, test_data, final_test_data, embedding_type, embed_model, is_3d, output_dir):
    if embedding_type == "A_minus_B":
        X_train, X_test, X_final_test = _embed_A_minus_B(train_data, test_data, final_test_data, embed_model)

    elif embedding_type is not None and embedding_type != "A_minus_B":
        if embedding_type == "by_A":
            col_to_embed = "A"
        elif embedding_type == "by_B":
            col_to_embed = "B"
        elif embedding_type == "by_A_and_B":
            col_to_embed = "combined_text"

        X_train = embed_column(train_data, col_to_embed, embed_model)
        X_test = embed_column(test_data, col_to_embed, embed_model)
        
    y_train = train_data[LABEL_COL]
    y_test = test_data[LABEL_COL]

    for method in ["pca", "tsne", "umap"]:
        _plot_embeddings(X_train, X_test, y_train, y_test, embedding_type, output_dir, method, embed_model, is_3d)

def _plot_embeddings(X_train, X_test, y_train, y_test, embedding_type, output_dir, method, embed_model, is_3d):
    n_components = 3 if is_3d else 2

    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, init='pca', random_state=42)
    elif method == "umap":
        if umap is None:
            raise ImportError("UMAP not installed. Try `pip install umap-learn`")
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Handle t-SNE separately (no .transform)
    if method == "tsne":
        from numpy import vstack
        X_all = vstack([X_train, X_test])
        X_all_2d = reducer.fit_transform(X_all)
        X_train_2d = X_all_2d[:len(X_train)]
        X_test_2d = X_all_2d[len(X_train):]
    else:
        X_train_2d = reducer.fit_transform(X_train)
        X_test_2d = reducer.transform(X_test)

    # Create subplots
    fig = plt.figure(figsize=(16, 6))
    if is_3d:
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)[1]
        ax1, ax2 = axs

    # Plot train
    scatter_train = ax1.scatter(
        X_train_2d[:, 0], X_train_2d[:, 1], X_train_2d[:, 2] if is_3d else None,
        c=y_train, cmap='coolwarm', alpha=0.7
    )
    ax1.set_title("Train Embeddings")
    ax1.set_xlabel(f"{method.upper()} 1")
    ax1.set_ylabel(f"{method.upper()} 2")
    if is_3d:
        ax1.set_zlabel(f"{method.upper()} 3")
    ax1.grid(True)

    # Plot test
    scatter_test = ax2.scatter(
        X_test_2d[:, 0], X_test_2d[:, 1], X_test_2d[:, 2] if is_3d else None,
        c=y_test, cmap='coolwarm', alpha=0.7
    )
    ax2.set_title("Test Embeddings")
    ax2.set_xlabel(f"{method.upper()} 1")
    ax2.set_ylabel(f"{method.upper()} 2")
    if is_3d:
        ax2.set_zlabel(f"{method.upper()} 3")
    ax2.grid(True)

    # Add colorbar
    if not is_3d:
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(scatter_test, cax=cax)
        cbar.set_label("Label")
    else:
        fig.colorbar(scatter_test, ax=[ax1, ax2], shrink=0.6, label="Label")

    # Title & Save
    fig.suptitle(f"{method.upper()} of Embeddings by Split ({embedding_type}, {embed_model})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(f"{output_dir}/embed_{method}", exist_ok=True)
    if is_3d:
        is_3d_suffix = "_3d"
    else:
        is_3d_suffix = ""
    fig_path = os.path.join(output_dir, f"embed_{method}/embed_{embedding_type}_{embed_model}{is_3d_suffix}.png")
    plt.savefig(fig_path)
    plt.close()

def _validate_df_type(X_train, X_test, X_final_test, model_name, vectorize_type, embedding_type, target_dim=500):
    if model_name in ["tabpfn", "tabstar"] and (vectorize_type is not None or embedding_type is not None):
        if vectorize_type is not None:
            # Convert sparse matrices to dense
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            X_final_test = X_final_test.toarray()
            logger.info("Converted sparse matrices to dense for TabPFN or TabStar model.")
        
        # reduce dimension to 500 if needed
        if X_train.shape[1] > 500:
            # if pandas
            if isinstance(X_train, pd.DataFrame):
                X_train, X_test, X_final_test = do_PCA_for_df(X_train, X_test, X_final_test, target_dim=target_dim)
            else:
                X_train, X_test, X_final_test = do_PCA_for_numpy_array(X_train, X_test, X_final_test, target_dim=target_dim)
            
        # if tabstar then convert to pandas DataFrame
        if model_name == "tabstar":
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
            X_final_test = pd.DataFrame(X_final_test)
            logger.info("Converted sparse matrices to pandas DataFrame for TabStar model.")
            
            # set columns names to "dim_0", "dim_1", ..., "dim_499"
            columns = [f"dim_{i}" for i in range(X_train.shape[1])]
            X_train.columns = columns
            X_test.columns = columns
            X_final_test.columns = columns

    return X_train, X_test, X_final_test

def do_PCA_for_df(X_train, X_test, X_final_test, target_dim):
    from sklearn.decomposition import PCA
    # Detect string (or categorical) columns
    str_cols = X_train.select_dtypes(include=['object', 'string']).columns
    k_str_cols = len(str_cols)
    
    logger.info(f"Found {k_str_cols} string columns: {list(str_cols)}")

    # Separate numeric and string parts
    X_train_str = X_train[str_cols].reset_index(drop=True)
    X_test_str = X_test[str_cols].reset_index(drop=True)
    X_final_test_str = X_final_test[str_cols].reset_index(drop=True)

    X_train_num = X_train.drop(columns=str_cols)
    X_test_num = X_test.drop(columns=str_cols)
    X_final_test_num = X_final_test.drop(columns=str_cols)
    
    pca_components = max(target_dim - k_str_cols, 1)

    # Apply PCA only to numeric features
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_num)
    X_test_pca = pca.transform(X_test_num)
    X_final_test_pca = pca.transform(X_final_test_num)

    # Convert PCA outputs to DataFrames for easy concat
    pca_cols = [f"pca_{i}" for i in range(pca_components)]
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_cols)
    X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols)
    X_final_test_pca_df = pd.DataFrame(X_final_test_pca, columns=pca_cols)

    # Concatenate PCA features with original string columns
    X_train = pd.concat([X_train_pca_df, X_train_str], axis=1)
    X_test = pd.concat([X_test_pca_df, X_test_str], axis=1)
    X_final_test = pd.concat([X_final_test_pca_df, X_final_test_str], axis=1)
    
    # log n cols
    logger.info(f"Reduced dimensions to {X_train.shape[1]} columns after PCA, including {k_str_cols} string columns.")

    return X_train, X_test, X_final_test

def do_PCA_for_numpy_array(X_train, X_test, X_final_test, target_dim):
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=target_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_final_test_pca = pca.transform(X_final_test)
    
    logger.info(f"Reduced dimensions to {X_train_pca.shape[1]} columns after PCA.")
    
    return X_train_pca, X_test_pca, X_final_test_pca
    
    