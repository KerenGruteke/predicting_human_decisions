import os
from src.utils.data_utils import load_and_prepare_data, split_data, save_preds_to_csv, save_model_and_vectorizer, LABEL_COL
from src.utils.mse_logger import update_mse_log, plot_mse, plot_mse_by_model_family
from src.models.get_model import get_model
from src.run.evaluator import evaluate_model
from src.run.process_data import process_data, _process_and_plot_embeddings, _validate_df_type
from loguru import logger


TRAIN_PROBLEMS_PATH = 'src/data/text_task/train_problems.csv'
TEST_PROBLEMS_PATH = 'src/data/text_task/test_problems.csv'
OUTPUT_DIR = 'src/results'

def run_pipeline(exp_name, model_name, vectorize_type=None, embedding_type=None, enrich_type=None, embed_model=None, baseline_type=None, ensemble_type=None, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    if embed_model is not None and enrich_type is not None:
        exp_name += f" | {embed_model}"
        
    logger.info(f"Running pipeline for experiment: {exp_name}")

    # Load and prepare data
    df = load_and_prepare_data(TRAIN_PROBLEMS_PATH)
    train_data, test_data = split_data(df)
    final_test_data = load_and_prepare_data(TEST_PROBLEMS_PATH)

    # Process data
    X_train, X_test, X_final_test, vectorizer = process_data(
        train_data=train_data.drop(columns=[LABEL_COL]),
        test_data=test_data.drop(columns=[LABEL_COL]),
        final_test_data=final_test_data,
        vectorize_type=vectorize_type,
        enrich_type=enrich_type,
        embedding_type=embedding_type,
        embed_model=embed_model,
        baseline_type=baseline_type,
        ensemble_type=ensemble_type
    )
    X_train, X_test, X_final_test = _validate_df_type(X_train, X_test, X_final_test, model_name, vectorize_type, embedding_type)
    
    y_train = train_data[LABEL_COL]
    y_test = test_data[LABEL_COL]

    # Train model
    model = get_model(model_name)
    logger.info(f"Training model: {model_name}")
    model.fit(X_train, y_train)

    # Evaluate
    train_mse, _ = evaluate_model(model, X_train, y_train, model_name, "train", output_dir, exp_name, train_data)
    test_mse, _ = evaluate_model(model, X_test, y_test, model_name, "test", output_dir, exp_name, test_data)
    update_mse_log(exp_name, model_name, train_mse, test_mse, output_dir)

    # Predict on final test set
    final_preds = model.predict(X_final_test)
    
    save_preds_to_csv(final_test_data, final_preds, output_dir, exp_name, model_name)
    save_model_and_vectorizer(model, vectorizer, output_dir, exp_name, model_name)

def plot_embeddings(embedding_type, embed_model, is_3d, output_dir=OUTPUT_DIR):
    # Load and prepare data
    df = load_and_prepare_data(TRAIN_PROBLEMS_PATH)
    train_data, test_data = split_data(df)
    final_test_data = load_and_prepare_data(TEST_PROBLEMS_PATH)
    
    _process_and_plot_embeddings(
        train_data=train_data,
        test_data=test_data,
        final_test_data=final_test_data,
        embedding_type=embedding_type,
        embed_model=embed_model,
        is_3d=is_3d,
        output_dir=output_dir
    )


if __name__ == "__main__":
    
    # ---------------------
    # Baseline Models
    # ---------------------
    
    # for baseline_type in ["length"]:
    #     run_pipeline(exp_name="ridge_length", model_name="ridge", baseline_type=baseline_type)
    #     run_pipeline(exp_name="xgboost_length", model_name="xgboost", baseline_type=baseline_type)
    #     run_pipeline(exp_name="random_forest_length", model_name="random_forest", baseline_type=baseline_type)
    #     run_pipeline(exp_name="tabpfn_length", model_name="tabpfn", baseline_type=baseline_type)
    #     run_pipeline(exp_name="tabstar_length", model_name="tabstar", baseline_type=baseline_type)

    # ---------------------
    # TF-IDF Vectorization
    # ---------------------
    
    # ridge
    # run_pipeline(exp_name="ridge_tfidf_by_A", model_name="ridge", vectorize_type="tfidf_by_A")
    # run_pipeline(exp_name="ridge_tfidf_by_B", model_name="ridge", vectorize_type="tfidf_by_B")
    # run_pipeline(exp_name="ridge_tfidf_by_A_and_B", model_name="ridge", vectorize_type="tfidf_by_A_and_B")
    
    # xgboost
    # run_pipeline(exp_name="xgboost_tfidf_by_A", model_name="xgboost", vectorize_type="tfidf_by_A")
    # run_pipeline(exp_name="xgboost_tfidf_by_B", model_name="xgboost", vectorize_type="tfidf_by_B")
    # run_pipeline(exp_name="xgboost_tfidf_by_A_and_B", model_name="xgboost", vectorize_type="tfidf_by_A_and_B")
    
    # random_forest
    # run_pipeline(exp_name="random_forest_tfidf_by_A", model_name="random_forest", vectorize_type="tfidf_by_A")
    # run_pipeline(exp_name="random_forest_tfidf_by_B", model_name="random_forest", vectorize_type="tfidf_by_B")
    # run_pipeline(exp_name="random_forest_tfidf_by_A_and_B", model_name="random_forest", vectorize_type="tfidf_by_A_and_B")
    
    # tabpfn
    # run_pipeline(exp_name="tabpfn_tfidf_by_A", model_name="tabpfn", vectorize_type="tfidf_by_A")
    # run_pipeline(exp_name="tabpfn_tfidf_by_B", model_name="tabpfn", vectorize_type="tfidf_by_B")
    # run_pipeline(exp_name="tabpfn_tfidf_by_A_and_B", model_name="tabpfn", vectorize_type="tfidf_by_A_and_B")
    
    # tabstar
    # run_pipeline(exp_name="tabstar_tfidf_by_A", model_name="tabstar", vectorize_type="tfidf_by_A")
    # run_pipeline(exp_name="tabstar_tfidf_by_B", model_name="tabstar", vectorize_type="tfidf_by_B")
    # run_pipeline(exp_name="tabstar_tfidf_by_A_and_B", model_name="tabstar", vectorize_type="tfidf_by_A_and_B")
    
    # ---------------------
    # Embeddings
    # ---------------------
    
    # by A
    # for embed_model in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1']:
    #     run_pipeline(exp_name=f"ridge_embed_by_A | {embed_model}", model_name="ridge", embedding_type="by_A", embed_model=embed_model)
    #     run_pipeline(exp_name=f"xgboost_embed_by_A | {embed_model}", model_name="xgboost", embedding_type="by_A", embed_model=embed_model)
    #     run_pipeline(exp_name=f"random_forest_embed_by_A |{embed_model}", model_name="random_forest", embedding_type="by_A", embed_model=embed_model)
    #     run_pipeline(exp_name=f"tabpfn_embed_by_A | {embed_model}", model_name="tabpfn", embedding_type="by_A", embed_model=embed_model)
    #     run_pipeline(exp_name=f"tabstar_embed_by_A | {embed_model}", model_name="tabstar", embedding_type="by_A", embed_model=embed_model)
    
    # by B
    # run_pipeline(exp_name="ridge_embed_by_B", model_name="ridge", embedding_type="by_B")
    # run_pipeline(exp_name="xgboost_embed_by_B", model_name="xgboost", embedding_type="by_B")
    # run_pipeline(exp_name="random_forest_embed_by_B", model_name="random_forest", embedding_type="by_B")
    # run_pipeline(exp_name="tabpfn_embed_by_B", model_name="tabpfn", embedding_type="by_B")
    # run_pipeline(exp_name="tabstar_embed_by_B", model_name="tabstar", embedding_type="by_B")
    
    # by A and B
    # run_pipeline(exp_name="ridge_embed_by_A_and_B", model_name="ridge", embedding_type="by_A_and_B")
    # run_pipeline(exp_name="xgboost_embed_by_A_and_B", model_name="xgboost", embedding_type="by_A_and_B")
    # run_pipeline(exp_name="random_forest_embed_by_A_and_B", model_name="random_forest", embedding_type="by_A_and_B")
    # run_pipeline(exp_name="tabpfn_embed_by_A_and_B", model_name="tabpfn", embedding_type="by_A_and_B")
    # run_pipeline(exp_name="tabstar_embed_by_A_and_B", model_name="tabstar", embedding_type="by_A_and_B")
    
    # A minus B
    # for embed_model in ['all-mpnet-base-v2']: #['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1']
        # run_pipeline(exp_name=f"ridge_embed_A_minus_B | {embed_model}", model_name="ridge", embedding_type="A_minus_B", embed_model=embed_model)
        # run_pipeline(exp_name=f"xgboost_embed_A_minus_B | {embed_model}", model_name="xgboost", embedding_type="A_minus_B", embed_model=embed_model)
        # run_pipeline(exp_name=f"random_forest_embed_A_minus_B | {embed_model}", model_name="random_forest", embedding_type="A_minus_B", embed_model=embed_model)
        # run_pipeline(exp_name=f"tabpfn_embed_A_minus_B | {embed_model}", model_name="tabpfn", embedding_type="A_minus_B", embed_model=embed_model)
        # run_pipeline(exp_name=f"tabstar_embed_A_minus_B | {embed_model}", model_name="tabstar", embedding_type="A_minus_B", embed_model=embed_model)
    
    
    # # plot embeddings with PCA
    # is_3d = True  # Change to False for 2D plots
    # plot_embeddings(embedding_type="A_minus_B", embed_model='all-MiniLM-L6-v2', is_3d=is_3d)
    # plot_embeddings(embedding_type="A_minus_B", embed_model='all-mpnet-base-v2', is_3d=is_3d)
    # plot_embeddings(embedding_type="A_minus_B", embed_model='all-distilroberta-v1', is_3d=is_3d)

    # plot_embeddings(embedding_type="by_A", embed_model='all-MiniLM-L6-v2', is_3d=is_3d)
    # plot_embeddings(embedding_type="by_A", embed_model='all-mpnet-base-v2', is_3d=is_3d)
    # plot_embeddings(embedding_type="by_A", embed_model='all-distilroberta-v1', is_3d=is_3d)

    # ---------------------
    # Enrichment with Key Words
    # ---------------------
    # vector size 125 * 3 = 375 (A, B, A-B)
    # for embed_model in ['all-mpnet-base-v2']: # all-MiniLM-L6-v2', 'all-distilroberta-v1', 
    #     for enrich_type in ["key_words", "risk_key_words"]:
    #         run_pipeline(exp_name=f"ridge_enrich_{enrich_type}", model_name="ridge", enrich_type=enrich_type, embed_model=embed_model)
    #         run_pipeline(exp_name=f"xgboost_enrich_{enrich_type}", model_name="xgboost", enrich_type=enrich_type, embed_model=embed_model)
    #         run_pipeline(exp_name=f"random_forest_enrich_{enrich_type}", model_name="random_forest", enrich_type=enrich_type, embed_model=embed_model)
    #         run_pipeline(exp_name=f"tabpfn_enrich_{enrich_type}", model_name="tabpfn", enrich_type=enrich_type, embed_model=embed_model)
    #         run_pipeline(exp_name=f"tabstar_enrich_{enrich_type}", model_name="tabstar", enrich_type=enrich_type, embed_model=embed_model)
            
            # run_pipeline(exp_name=f"tabstar_enrich_{enrich_type}_include_A_B", model_name="tabstar", enrich_type=f"{enrich_type}_include_A_B", embed_model=embed_model)

    # for embed_model in ['all-distilroberta-v1']: # 'all-MiniLM-L6-v2', 'all-mpnet-base-v2'
        # run_pipeline(exp_name="ridge_enrich_10_key_words", model_name="ridge", enrich_type="10_key_words", embed_model=embed_model)
        # run_pipeline(exp_name="xgboost_enrich_10_key_words", model_name="xgboost", enrich_type="10_key_words", embed_model=embed_model)
        # run_pipeline(exp_name="random_forest_enrich_10_key_words", model_name="random_forest", enrich_type="10_key_words", embed_model=embed_model)
        # run_pipeline(exp_name="tabpfn_enrich_10_key_words", model_name="tabpfn", enrich_type="10_key_words", embed_model=embed_model)
        # run_pipeline(exp_name="tabstar_enrich_10_key_words", model_name="tabstar", enrich_type="10_key_words", embed_model=embed_model)

    # run_pipeline(exp_name="ridge_enrich_key_words_only_diff", model_name="ridge", enrich_type="key_words_only_diff")
    # run_pipeline(exp_name="xgboost_enrich_key_words_only_diff", model_name="xgboost", enrich_type="key_words_only_diff")
    # run_pipeline(exp_name="random_forest_enrich_key_words_only_diff", model_name="random_forest", enrich_type="key_words_only_diff")
    # run_pipeline(exp_name="tabpfn_enrich_key_words_only_diff", model_name="tabpfn", enrich_type="key_words_only_diff")
    # run_pipeline(exp_name="tabstar_enrich_key_words_only_diff", model_name="tabstar", enrich_type="key_words_only_diff")
    
    # for embed_model in ['all-distilroberta-v1']: # 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 
        # run_pipeline(exp_name="tabstar_enrich_key_words_include_A_B", model_name="tabstar", enrich_type="key_words_include_A_B", embed_model=embed_model)
        # run_pipeline(exp_name="tabstar_enrich_key_words_diff_include_A_B", model_name="tabstar", enrich_type="key_words_diff_include_A_B", embed_model=embed_model)
        # run_pipeline(exp_name="tabstar_enrich_10_key_words_include_A_B", model_name="tabstar", enrich_type="10_key_words_include_A_B", embed_model=embed_model)

    # for embed_model in ['all-mpnet-base-v2', 'all-distilroberta-v1', 'all-MiniLM-L6-v2']:
        # enrich_type = "not_sure_key_words"
        # run_pipeline(exp_name=f"ridge_enrich_{enrich_type}", model_name="ridge", enrich_type=enrich_type, embed_model=embed_model)
        # run_pipeline(exp_name=f"xgboost_enrich_{enrich_type}", model_name="xgboost", enrich_type=enrich_type, embed_model=embed_model)
        # run_pipeline(exp_name=f"random_forest_enrich_{enrich_type}", model_name="random_forest", enrich_type=enrich_type, embed_model=embed_model)
        # run_pipeline(exp_name=f"tabpfn_enrich_{enrich_type}", model_name="tabpfn", enrich_type=enrich_type, embed_model=embed_model)
        # run_pipeline(exp_name=f"tabstar_enrich_{enrich_type}", model_name="tabstar", enrich_type=enrich_type, embed_model=embed_model)

        # run_pipeline(exp_name="tabstar_enrich_risk_key_words_include_A_B", model_name="tabstar", enrich_type="risk_key_words_include_A_B", embed_model=embed_model)
        # run_pipeline(exp_name=f"tabstar_enrich_optimism_key_words_include_A_B | {embed_model}", model_name="tabstar", enrich_type="optimism_key_words_include_A_B", embed_model=embed_model)
    # run_pipeline(exp_name="tabstar_embed_A_minus_B_enrich_not_sure_key_words_include_A_B", model_name="tabstar", enrich_type="not_sure_key_words_include_A_B", embed_model='all-mpnet-base-v2', embedding_type="A_minus_B")

    # run_pipeline(exp_name="tabstar_A_B_only", model_name="tabstar", enrich_type="A_B_only")

    # ---------------------
    # Ensemble Models
    # ---------------------
    
    # for model_name in ["ridge", "xgboost", "random_forest", "tabpfn", "tabstar"]:
    #     # run_pipeline(exp_name=f"{model_name}_ensemble_key_words", model_name=model_name, ensemble_type="key_words", embed_model="all-mpnet-base-v2")
        # run_pipeline(exp_name=f"{model_name}_ensemble_key_words_and_embed_A_minus_B", model_name=model_name, ensemble_type="key_words_and_embed_A_minus_B", embed_model="all-mpnet-base-v2")
        # run_pipeline(exp_name=f"{model_name}_ensemble_key_words_and_embed_A_minus_B_include_A_and_B", model_name=model_name, ensemble_type="key_words_and_embed_A_minus_B_include_A_and_B", embed_model="all-mpnet-base-v2")
    
    
    # run_pipeline(exp_name="tabstar_ensemble_key_words_and_embed_A_minus_B_include_A_and_B_and_raw_A_B", model_name="tabstar", ensemble_type="key_words_and_embed_A_minus_B_include_A_and_B_and_raw_A_B", embed_model="all-mpnet-base-v2")

    # ---------------------
    # LLMs Annotations Features
    # ---------------------
    
    # for model_name in ["ridge", "xgboost", "random_forest", "tabpfn", "tabstar"]:
    #     run_pipeline(exp_name=f"{model_name}_llm_annotations_values_and_prob", model_name=model_name, enrich_type="llm_annotations_values_and_prob")

    run_pipeline(exp_name="tabstar_gemini_annotations_values_and_prob_include_A_B", model_name="tabstar", enrich_type="gemini_annotations_values_and_prob_include_A_B")
    run_pipeline(exp_name="tabstar_llama_annotations_values_and_prob_include_A_B", model_name="tabstar", enrich_type="llama_annotations_values_and_prob_include_A_B")


    # ---------------------
    # Plot MSE
    # ---------------------
    
    plot_mse(f"{OUTPUT_DIR}/mse_log.csv")
    plot_mse_by_model_family(f"{OUTPUT_DIR}/mse_log.csv")