from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

def get_vectorizer():
    return TfidfVectorizer()

def embed_column(df, column, embed_model):
    """
    Embeds the given column of the DataFrame using a SentenceTransformer model.

    Args:
        df (pd.DataFrame): DataFrame containing the text column.
        column (str): Name of the column to embed.
        model_name (str): Name of the pre-trained sentence-transformers model.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_rows, embedding_dim).
    """
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(df[column].tolist(), show_progress_bar=True)
    return embeddings