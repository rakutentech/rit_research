import pandas as pd
import numpy as np
from tqdm import tqdm

def convert_to_p_e_given_a(alpha_ae):
    """
    With soft max function, it will transform embedding matrix (alpha_ae)
    into the probability of embedding given item.
    
    Parameters:
        alpha_a_e: 
    """
    alpha_ae_exp = alpha_ae.applymap(np.exp)
    for i in alpha_ae_exp.index:
        alpha_ae_exp.loc[i] /= alpha_ae_exp.loc[i].sum()
    return alpha_ae_exp



def get_embedding_with_LSI(item_features, num_embedding_feature):
    """
    Extracts embeddings from item features using Latent Semantic Indexing (LSI).

    This function applies Singular Value Decomposition (SVD) on the item_features to reduce its dimensionality,
    obtaining a dense representation of items in a lower-dimensional space defined by num_embedding_feature.

    Parameters:
        item_features (DataFrame or ndarray): The feature matrix of the items.
        num_embedding_feature (int): The number of embedding features to extract using LSI.

    Returns:
        DataFrame: A DataFrame containing the item embeddings in the reduced feature space.
    """
    U, S_diags, V_t = np.linalg.svd(item_features, full_matrices=True, compute_uv=True, hermitian=False)
    U_k = pd.DataFrame(U).iloc[:, :num_embedding_feature]
    return U_k


def resample_dataset_with_embed_vector(df, p_e_a, size=100000):
    """
    Resamples the given dataset with embedded vectors according to the provided embedding probabilities.

    This function creates a new dataset where each original item is replaced with a potential embedding
    vector based on the calculated probabilities. It also simulates clicks as a binary outcome based on these probabilities.

    Parameters:
        df (DataFrame): The original dataset with items and clicks.
        p_e_a (DataFrame): A DataFrame containing the probabilities of each embedding given an item.

    Returns:
        DataFrame: A new DataFrame where each item is replaced with an embedding, and clicks are resampled based on the embedding probabilities.
    """
    results = []
    df_sample = df.sample(n=size)
    for row in tqdm(df_sample.iterrows()):
        item = int(row[1].arm)
        click = row[1].click
        probs = p_e_a.iloc[item]
        for embed, prob in enumerate(probs):
            row[1].arm = embed
            row[1].click = int(np.random.rand() < prob) * click
            results.append(list(row[1].values))
    df_embed = pd.DataFrame(results, columns=df.columns)
    return df_embed

