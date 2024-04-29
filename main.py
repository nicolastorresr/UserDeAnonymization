import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import recordlinkage

def preprocess_data(dataset, min_ratings=10):
    """
    Preprocess the dataset by filtering out users with fewer than min_ratings
    and normalizing the rating values between 0 and 1.
    """
    user_counts = dataset.groupby('user_id')['rating'].count()
    valid_users = user_counts[user_counts >= min_ratings].index
    filtered_data = dataset[dataset['user_id'].isin(valid_users)]
    
    rating_min = filtered_data['rating'].min()
    rating_max = filtered_data['rating'].max()
    filtered_data['rating'] = (filtered_data['rating'] - rating_min) / (rating_max - rating_min)
    
    return filtered_data

def extract_quasi_identifiers(dataset):
    """
    Extract the rating vectors for each user as quasi-identifiers.
    """
    user_ratings = dataset.groupby('user_id')['rating'].apply(list).reset_index()
    user_ratings.columns = ['user_id', 'rating_vector']
    return user_ratings

def compute_similarities(dataset1, dataset2):
    """
    Compute the pairwise cosine similarities between rating vectors of users across the two datasets.
    """
    similarities = cosine_similarity(dataset1['rating_vector'], dataset2['rating_vector'])
    return similarities

def perform_record_linkage(similarities, m_prob, u_prob):
    """
    Perform record linkage using the Fellegi-Sunter model with the given probabilities.
    """
    indexer = recordlinkage.Index()
    indexer.full()
    
    pair_scores = indexer.score_pairs(similarities, m_prob, u_prob)
    
    # Set appropriate threshold based on domain knowledge or tuning
    threshold = ...
    
    clusters = pair_scores[pair_scores > threshold].to_clusters()
    
    return clusters
