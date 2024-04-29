# De-anonymizing Users Across Rating Datasets via Record Linkage and Quasi-Identifier Attacks

This implementation follows the steps outlined in the proposed approach:

1. The `preprocess_data` function filters out users with fewer than a specified number of ratings and normalizes the rating values between 0 and 1.
2. The `extract_quasi_identifiers` function extracts the rating vectors for each user as quasi-identifiers.
3. The `compute_similarities` function computes the pairwise cosine similarities between rating vectors of users across the two datasets.
4. The `perform_record_linkage` function performs record linkage using the Fellegi-Sunter model from the recordlinkage library, taking the similarities and probabilities as input.
