# Predicting Spotify Song Popularity

This project explores the use of machine learning techniques to predict whether a song is **popular** based on its audio and metadata features from the Spotify Tracks Dataset. 

The main goal is to classify songs as popular (popularity score ≥ 50) or not popular using supervised learning models.

## Dataset

We used the [`spotify-tracks-dataset`](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) from Hugging Face, which contains 114,000+ songs and a variety of features, including:

- Acoustic, rhythmic, and tonal properties (e.g., `danceability`, `energy`, `tempo`, `valence`)
- Track metadata (e.g., `explicit`, `track_genre`, `key`, `mode`)
- Spotify popularity score (ranging from 0 to 100)

## Data Preprocessing

Key preprocessing steps:

- Dropped irrelevant columns (`track_name`, `album_name`, `artists`)
- Handled duplicates and outliers
- One-hot encoded categorical features (e.g., `key`, `mode`, `track_genre`)
- Standardized numerical features
- Created a binary label for popularity:
  - Popular: popularity ≥ 50
  - Not popular: popularity < 50

## Models Trained

| Model           | Test Accuracy | AUC   |
|----------------|----------------|-------|
| K-Nearest Neighbors (KNN) | 56.8%         | 0.58  |
| Random Forest             | 66.0%         | 0.71  |
| Neural Network (PyTorch)  | **78.0%**     | -     |

The **Neural Network** outperformed the other models, demonstrating its ability to learn non-linear patterns from complex features.

## Neural Network Highlights

- One hidden layer with ReLU activation
- Sigmoid output layer for binary classification
- Binary Cross Entropy Loss
- Learning rate tuning and loss tracking across 200 epochs
- No overfitting observed — training and validation loss decreased smoothly

## Class Balance

A threshold of 50 for popularity gave a **relatively balanced** dataset (about 45% popular vs. 55% not popular). This allowed effective model training without applying explicit resampling or class weighting techniques.

## Key Insights

- Neural networks are especially well-suited for this task due to their capacity for modeling complex, non-linear relationships.
- Simpler models like KNN struggle in high-dimensional spaces with weak linear correlations.
- Random forests are robust and interpretable, but may underperform compared to deep learning in nuanced classification tasks.
