# Music Recommendation System (K-Nearest Neighbours)

A content-based, instance-based machine learning model (K-Nearest Neighbours) that recommends the next 5 songs given a seed queue of 10 songs, built using Linear Algebra and Python (`numpy`, `pandas`, `scipy`).

## Project Overview
While production platforms like Spotify use collaborative filtering, deep learning embeddings, and contextual bandits, this project implements the core mathematical backbone of a music recommender system using one-hot binary feature vectors and vector space distance metrics.

## Dataset Layout
The self-generated dataset (`..._Data.csv`) contains 126 songs across 18 artists and 9 genres:
* **Column 1 (`Song_Name`)**: Song title and artist string (`"Song - Artist"`).
* **Columns 2 to 19**: One-hot binary encoded Artist columns (`1` if the song belongs to the artist, `0` otherwise).
* **Columns 20 to 28**: One-hot binary encoded Genre columns (`1` if the song belongs to the genre, `0` otherwise).

## Features Implemented
* **Dynamic Feature Extraction**: Automatically detects the song column and binary feature columns without hardcoding column names or indices.
* **Recency-Weighted Queue Centroid**: Dynamically computes a user taste profile vector from the active queue, weighting recently added songs higher so recommendations adapt immediately when the user's mood changes.
* **Sparsity Optimization**: Uses `scipy.sparse.csr_matrix` to handle high-dimensional binary sparsity (~89% zeros) efficiently.
* **Dynamic Queue Manipulation & Edge Case Detection**:
  * Append recommended songs or any custom song from the world dataset.
  * Insert songs between queue positions or delete songs by title/index.
  * Clear the entire queue and auto-generate a new playlist purely from recommendations.
  * Handles missing songs, duplicate prevention, empty queue checks, and case-insensitive string matching.
* **Distance Metric Comparison**: Compares recommendations across Cosine, Euclidean, and Jaccard distances.

## Deliverables in This Repository
1. `Data.csv` — One-hot encoded binary world dataset (126 songs).
2. `Data_Gen.ipynb` — Dataset generation notebook.
3. `MusicRecommender.ipynb` — Main KNN music recommender notebook with queue manipulation and analysis.

## How to Run
### Option 1: Run on Google Colab (Recommended)
1. Open `MusicRecommender.ipynb` in Google Colab.
2. Upload `Data.csv` to the Colab Files sidebar.
3. Run all cells (`Runtime -> Run all`).

### Option 2: Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR-GITHUB-USERNAME/knn-music-recommender.git](https://github.com/YOUR-GITHUB-USERNAME/knn-music-recommender.git)
   cd knn-music-recommender
