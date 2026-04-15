# Book Recommendation System

A scalable book recommendation system built with Apache Spark that combines collaborative filtering and content-based filtering techniques to provide personalized book recommendations.

## Overview

This project implements a hybrid recommendation system that leverages both collaborative filtering (using Matrix Factorization with ALS) and content-based filtering to recommend books to users. The system is built using PySpark for distributed computing, making it capable of handling large-scale datasets efficiently.

## Features

- **Collaborative Filtering**: Matrix Factorization using Alternating Least Squares (ALS) algorithm from Spark MLlib
- **Content-Based Filtering**: Cosine similarity-based recommendations using book metadata
- **Hybrid Approach**: Combines ALS predictions with content similarity for improved recommendations
- **Comprehensive Evaluation**: Implements multiple evaluation metrics including Precision@K, Recall@K, and NDCG@K
- **Hyperparameter Tuning**: Automated model optimization using cross-validation
- **Data Visualization**: Exploratory data analysis with matplotlib and seaborn
- **Scalable Architecture**: Built on Apache Spark for distributed processing

## Dataset

The project uses the **Book-Crossing Dataset**, which includes:

- **Ratings.csv**: ~1.1 million user-book ratings (User-ID, ISBN, Rating)
- **Users.csv**: ~278k user profiles (User-ID, Location, Age)
- **Books metadata**: Book information including titles, authors, publishers, and publication years

## Technical Stack

- **Apache Spark 3.5.0**: Distributed computing framework
- **PySpark**: Python API for Spark
- **Pandas 2.1.4**: Data manipulation and analysis
- **NumPy 1.26.2**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Development and experimentation environment

## Architecture

### 1. Data Preprocessing

- Data loading and cleaning
- Feature engineering (author encoding, publisher encoding, publication year normalization)
- String indexing for categorical variables
- Handling missing values and outliers

### 2. Exploratory Data Analysis

- Rating distribution analysis
- User behavior patterns
- Book popularity metrics
- Age demographics visualization

### 3. Collaborative Filtering (ALS)

- Matrix factorization using implicit feedback
- Hyperparameter optimization:
  - Rank (latent factors): [15, 20, 25]
  - Regularization parameter: [0.05, 0.1, 0.15]
  - Max iterations: [15, 20, 25]
  - Alpha (confidence): [10, 45, 50]
- Cross-validation with 5 folds
- RMSE-based model selection

### 4. Content-Based Filtering

- Feature vectors based on book metadata
- Cosine similarity calculation
- Book-to-book similarity matrix

### 5. Hybrid Recommendation

- Weighted combination of ALS ratings and content similarity
- Configurable weights (default: 60% ALS, 40% similarity)
- Re-ranking mechanism for improved relevance

## Model Evaluation

The system is evaluated using multiple metrics:

- **Precision@K**: Proportion of relevant items in top-K recommendations
- **Recall@K**: Proportion of relevant items that are recommended
- **NDCG@K** (Normalized Discounted Cumulative Gain): Ranking quality metric
- **RMSE** (Root Mean Square Error): Prediction accuracy

## Installation

```bash
# Clone the repository
git clone https://github.com/giahuy1310/bookrecommendation.git
cd bookrecommendation

# Install dependencies
pip install -r requirements.txt

# Download the dataset (automatically handled in notebook)
```

## Requirements

```
pyspark==3.5.0
findspark==2.0.1
pandas==2.1.4
numpy==1.26.2
```

## Usage

1. **Open the Jupyter Notebook**:

```bash
jupyter notebook BigData_off1.ipynb
```

1. **Run all cells** to:
  - Load and preprocess the data
  - Train the recommendation models
  - Generate recommendations
  - Evaluate model performance
2. **Get Recommendations** for a user:

```python
# Get top-10 recommendations for a specific user and book
recommendations = get_reranked_recommendations(
    user_id=12345,
    current_book_isbn='0316666343',
    num_recommendations=10
)
```

## Key Functions

### `get_reranked_recommendations(user_id, current_book_isbn, num_recommendations=10)`

Generates personalized book recommendations combining ALS predictions with content-based similarity.

**Parameters:**

- `user_id`: Target user ID
- `current_book_isbn`: ISBN of the current book for context
- `num_recommendations`: Number of recommendations to return

**Returns:** DataFrame with columns:

- `ISBN`: Book identifier
- `Book-Title`: Title of the book
- `Book-Author`: Author name
- `ALS_Rating`: Predicted rating from ALS model
- `Similarity`: Content-based similarity score
- `Final_Score`: Combined weighted score

### `evaluate_model_metrics(model, test_data, k=10)`

Evaluates the recommendation model using multiple metrics.

**Returns:**

- Precision@K
- Recall@K
- NDCG@K

## Results

The hybrid model achieves:

- Balanced recommendations combining user preferences and content similarity
- Improved diversity in recommendations
- Better handling of cold-start problems through content-based component

## Visualizations

The project includes various visualizations:

- Rating distribution histograms
- User age demographics
- Publication year trends
- Recommendation quality heatmaps
- Estimated rating distributions

## Project Structure

```
bookrecommendation/
├── BigData_off1.ipynb          # Main notebook with implementation
├── requirements.txt             # Python dependencies
├── LICENSE                      # Apache 2.0 License
├── Ratings.csv                  # User-book ratings dataset
├── Users.csv                    # User demographics
├── data.zip                     # Dataset archive
├── reranked_recommendations.csv # Sample output
├── reranked_recommendations.pkl # Serialized recommendations
└── *.png                        # Visualization outputs
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Book-Crossing Dataset from [source]
- Apache Spark MLlib documentation
- Research papers on hybrid recommendation systems

## Future Improvements

- Implement deep learning-based recommendations
- Add real-time recommendation API
- Incorporate user reviews and ratings text analysis
- Implement session-based recommendations
- Add A/B testing framework
- Deploy as a web service

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating big data processing and recommendation system techniques using Apache Spark.