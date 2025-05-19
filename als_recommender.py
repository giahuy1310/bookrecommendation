import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when, isnan, count, desc
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

class BookRecommender:
    def __init__(self, spark_memory: str = "4g"):
        """Initialize the BookRecommender with Spark configuration"""
        self.spark = SparkSession.builder \
            .appName("BookRecommendationALS") \
            .config("spark.driver.memory", spark_memory) \
            .getOrCreate()
        
        self.model = None
        self.ratings_df = None
        self.books_df = None
        
    def load_data(self, ratings_path: str = "Ratings.csv", books_path: str = "Books.csv"):
        """Load and preprocess the ratings and books data"""
        # Load ratings data
        self.ratings_df = self.spark.read.csv(ratings_path, header=True, inferSchema=True)
        
        # Load books data if available
        try:
            self.books_df = self.spark.read.csv(books_path, header=True, inferSchema=True)
        except:
            print("Books data not found, proceeding with ratings only")
        
        # Convert ratings to implicit feedback
        self.ratings_df = self.ratings_df.select(
            col("User-ID").cast("integer"),
            col("ISBN").cast("string"),
            when(isnan(col("Book-Rating")), 0).otherwise(1).alias("implicit_rating")
        )
        
        # Remove any rows with null values
        self.ratings_df = self.ratings_df.na.drop()
        
        return self
    
    def train(self, 
             max_iter: int = 15,
             reg_param: float = 0.01,
             alpha: float = 0.01,
             rank: int = 10,
             train_ratio: float = 0.8,
             validation_ratio: float = 0.1):
        """Train the ALS model with specified parameters"""
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Split the data
        train, validation, test = self.ratings_df.randomSplit(
            [train_ratio, validation_ratio, 1 - train_ratio - validation_ratio], 
            seed=42
        )
        
        # Build the ALS model
        als = ALS(
            maxIter=max_iter,
            regParam=reg_param,
            alpha=alpha,
            rank=rank,
            userCol="User-ID",
            itemCol="ISBN",
            ratingCol="implicit_rating",
            coldStartStrategy="drop",
            nonnegative=True,
            implicitPrefs=True
        )
        
        # Train the model
        self.model = als.fit(train)
        
        # Evaluate the model
        predictions = self.model.transform(validation)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="implicit_rating",
            predictionCol="prediction"
        )
        rmse = evaluator.evaluate(predictions)
        print(f"Root-mean-square error = {rmse}")
        
        return self
    
    def get_recommendations(self, 
                          user_id: int, 
                          n_recommendations: int = 10,
                          include_book_info: bool = True) -> List[Tuple]:
        """Get top N recommendations for a user with optional book information"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Generate recommendations
        user_recs = self.model.recommendForUserSubset(
            self.spark.createDataFrame([(user_id,)], ["User-ID"]),
            n_recommendations
        )
        
        # Get recommendations
        recommendations = user_recs.select("recommendations").collect()[0][0]
        recs = [(rec.ISBN, rec.rating) for rec in recommendations]
        
        # Add book information if available and requested
        if include_book_info and self.books_df is not None:
            isbns = [rec[0] for rec in recs]
            book_info = self.books_df.filter(col("ISBN").isin(isbns)) \
                .select("ISBN", "Book-Title", "Book-Author") \
                .collect()
            
            # Create a mapping of ISBN to book info
            book_map = {row.ISBN: (row["Book-Title"], row["Book-Author"]) for row in book_info}
            
            # Combine recommendations with book info
            recs = [(isbn, rating, *book_map.get(isbn, ("Unknown", "Unknown"))) 
                   for isbn, rating in recs]
        
        return recs
    
    def get_user_stats(self, user_id: int) -> dict:
        """Get statistics about a user's reading history"""
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        user_ratings = self.ratings_df.filter(col("User-ID") == user_id)
        
        stats = {
            "total_books": user_ratings.count(),
            "active_books": user_ratings.filter(col("implicit_rating") == 1).count()
        }
        
        return stats
    
    def get_popular_books(self, n: int = 10) -> List[Tuple]:
        """Get the most popular books based on number of interactions"""
        if self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        popular_books = self.ratings_df \
            .filter(col("implicit_rating") == 1) \
            .groupBy("ISBN") \
            .agg(count("*").alias("interaction_count")) \
            .orderBy(desc("interaction_count")) \
            .limit(n) \
            .collect()
        
        return [(row.ISBN, row.interaction_count) for row in popular_books]

def main():
    # Example usage
    recommender = BookRecommender()
    
    # Load data
    recommender.load_data()
    
    # Train model
    recommender.train()
    
    # Example: Get recommendations for a user
    user_id = 1
    print(f"\nGetting recommendations for user {user_id}")
    
    # Get user statistics
    stats = recommender.get_user_stats(user_id)
    print(f"\nUser Statistics:")
    print(f"Total books in history: {stats['total_books']}")
    print(f"Active interactions: {stats['active_books']}")
    
    # Get recommendations
    print("\nPersonalized Recommendations:")
    recs = recommender.get_recommendations(user_id, include_book_info=True)
    for rec in recs:
        if len(rec) == 4:  # With book info
            isbn, rating, title, author = rec
            print(f"Book: {title}")
            print(f"Author: {author}")
            print(f"ISBN: {isbn}")
            print(f"Confidence: {rating:.3f}\n")
        else:  # Without book info
            isbn, rating = rec
            print(f"ISBN: {isbn}, Confidence: {rating:.3f}")
    
    # Get popular books
    print("\nMost Popular Books:")
    popular = recommender.get_popular_books(5)
    for isbn, count in popular:
        print(f"ISBN: {isbn}, Interactions: {count}")

if __name__ == "__main__":
    main() 