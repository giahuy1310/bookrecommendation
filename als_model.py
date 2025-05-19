import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, expr, when, isnan
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder \
    .appName("BookRecommendationALS") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

def load_data():
    """Load and preprocess the data"""
    # Load ratings data
    ratings_df = spark.read.csv("Ratings.csv", header=True, inferSchema=True)
    
    # Convert ratings to implicit feedback (1 if rated, 0 if not)
    # Handle NaN values as no interaction (0)
    implicit_ratings = ratings_df.select(
        col("User-ID").cast("integer"),
        col("ISBN").cast("string"),
        when(isnan(col("Book-Rating")), 0).otherwise(1).alias("implicit_rating")
    )
    
    return implicit_ratings

def train_als_model(ratings_df):
    """Train ALS model on the given ratings"""
    # Split the data
    train, validation, test = ratings_df.randomSplit([0.6, 0.2, 0.2], seed=42)
    
    # Build the ALS model
    als = ALS(
        maxIter=15,
        regParam=0.01,
        alpha=0.01,  # Confidence parameter for implicit feedback
        userCol="User-ID",
        itemCol="ISBN",
        ratingCol="implicit_rating",
        coldStartStrategy="drop",
        nonnegative=True,
        implicitPrefs=True  # Enable implicit feedback mode
    )
    
    # Train the model
    model = als.fit(train)
    
    # Evaluate the model
    predictions = model.transform(validation)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="implicit_rating",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse}")
    
    return model, test

def get_recommendations(model, user_id, n_recommendations=10):
    """Get top N recommendations for a user"""
    # Generate recommendations
    user_recs = model.recommendForUserSubset(
        spark.createDataFrame([(user_id,)], ["User-ID"]),
        n_recommendations
    )
    
    # Convert to pandas for easier viewing
    recommendations = user_recs.select("recommendations").collect()[0][0]
    return [(rec.ISBN, rec.rating) for rec in recommendations]

def main():
    # Load data
    implicit_ratings = load_data()
    
    # Train implicit feedback model
    print("Training implicit feedback model...")
    model, test = train_als_model(implicit_ratings)
    
    # Example: Get recommendations for a user
    user_id = 1  # Replace with actual user ID
    print(f"\nGetting recommendations for user {user_id}")
    
    print("\nImplicit feedback recommendations:")
    recs = get_recommendations(model, user_id)
    for isbn, rating in recs:
        print(f"ISBN: {isbn}, Confidence: {rating:.3f}")

if __name__ == "__main__":
    main() 