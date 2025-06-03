# Back-end logic: Full version of ProductRecommender with all functionalities.

import json
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
from ast import literal_eval
import base64
import logging
from pathlib import Path
from textblob import TextBlob
from transformers import pipeline
from tqdm import tqdm
from collections import Counter
import random
import torch

import threading
from queue import Queue

import os
import pickle
import gc

from transformers import CLIPProcessor, CLIPModel
# Enable tqdm for pandas
tqdm.pandas()

class ProductRecommender:
    def __init__(self, reviews_path, meta_path):
        """Initialize the recommender with data files."""
        self.logger = self._setup_logging()
        try:
            self.load_data(reviews_path, meta_path)
            self.initialize_model()
            self.process_data()
            self.initialize_sentiment_models()
        except Exception as e:
            self.logger.error(f"Failed to initialize ProductRecommender: {str(e)}")
            raise

    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger('ProductRecommender')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_data(self, reviews_path, meta_path):
        """Load and process the JSONL data files."""
        def load_jsonl(file_path):
            data = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in tqdm(file, desc=f"Loading {file_path}", unit="lines"):
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping invalid JSON line: {e}")
            if not data:
                raise ValueError(f"No valid data could be loaded from {file_path}")
            return pd.DataFrame(data)

        self.logger.info("Loading review data...")
        self.reviews_df = load_jsonl(reviews_path)

        self.logger.info("Loading metadata...")
        self.meta_df = load_jsonl(meta_path)

    def initialize_model(self):
        """Initialize the text embedding model with GPU support."""
        self.logger.info("Initializing text embedding model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def initialize_sentiment_models(self):
        """Initialize sentiment analysis models with GPU support."""
        self.logger.info("Initializing sentiment analysis models...")
        device = 0 if torch.cuda.is_available() else -1
        self.textblob_sentiment = self._textblob_sentiment
        self.transformers_sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

    def _textblob_sentiment(self, text):
        """Perform sentiment analysis using TextBlob."""
        try:
            text = str(text) if text is not None else ''
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0.05:
                sentiment = 'Positive'
            elif polarity < -0.05:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            return {'polarity': polarity, 'sentiment': sentiment}
        except Exception as e:
            self.logger.warning(f"Error in TextBlob sentiment analysis: {str(e)}")
            return {'polarity': 0, 'sentiment': 'Neutral'}

    def process_data(self):
        """Process and merge the review and metadata."""
        def preprocess_text(text):
            if pd.isnull(text):
                return ''
            return re.sub(r'[^\w\s]', '', str(text).lower())

        try:
            # Check if 'text' column exists in reviews_df
            if 'text' not in self.reviews_df.columns:
                self.logger.warning("'text' column is missing in reviews_df. Creating a placeholder.")
                self.reviews_df['text'] = ""

            # Preprocess review text
            self.logger.info("Processing review text...")
            self.reviews_df['processed_text'] = self.reviews_df['text'].apply(preprocess_text)
            self.logger.info(f"Processed text column sample: {self.reviews_df['processed_text'].head()}")

            # Check if 'title' column exists in meta_df
            if 'title' not in self.meta_df.columns:
                self.logger.warning("'title' column is missing in meta_df. Creating a placeholder.")
                self.meta_df['title'] = ""

            # Preprocess product titles
            self.logger.info("Processing product titles...")
            self.meta_df['processed_title'] = self.meta_df['title'].apply(preprocess_text)

            # Merge reviews and metadata
            self.logger.info("Merging review and metadata...")
            if 'asin' not in self.reviews_df.columns or 'parent_asin' not in self.meta_df.columns:
                raise KeyError("'asin' or 'parent_asin' column is missing in the datasets.")
            self.merged_df = pd.merge(
                self.reviews_df,
                self.meta_df,
                left_on='asin',
                right_on='parent_asin',
                how='inner'
            )

            # Confirm `processed_text` existence in `merged_df`
            if 'processed_text' not in self.merged_df.columns:
                raise KeyError("The 'processed_text' column is missing in the merged dataset.")

            # Add rating metrics
            self.logger.info("Calculating rating metrics...")
            ratings_data = self.merged_df.groupby('asin').agg({
                'rating': ['mean', 'count']
            }).reset_index()
            ratings_data.columns = ['asin', 'avg_rating', 'review_count']
            self.merged_df = pd.merge(self.merged_df, ratings_data, on='asin', how='left')
            self.merged_df = self.merged_df[self.merged_df['avg_rating'] > 3.5]

            self.logger.info(f"Processed data successfully. Final dataset contains {len(self.merged_df)} entries.")
            # Log the columns of merged_df
            self.logger.info(f"Columns in merged_df: {self.merged_df.columns.tolist()}")
            print(f"Columns in merged_df: {self.merged_df.columns.tolist()}")  # Optional: for debugging during execution

        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise




    def generate_product_caption(self, product_row):
        """
        Generate a meaningful caption by summarizing multiple reviews.

        Args:
            product_row (pd.Series): A row from the merged DataFrame.

        Returns:
            str: A generated product caption summarizing the reviews.
        """
        try:
            # Extract reviews for the product
            reviews = self.reviews_df[self.reviews_df['asin'] == product_row['asin']]['text']
            if reviews.empty:
                return "No reviews available for this product."

            # Combine reviews into a single text block
            combined_reviews = ' '.join(reviews.sample(min(5, len(reviews))))  # Use up to 5 random reviews

            # Preprocess reviews (remove special characters, keep only meaningful text)
            meaningful_text = re.sub(r'[^\w\s]', '', combined_reviews.lower())
            meaningful_text = ' '.join([word for word in meaningful_text.split() if len(word) > 2])

            # Extract key sentences using simple sentence splitting
            sentences = re.split(r'\.|\?|!', combined_reviews)
            meaningful_sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 20]

            # Create a summary sentence
            summary_sentence = ' '.join(meaningful_sentences[:3])  # Take the first 3 meaningful sentences

            return f"Customer reviews highlight: {summary_sentence}"
        except Exception as e:
            self.logger.warning(f"Error generating product caption: {str(e)}")
            return "No meaningful review data."
    

    def batch_process_embeddings(self, texts, batch_size=10000):
        """
        Process embeddings in smaller batches to avoid memory issues.
        Args:
            texts (list): List of texts to encode.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Combined embeddings from all batches.
        """
        embeddings = []
        for start_idx in range(0, len(texts), batch_size):
            batch_texts = texts[start_idx:start_idx + batch_size]
            batch_embeddings = self.text_model.encode(batch_texts, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)


    def get_recommendations(self, query, top_n=5, similarity_threshold=0.7, batch_size=10000, cache_file="text_embeddings.pkl"):
        """
        Get product recommendations based on text similarity with cached embeddings.

        Args:
            query (str): The query text.
            top_n (int): Number of recommendations to return.
            similarity_threshold (float): Minimum similarity score for recommendations.
            batch_size (int): Batch size for processing embeddings.
            cache_file (str): Path to the cache file for text embeddings.

        Returns:
            pd.DataFrame: Recommended products based on text similarity.
        """
        try:
            # Encode the query embedding
            query_embedding = self.text_model.encode(query)

            # Check if cached embeddings exist
            if os.path.exists(cache_file):
                self.logger.info(f"Loading cached embeddings from {cache_file}...")
                with open(cache_file, "rb") as f:
                    product_embeddings = pickle.load(f)
            else:
                self.logger.info("Generating embeddings and saving to cache...")
                product_texts = self.merged_df['processed_text'].tolist()

                # Process embeddings in batches
                product_embeddings = self.batch_process_embeddings(product_texts, batch_size)

                # Save embeddings to cache
                with open(cache_file, "wb") as f:
                    pickle.dump(product_embeddings, f)

            # Compute similarity scores
            similarity_scores = cosine_similarity([query_embedding], product_embeddings)[0]

            # Add similarity scores to DataFrame
            self.merged_df['similarity_score'] = similarity_scores

            # Filter recommendations based on similarity threshold
            filtered_recommendations = self.merged_df[self.merged_df['similarity_score'] >= similarity_threshold]

            # Sort and limit the number of recommendations
            top_recommendations = filtered_recommendations.sort_values(
                'similarity_score', ascending=False
            ).head(top_n)

            return top_recommendations[['title_y', 'avg_rating', 'similarity_score', 'price', 'images_y', 'asin']]
        except Exception as e:
            self.logger.error(f"Error in text-based recommendations: {str(e)}")
            return pd.DataFrame()


    def get_image_as_base64_text(self, image_data):
        """
        Convert the first valid image URL in the data to a resized Base64 string.

        Args:
            image_data (str or list or dict): Image data containing URLs.

        Returns:
            str: Base64-encoded string of the resized image, or None if processing fails.
        """
        try:
            if isinstance(image_data, str):
                if image_data.startswith(('http://', 'https://')):
                    image_url = image_data
                else:
                    image_data = literal_eval(image_data)
            elif isinstance(image_data, list):
                image_data = image_data[0]
            elif isinstance(image_data, dict):
                image_url = image_data.get('hi_res') or image_data.get('large') or image_data.get('thumb')
            else:
                self.logger.warning("Unsupported image data format.")
                return None

            if isinstance(image_data, dict):
                image_url = image_data.get('hi_res') or image_data.get('large') or image_data.get('thumb')
            else:
                image_url = image_data

            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            # Resize the image to half its size
            img = img.resize((img.width, img.height))

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            self.logger.warning(f"Error processing image: {str(e)}")
            return None

    def feature_based_evaluation(self, query, recommendations, k=5, similarity_threshold=0.7):
        """
        Evaluate recommendations using feature-based similarity.

        Args:
            query (str): Search query.
            recommendations (pd.DataFrame): Recommendations DataFrame.
            k (int): Number of top recommendations to evaluate.
            similarity_threshold (float): Threshold for relevance.

        Returns:
            dict: Precision and Recall metrics.
        """
        query_embedding = self.text_model.encode(query)
        product_embeddings = self.text_model.encode(recommendations['processed_text'].tolist())

        # Count relevant items
        relevant_count = 0
        total_relevant = 0
        for idx, product_embedding in enumerate(product_embeddings):
            similarity_score = cosine_similarity([query_embedding], [product_embedding])[0][0]
            if similarity_score >= similarity_threshold:
                total_relevant += 1
                if idx < k:
                    relevant_count += 1

        precision = relevant_count / k if k > 0 else 0
        recall = relevant_count / total_relevant if total_relevant > 0 else 0

        return {"Precision@k": precision, "Recall@k": recall}

    def calculate_mrr(self, recommendations, ground_truth_asin):
        """
        Calculate Mean Reciprocal Rank (MRR).

        Args:
            recommendations (pd.DataFrame): DataFrame of recommended products.
            ground_truth_asin (str): The ground truth ASIN.

        Returns:
            float: MRR score.
        """
        try:
            ranks = recommendations['asin'].tolist()
            if ground_truth_asin in ranks:
                rank = ranks.index(ground_truth_asin) + 1
                return 1 / rank
            else:
                return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating MRR: {str(e)}")
            return 0.0


    def calculate_ndcg(self, recommendations, ground_truth_asin, k=5):
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG).

        Args:
            recommendations (pd.DataFrame): DataFrame of recommended products.
            ground_truth_asin (str): The ground truth ASIN.
            k (int): Number of top recommendations to consider.

        Returns:
            float: NDCG score.
        """
        try:
            dcg = 0.0
            idcg = 1.0  # Ideal DCG is 1 (ground truth ASIN is at the top)

            ranks = recommendations['asin'].tolist()[:k]
            for idx, asin in enumerate(ranks):
                if asin == ground_truth_asin:
                    dcg = 1 / np.log2(idx + 2)  # DCG calculation

            return dcg / idcg  # Normalize by IDCG
        except Exception as e:
            self.logger.error(f"Error calculating NDCG: {str(e)}")
            return 0.0


    def evaluate_alternative(self, query, recommendations, ground_truth_asin, k=5):
        """
        Evaluate recommendations using MRR and NDCG.

        Args:
            query (str): Search query.
            recommendations (pd.DataFrame): Recommendations DataFrame.
            ground_truth_asin (str): The ground truth ASIN.
            k (int): Number of top recommendations to evaluate.

        Returns:
            dict: MRR and NDCG scores.
        """
        query_embedding = self.text_model.encode(query)

        # Calculate MRR
        mrr = self.calculate_mrr(recommendations, ground_truth_asin)

        # Calculate NDCG
        ndcg = self.calculate_ndcg(recommendations, ground_truth_asin, k)

        return {
            "MRR": mrr,
            "NDCG@k": ndcg
        }

    def get_sentiment_score(self, asin):
        """
        Calculate sentiment score for a given product based on its reviews.
        Args:
            asin (str): The ASIN of the product.

        Returns:
            float: Sentiment score (average sentiment of all reviews for the product).
        """
        try:
            product_reviews = self.reviews_df[self.reviews_df['asin'] == asin]
            if product_reviews.empty:
                return 1.0  # Default neutral sentiment if no reviews

            # Calculate sentiment for each review
            sentiments = product_reviews['processed_text'].apply(self._textblob_sentiment)
            sentiment_values = sentiments.apply(lambda x: 1 if x['sentiment'] == 'Positive' else (-1 if x['sentiment'] == 'Negative' else 0))
            return sentiment_values.mean()  # Average sentiment score
        except Exception as e:
            self.logger.error(f"Error calculating sentiment score for ASIN {asin}: {str(e)}")
            return 1.0  # Default neutral sentiment on error

    def sentiment_weighted_recommendations(self, recommendations, num_results=5):
        """
        Adjust the similarity scores of existing recommendations using sentiment weights.
        Args:
            recommendations (pd.DataFrame): Recommendations generated by get_recommendations().
            num_results (int): Number of recommendations to return.

        Returns:
            pd.DataFrame: Recommendations with sentiment-weighted scores.
        """
        recommendations['sentiment_score'] = recommendations['asin'].apply(self.get_sentiment_score)
        recommendations['final_score'] = recommendations['similarity_score'] * recommendations['sentiment_score']
        return recommendations.sort_values(by='final_score', ascending=False).head(num_results)

    def get_valid_image_url(self, images_list):
        """
        Retrieve the most suitable image URL from the provided images list.
        Args:
            images_list (list): List of dictionaries containing image URLs.

        Returns:
            str: The best available image URL, or None if no valid URL is found.
        """
        if not isinstance(images_list, list) or not images_list:
            return None

        # Prioritize the best resolution: hi_res > large > thumb
        for image_data in images_list:
            url = image_data.get("large") or image_data.get("thumb")
            if url:
                return url

        return None

    def resize_image(self,image, max_size=(128, 128)):
        """
        Resize the image to the specified maximum size while maintaining aspect ratio.
        Args:
            image (PIL.Image.Image): The image to resize.
            max_size (tuple): The maximum width and height.

        Returns:
            PIL.Image.Image: The resized image.
        """
        return image.resize(max_size, Image.Resampling.LANCZOS)


    def get_image_based_recommendations(self, query_image_url, num_results=5, cache_file="image_embeddings.pkl"):
        """
        Recommend products based on visual similarity using CLIP with batch processing and caching.

        Args:
            query_image_url (str): URL or local path of the query image.
            num_results (int): Number of recommendations to return.
            cache_file (str): Path to the cache file for embeddings.

        Returns:
            pd.DataFrame: Recommended products based on image similarity.
        """
        try:
            # Initialize CLIP processor and model
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda" if torch.cuda.is_available() else "cpu")

            # Load and process the query image
            if query_image_url.startswith(('http://', 'https://')):
                response = requests.get(query_image_url, timeout=5)
                response.raise_for_status()
                query_image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                query_image = Image.open(query_image_url).convert("RGB")
            query_image = query_image.resize((128, 128), Image.Resampling.LANCZOS)

            # Encode the query image
            query_inputs = processor(images=query_image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            query_embedding = model.get_image_features(**query_inputs).detach().cpu().numpy()

            # Check if cached embeddings exist
            if os.path.exists(cache_file):
                self.logger.info(f"Loading cached embeddings from {cache_file}...")
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                product_embeddings = cache_data["embeddings"]
                valid_indices = cache_data["indices"]
            else:
                self.logger.info("Generating new embeddings and caching them...")
                product_embeddings = []
                valid_indices = []
                product_images = [
                    (index, self.get_valid_image_url(row['images_y']))
                    for index, row in self.merged_df.iterrows()
                    if isinstance(row['images_y'], list) and row['images_y']
                ]

                batch_size = 16
                for i in tqdm(range(0, len(product_images), batch_size), desc="Processing product images"):
                    batch = product_images[i:i + batch_size]
                    batch_indices, batch_urls = zip(*batch)
                    batch_images = []

                    for url in batch_urls:
                        try:
                            response = requests.get(url, timeout=5)
                            response.raise_for_status()
                            product_image = Image.open(BytesIO(response.content)).convert("RGB")
                            resized_image = self.resize_image(product_image)
                            batch_images.append(resized_image)
                        except Exception as e:
                            self.logger.warning(f"Skipping image URL {url}: {str(e)}")

                    if batch_images:
                        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(
                            "cuda" if torch.cuda.is_available() else "cpu")
                        batch_embeddings = model.get_image_features(**inputs).detach().cpu().numpy()
                        product_embeddings.extend(batch_embeddings)
                        valid_indices.extend(batch_indices)

                    # Free memory
                    #del batch_images, inputs, batch_embeddings
                    gc.collect()

                # Cache embeddings
                with open(cache_file, "wb") as f:
                    pickle.dump({"embeddings": np.array(product_embeddings), "indices": valid_indices}, f)

            # Compute similarity scores
            product_embeddings = np.array(product_embeddings)
            similarities = cosine_similarity(query_embedding, product_embeddings).flatten()

            # Assign scores back to DataFrame
            self.merged_df['image_similarity_score'] = 0
            self.merged_df.loc[valid_indices, 'image_similarity_score'] = similarities

            # Return top recommendations
            recommendations = self.merged_df.sort_values(by='image_similarity_score', ascending=False).head(num_results)
            return recommendations[['title_y', 'avg_rating', 'image_similarity_score', 'images_y', 'asin']]
        except Exception as e:
            self.logger.error(f"Error in image-based recommendations: {str(e)}")
            return pd.DataFrame()


    def get_image_as_base64(self, image_data):
        """
        Convert the first valid image URL in the data to a resized Base64 string.

        Args:
            image_data (str or list or dict): Image data containing URLs.

        Returns:
            str: Base64-encoded string of the resized image, or None if processing fails.
        """
        try:
            # Check and extract image URL
            if isinstance(image_data, str):
                image_data = literal_eval(image_data) if image_data.startswith("{") else image_data
            if isinstance(image_data, list) and len(image_data) > 0:
                image_data = image_data[0]  # Use the first entry in the list
            if isinstance(image_data, dict):
                image_url = (
                    image_data.get("large_image_url") or
                    image_data.get("medium_image_url") or
                    image_data.get("small_image_url")
                )
            elif isinstance(image_data, str) and image_data.startswith(('http://', 'https://')):
                image_url = image_data
            else:
                self.logger.warning("Unsupported image data format.")
                return None

            # Download and process the image
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = img.resize((128, 128))  # Resize to a smaller size for performance

            # Encode the image as Base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            self.logger.warning(f"Error processing image: {str(e)}")
            return None



    def combine_features(self, text_recommendations, image_recommendations, num_results=5):
        """
        Combine text-based, image-based, and sentiment scores into a single final score.
        Args:
            text_recommendations (pd.DataFrame): Recommendations based on text similarity.
            image_recommendations (pd.DataFrame): Recommendations based on image similarity.
            num_results (int): Number of top combined recommendations to return.

        Returns:
            pd.DataFrame: Combined recommendations with a final score.
        """
        combined = text_recommendations.merge(
            image_recommendations, on='asin', suffixes=('_text', '_image'), how='outer'
        ).fillna(0)

        # Combine similarity and sentiment scores
        combined['final_score'] = (
            0.4 * combined['similarity_score_text'] + 
            0.4 * combined['image_similarity_score'] + 
            0.2 * combined['sentiment_score']
        )

        return combined.sort_values(by='final_score', ascending=False).head(num_results)





