# Multimodal Sentiment - Weighted Product Recommendation System

A recommendation engine that intelligently combines textual reviews, product images, and sentiment analysis to provide more accurate and diverse product recommendations in the Health and Personal Care category.

## 📌 Project Overview

Traditional recommendation systems often rely on either textual data or visual data alone. This project introduces a **multimodal recommendation system** that integrates both **product reviews (text)** and **product images**, further enhanced with **sentiment analysis**, to produce more personalized and effective recommendations.

## 🎯 Objective

To develop a hybrid recommendation system that leverages **sentence embeddings, image embeddings, and sentiment scores** to improve product ranking and recommendation relevance.

---

## 🗂️ Dataset

The project uses the Amazon Health and Personal Care dataset.

- `Health_and_Personal_Care.jsonl`: User reviews and associated data
- `meta_Health_and_Personal_Care.jsonl`: Product metadata including titles, images, ratings, and price

### Key Features

- **Text Data**: Processed review text
- **Visual Data**: Product image URLs
- **Metadata**: Ratings, titles, prices
- **Sentiment Scores**: Extracted using `TextBlob` and `DistilBERT` transformer models

---

## 🛠️ Methodology

### 🧠 Models Used

- **Text Similarity**: Sentence-BERT (all-MiniLM-L6-v2) + Cosine similarity
- **Image Similarity**: CLIP (clip-vit-base-patch32) + Cosine similarity
- **Sentiment Analysis**: TextBlob + Transformer-based model (DistilBERT)

### ⚙️ Preprocessing

- Text cleaning via regex
- Sentiment scoring from user reviews
- Image normalization and quality enhancement

---

## 🧩 System Architecture

### `app.py` – Frontend

- Accepts text queries and image uploads
- Calls:
  - `get_recommendations()` for text-based retrieval
  - `get_image_based_recommendations()` for visual similarity
  - `combine_features()` for ranking based on fusion

### `pro.py` – Backend

- Functions:
  - `load_data()` and `process_data()`
  - `batch_process_embeddings()` for scalable embeddings
  - `sentiment_weighted_recommendations()` for final scoring

---

## 📈 Evaluation

- **Metrics**:
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

- **Results**:
  - Multimodal models significantly outperformed unimodal (text-only/image-only)
  - Sentiment weighting improved ranking precision

---

## 🔮 Future Work

- Optimize for large-scale datasets (memory/GPU handling)
- Implement advanced ensemble models for score fusion
- Add personalized user profiles for recommendation context

---

## 🧑‍💻 Contributors

- **Bharath Badri Venkata**

---

## 📎 Acknowledgements

This project leverages open-source libraries such as Hugging Face Transformers, Sentence-Transformers, OpenAI CLIP, and Scikit-learn.

---

## 📌 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

