import streamlit as st
from pro import ProductRecommender

# Initialize the recommender system
reviews_path = r"C:\Users\BVB\source\repos\projectst\projectst\Health_and_Personal_Care.jsonl"
meta_path = r"C:\Users\BVB\source\repos\projectst\projectst\meta_Health_and_Personal_Care.jsonl"
recommender = ProductRecommender(reviews_path, meta_path)

# Custom styles for better UI
st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #ffffff, #d9e2f3);
        font-family: Arial, sans-serif;
    }
    .recommendation-container {
        border: 2px solid #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("Enhanced Product Recommender System")

# Input Section
query = st.text_input("Search for a Product:", placeholder="Enter product name or description")
uploaded_image = st.file_uploader("Upload an Image for Recommendations", type=["jpg", "png", "jpeg"])
num_results = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)
similarity_threshold = st.slider("Minimum Similarity Score (Text/Image)", min_value=0.0, max_value=1.0, value=0.7)

# Recommendation button
if st.button("Get Recommendations"):
    text_recommendations = None
    image_recommendations = None

    with st.spinner("Processing recommendations..."):
        try:
            # Text-Based Recommendations
            if query:
                text_recommendations = recommender.get_recommendations(query, num_results, similarity_threshold)
                if not text_recommendations.empty:
                    st.subheader("Text-Based Recommendations")
                    for _, product in text_recommendations.iterrows():
                        st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
                        st.write(f"**{product['title_y']}**")
                        st.write(f"Price: {product.get('price', 'N/A')}")
                        st.write(f"Average Rating: {product['avg_rating']:.1f}")
                        st.write(f"Similarity Score: {product['similarity_score']:.2f}")
                        st.write(f"ASIN: {product['asin']}")
                        if product['images_y']:
                            image_base64 = recommender.get_image_as_base64_text(product['images_y'])
                            if image_base64:
                                st.image(f"data:image/jpeg;base64,{image_base64}", use_column_width=True)
                            else:
                                st.warning("Image could not be loaded.")
                        st.markdown('</div>', unsafe_allow_html=True)

            # Image-Based Recommendations
            if uploaded_image:
                with open("query_image.jpg", "wb") as f:
                    f.write(uploaded_image.getbuffer())
                query_image_url = "query_image.jpg"  # Path to the uploaded image
                with st.spinner("Processing image-based recommendations..."):
                    image_recommendations = recommender.get_image_based_recommendations(query_image_url, num_results)
                    if not image_recommendations.empty:
                        st.subheader("Image-Based Recommendations")
                        for _, product in image_recommendations.iterrows():
                            st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
                            st.write(f"**{product['title_y']}**")
                            st.write(f"Price: {product.get('price', 'N/A')}")
                            st.write(f"Average Rating: {product['avg_rating']:.1f}")
                            st.write(f"Image Similarity Score: {product['image_similarity_score']:.2f}")
                            st.write(f"ASIN: {product['asin']}")
                            if product['images_y']:
                                image_base64 = recommender.get_image_as_base64_text(product['images_y'])
                                if image_base64:
                                    st.image(f"data:image/jpeg;base64,{image_base64}", use_column_width=True)
                                else:
                                    st.warning("Image could not be loaded.")
                            st.markdown('</div>', unsafe_allow_html=True)

            # Combined Recommendations
            if text_recommendations is not None or image_recommendations is not None:
                st.subheader("Combined Recommendations")
                combined_recommendations = recommender.combine_features(
                    text_recommendations or pd.DataFrame(),
                    image_recommendations or pd.DataFrame(),
                    num_results
                )
                if not combined_recommendations.empty:
                    for _, product in combined_recommendations.iterrows():
                        st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
                        st.write(f"**{product['title_y']}**")
                        st.write(f"Price: {product.get('price', 'N/A')}")
                        st.write(f"Average Rating: {product['avg_rating']:.1f}")
                        st.write(f"Final Combined Score: {product['final_score']:.2f}")
                        st.write(f"ASIN: {product['asin']}")
                        if product['images_y']:
                            image_base64 = recommender.get_image_as_base64_text(product['images_y'])
                            if image_base64:
                                st.image(f"data:image/jpeg;base64,{image_base64}", use_column_width=True)
                            else:
                                st.warning("Image could not be loaded.")
                        st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
