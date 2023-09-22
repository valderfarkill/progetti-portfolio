import numpy as np
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

# Load the pre-trained machine learning model
model = joblib.load('real_fake.pkl')

# Initialize NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.ediltecnico.it/wp-content/uploads/2019/01/bim-immobili-web-1280x720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Define a function to preprocess text data
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'\W+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    # Join the cleaned words back into a sentence
    text = ' '.join(words)
    return text

# Streamlit app
def main():
    st.title('Fake News Detection App')

    # Input text from user
    news_text = st.text_area('Enter the news text:', '')

    if st.button('Detect'):
        if news_text:
            # Preprocess the user input
            preprocessed_text = preprocess_text(news_text)

            # Vectorize the preprocessed text
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            
            tfidf_vector = tfidf_vectorizer.fit_transform([preprocessed_text])

            # Make a prediction using the loaded model
            prediction = model.predict(tfidf_vector)

            # Display the prediction result
            if prediction[0] == 1:
                result = "Fake News"
            else:
                result = "Real News"

            st.subheader('Prediction Result:')
            st.write(result)
        else:
            st.warning('Please enter news text.')

if __name__ == '__main__':
    add_bg_from_url()
    main()