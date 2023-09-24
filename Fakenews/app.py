import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix  # Import the csr_matrix class


# Streamlit app
def main():
    st.title('Fake News Detection App')

    absolute_path = os.path.dirname(__file__)
    relative_path = "real_fake.pkl"
    full_path = os.path.join(absolute_path, relative_path)
    
    newmodel = joblib.load(full_path)

    # Input text from user
    input1 = st.text_area('Enter the news text:', '')

    prediction = newmodel.predict([[input1]])
    prediction = prediction[0]
    st.write(f"Predicted: {prediction}")

if __name__ == '__main__':
    main()
