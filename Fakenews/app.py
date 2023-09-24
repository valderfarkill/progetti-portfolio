import streamlit as st
import joblib
import re
import os
import joblib


# Streamlit app
def main():
    st.title("Data Mining")
    
    word = st.text_input("Enter the text:  ")

    absolute_path = os.path.dirname(__file__)
    relative_path = "real_fake.pkl"
    full_path = os.path.join(absolute_path, relative_path)

    pipe = joblib.load(full_path)
    
    if word == "":
        st.warning("Input text is required.")
    else:
    
        text_fake = []
        text_fake.append(word)
        
        predictions = pipe.predict(text_fake)
        
        
        # Cambia il colore del background in base al sentiment
        
        if predictions[0] == 1:
                result = "Real News"
        else:
                result = "Fake News"

        st.subheader('Prediction Result:')
        st.write(result)
        
if __name__ == '__main__':
    main()
