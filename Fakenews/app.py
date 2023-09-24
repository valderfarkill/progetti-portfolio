import streamlit as st
import joblib
import os

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://s.france24.com/media/display/a234d7f4-0f85-11ee-beee-005056bfb2b6/w:1280/p:16x9/EN-TRUTH-OR-FAKE_1920x1080.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

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
    add_bg_from_url()
    main()
