import streamlit as st
import joblib
import pandas as pd
import os
import io
import sklearn

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://c.wallhere.com/photos/53/59/minimalism-161386.jpg!d");
             background-attachment: fixed;
             background-size: 
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

feature_order = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island_Biscoe', 'island_Dream', 'island_Torgersen', 'sex_female','sex_male']

# Streamlit app
def main():

    absolute_path = os.path.dirname(__file__)
    relative_path = "regression_penguins.pkl"
    full_path = os.path.join(absolute_path, relative_path)

    newmodel = joblib.load(full_path)


    st.title('Penguins Regression App')

    # Input features from user
    st.header('Input Features')
    bill_length_mm = st.number_input('Bill Length (mm)', min_value=32.1, step=0.01, max_value=59.6, value=32.1)
    bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=13.1, step=0.01, max_value=21.5, value=13.1)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=172.0, step=0.01, max_value=231.0, value=172.0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=2700.0, step=0.01, max_value=6300.0, value=2700.0)

    if st.button('Predict'):
            # Create a DataFrame with the user's input for penguin regression
            penguin_data = pd.DataFrame({
                'bill_length_mm': [bill_length_mm],
                'bill_depth_mm': [bill_depth_mm],
                'flipper_length_mm': [flipper_length_mm],
                'body_mass_g': [body_mass_g],
                'sex_female': [0],  # You may adjust these values based on user input
                'sex_male': [1],
                'island_Biscoe': [1],
                'island_Dream': [0],
                'island_Torgersen': [0]
            })

            # Reorder the columns based on the feature order used during training
            penguin_data = penguin_data[feature_order]
            # Make a regression prediction using the loaded model
            penguin_prediction = newmodel.predict(penguin_data)

            # Display the penguin regression prediction
            st.subheader('Penguin Regression Prediction:')
            st.write(f'The predicted penguin species is {penguin_prediction[0]}')

if __name__ == '__main__':
    add_bg_from_url() 
    main()
