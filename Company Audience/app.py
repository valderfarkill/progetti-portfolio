import numpy as np
import pandas as pd
import mlem
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xlsxwriter
import os
import io

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://blogs.glowscotland.org.uk/ea/cumnockmedia/files/2014/10/audience-orange.jpg?w=300");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def main(): 
    
    new_model = mlem.api.load('model.mlem')
    
    st.title('Company')

    path="https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Company.csv"
    df = pd.read_csv(path)

    st.dataframe(df)

    X = df.drop(columns="Sales")
    y = df["Sales"]

    st.subheader('Input')
    rd_spend = st.slider('TV', min_value=0, max_value=500, value=300)
    admin = st.slider('Radio', min_value=0, max_value=500, value=300)
    marketing = st.slider('Newspaper', min_value=0, max_value=500, value=300)

    input_data = {'TV': rd_spend,
                  'Radio': admin,
                  'Newspaper': marketing}

    input_df = pd.DataFrame(input_data, index=[0])
    prediction = new_model.predict([[rd_spend,admin,marketing]])

    st.subheader('Output')
    st.write(f'Audience prevista: {prediction[0]}')

    # Plotly graph
    fig = px.scatter_3d(df, x='TV', y='Radio', z='Newspaper', color_discrete_sequence=['red'])
    fig.update_layout(
        title="Regressione multipla",
        scene=dict(
            xaxis_title='TV',
            yaxis_title='Radio',
            zaxis_title='Newspaper'
        )
    )
    fig.add_trace(px.scatter_3d(input_df, x='TV', y='Radio', z='Newspaper', color='Radio').data[0])
    st.plotly_chart(fig, use_container_width=True)

    #mlem.api.save(model, 'model.mlem', sample_data=X_train)


    predictions = new_model.predict(X)
    custom_predictions = new_model.predict([[rd_spend, admin, marketing]])[0]
    st.write(custom_predictions)

    if st.button('Scarica'):
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        input_df.to_excel(writer, index=False, sheet_name='Input')
        pd.DataFrame(prediction, columns=['Profitto previsto']).to_excel(writer, index=False, sheet_name='Output')
        writer.save()
        output.seek(0)
        st.download_button(label="Download", data=output, file_name='output.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',)

if __name__ == '__main__':
    add_bg_from_url() 
    main()
    
# streamlit run app.py        