import streamlit as st
import pandas as pd
import joblib
import numpy
import io
import os
import xlsxwriter
import sklearn
import openpyxl

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images7.alphacoders.com/103/1038307.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



def main():
    st.text("Iris Classification")

    absolute_path = os.path.dirname(__file__)
    relative_path = "regression_iris.pkl"
    full_path = os.path.join(absolute_path, relative_path)

    file = st.file_uploader("Carica un file CSV o Excel", type=["csv", "xlsx"])
    
    if file is not None:
        df = pd.read_csv(file) if file.type == "application/vnd.ms-excel" else pd.read_excel(file)

        # Mostra i dati caricati
        st.write("Dati caricati:")
        st.write(df)
    
        #predictions = newmodel.predict(df[['R&D Spend', 'Administration', 'Marketing Spend']])
        #df['Predicted Profit'] = np.round(predictions, 1)
        #st.write(df)
    
        #download button
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer,sheet_name='Elenco_tot', index=False)
        writer.save()
        output.seek(0)
        st.download_button(
        label="Scarica file Excel",
        data=output,
        file_name='regr_iris.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    newmodel = joblib.load(full_path)

    sepal_lenght = st.number_input("sepal length", 1.0,10.0,3.0)
    sepal_width = st.number_input("sepal width", 1.0,10.0,3.0)
    petal_lenght = st.number_input("petal length", 1.0,10.0,3.0)
    petal_width = st.number_input("petal width", 1.0,10.0,3.0)
    prediction = newmodel.predict([[sepal_lenght,sepal_width, petal_lenght, petal_width]])[0]
    st.write(f"Classification iris: {prediction}")
    

    
if __name__ == "__main__":
    add_bg_from_url() 
    main()

#streamlit run app.py