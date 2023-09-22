import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io
import os

import warnings
warnings.filterwarnings('ignore')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapercave.com/wp/wp7802880.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )



def main():
    st.text("Startup Multi Linear")

    absolute_path = os.path.dirname(__file__)
    relative_path = "regression_test.pkl"
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
        file_name='regr_startup.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    input1 = st.number_input("Insert R&D value:",)
    input2 = st.number_input("Insert Administration value:",)
    input3 = st.number_input("Insert Marketing Spend value:",)

    newmodel = joblib.load(full_path)
    prediction = newmodel.predict([[input1, input2, input3]])[0]
    st.write(f"Predicted profit: {round(prediction,1)}$")  
    

    
if __name__ == "__main__":
    add_bg_from_url() 
    main()

#streamlit run app.py