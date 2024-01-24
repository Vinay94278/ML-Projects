import numpy as np
import pickle
import json
import streamlit as st

pickle_in = open("Banglore_Home_Price_Model.pickle","rb")
regressor = pickle.load(pickle_in)

with open('columns.json') as f:
    columns = json.load(f)["feature_columns"]
    options = columns[3:]

def predict_home_price(location,sqft,bath,bhk):
    try:
        loc_index = columns.index(location.lower())
    except:
        loc_index = -1
    y=np.zeros(len(columns))
    y[0] = sqft
    y[1] = bath
    y[2] = bhk
    if loc_index >= 0:
        y[loc_index] = 1

    return round(regressor.predict([y])[0],2)

def main():
    st.title("Banglore Home Price Estimator")
    location = st.selectbox('Location', options)
    sqft = st.number_input('Total Sqft',value=0.0,format="%.2f")
    bhk = st.number_input("BHK", min_value=1, max_value=5)
    bath = st.number_input("BATH", min_value=1, max_value=5)
    result = ""
    if st.button("Estimate"):
        result = predict_home_price(location,sqft,bath,bhk)
    st.success('Estimate Price : {}'.format(result))

if __name__ == '__main__':
    main()