import streamlit as st
import pandas as pd
import numpy as np
import pickle
#sklearn
import sklearn
import imblearn

# Print library versions
#st.write(f"Streamlit version: {st.__version__}")
#st.write(f"Pandas version: {pd.__version__}")
#st.write(f"Numpy version: {np.__version__}")
#st.write(f"Scikit-learn version: {sklearn.__version__}")
#st.write(f"Imbalanced-learn version: {imblearn.__version__}")

from sklearn.preprocessing import LabelEncoder

try:
    # Load the pre-trained model
    with open('bank_model.pkl.sav', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()


# Function to preprocess the input data using LabelEncoder
def preprocess(client_df):
    # List of categorical columns
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Initialize the LabelEncoder
    le = LabelEncoder()
    
    # Apply LabelEncoder to each categorical column
    for col in categorical_columns:
        client_df[col] = le.fit_transform(client_df[col])
    
    return client_df

# Function to ensure feature names match those used during training
def ensure_feature_order(data):
    model_features = [
        'duration', 'previous', 'emp.var.rate', 'euribor3m', 'nr.employed',
        'contacted_before', 'contact_cellular', 'contact_telephone',
        'month_dec', 'month_mar', 'month_may', 'month_oct', 'month_sep',
        'poutcome_nonexistent', 'poutcome_success'
    ]
    return data.reindex(columns=model_features, fill_value=0)

# Streamlit app
st.title("Bank Marketing Prediction App")

st.header("Client Information")
age = st.slider("Age", 18, 100)
job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
                           "retired", "self-employed", "services", "student", "technician", 
                           "unemployed", "unknown"])
marital = st.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
education = st.selectbox("Education", ["basic.4y", "basic.6y", "basic.9y", "high.school", 
                                       "illiterate", "professional.course", "university.degree", 
                                       "unknown"])
default = st.selectbox("Default", ["no", "yes", "unknown"])
balance = st.number_input("Balance", min_value=0)
housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
day = st.slider("Day of Contact", 1, 31)
month = st.selectbox("Month of Contact", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", 
                                          "sep", "oct", "nov", "dec"])
duration = st.number_input("Last Contact Duration", min_value=0)
campaign = st.number_input("Number of Contacts", min_value=1)
pdays = st.number_input("Days since Last Contact", min_value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0)
poutcome = st.selectbox("Previous Outcome", ["failure", "nonexistent", "success"])

# Create a dictionary from the input
client_data = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}

# Convert to DataFrame
client_df = pd.DataFrame([client_data])

# Preprocess the data
processed_data = preprocess(client_df)

# Ensure feature order matches model expectations
processed_data = ensure_feature_order(processed_data)

if st.button('Predict Subscription'):
    # Make a prediction
    prediction = model.predict(processed_data)

    # Display the result
    if prediction[0] == 1:
        st.success("The client is likely to subscribe to a term deposit.")
    else:
        st.warning("The client is unlikely to subscribe to a term deposit.")
