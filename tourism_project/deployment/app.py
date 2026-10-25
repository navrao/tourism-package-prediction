import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Tourism Package Prediction", page_icon="🌴")

# Load the model committed by the pipeline
model = joblib.load("tourism_project/deployment/model.joblib")

st.title("Wellness Tourism Package Prediction")
st.write("Fill in the customer details to see how likely they are to buy the package.")

# Customer details
st.subheader("Customer details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly income", 1000, 200000, 20000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Has passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
with col2:
    persons = st.number_input("People visiting", 1, 10, 3)
    children = st.number_input("Children visiting (under 5)", 0, 5, 0)
    trips = st.number_input("Trips per year", 0, 30, 3)
    star = st.selectbox("Preferred hotel star", [3.0, 4.0, 5.0])

# Interaction details
st.subheader("Sales interaction")
col3, col4 = st.columns(2)
with col3:
    contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    followups = st.number_input("Number of follow-ups", 0, 10, 3)
with col4:
    duration = st.number_input("Pitch duration (minutes)", 1, 150, 15)
    satisfaction = st.slider("Pitch satisfaction score", 1, 5, 3)

# Put the inputs into a dataframe (same columns as the training data)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

if st.button("Predict"):
    prob = model.predict_proba(input_df)[0, 1]
    if prob >= 0.5:
        st.success(f"Likely to buy the package (probability {prob:.0%})")
    else:
        st.warning(f"Unlikely to buy the package (probability {prob:.0%})")
    st.caption("Input sent to the model:")
    st.dataframe(input_df)
