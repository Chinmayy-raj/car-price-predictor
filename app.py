import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Predictor", page_icon="🚗")

# White UI
st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #ffffff;
    color: #000000;
}

/* Headings and text */
h1, h2, h3, h4, h5, h6, p, label {
    color: #000000 !important;
}

/* Input fields */
input, textarea {
    color: black !important;
    background-color: #f9f9f9 !important;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #f9f9f9 !important;
    color: black !important;
}

/* Dropdown text */
div[data-baseweb="select"] span {
    color: black !important;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: bold;
}

/* Slider text */
.stSlider label {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Valid brands
valid_brands = [
    "maruti", "hyundai", "honda", "toyota", "ford", "bmw",
    "audi", "mercedes", "tata", "mahindra", "kia", "skoda",
    "volkswagen", "nissan", "renault", "chevrolet"
]

# Header
st.title("🚗 Car Price Intelligence System")
st.markdown("### 👋 Hi car enthusiast!")

# Inputs
car_name = st.text_input("Car Name (e.g., Honda City)")

col1, col2 = st.columns(2)

with col1:
    year = st.slider("Year", 2000, 2025)
    present_price = st.number_input("Original Price (in lakhs)")
    kms_driven = st.number_input("Kilometers Driven")

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Number of Previous Owners", [1,2,3,4])

# Condition
st.subheader("🔍 Car Condition")

scratches = st.selectbox("Scratches", ["None", "Minor", "Major"])
paint_faded = st.selectbox("Paint Faded", ["No", "Yes"])
repainted = st.selectbox("Repainted", ["No", "Yes"])
accident_history = st.selectbox("Accident History", ["No", "Yes"])
service_history = st.selectbox("Service History", ["Yes", "No"])

engine_condition = st.slider("Engine", 1, 5)
brake_condition = st.slider("Brakes", 1, 5)
tire_condition = st.slider("Tyres", 1, 5)

# Encoding
car_age = 2025 - year
fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
seller_map = {"Dealer":0, "Individual":1}
trans_map = {"Manual":0, "Automatic":1}

# Predict
if st.button("Predict"):

    # Validation
    if not car_name:
        st.error("❌ Please enter a car name")
        st.stop()

    brand = car_name.split(" ")[0].lower()
    if brand not in valid_brands:
        st.error("❌ Invalid car brand")
        st.stop()

    # Model input
    input_data = np.array([[
        present_price,
        kms_driven,
        fuel_map[fuel_type],
        seller_map[seller_type],
        trans_map[transmission],
        owner,
        car_age
    ]])

    base_price = model.predict(input_data)[0]

    # Condition penalty
    penalty = 0
    if scratches == "Minor": penalty += 0.03
    elif scratches == "Major": penalty += 0.08
    if paint_faded == "Yes": penalty += 0.05
    if repainted == "Yes": penalty += 0.04
    if accident_history == "Yes": penalty += 0.10
    if service_history == "No": penalty += 0.07

    mech_factor = (engine_condition + brake_condition + tire_condition) / 15
    usage_penalty = min(kms_driven / 100000, 1) * 0.15
    owner_penalty = owner * 0.04

    final_price = base_price * (1 - penalty - usage_penalty - owner_penalty) * mech_factor
    final_price = max(0, final_price)

    st.success(f"💰 Estimated Price: ₹ {round(final_price,2)} lakhs")

    # Graph
    st.subheader("📉 Dynamic Depreciation")

    years = list(range(year, 2026))
    prices = []

    base_rate = 0.12 if fuel_type == "Petrol" else 0.09
    dynamic_rate = base_rate + penalty + usage_penalty + owner_penalty + (1 - mech_factor)*0.1
    dynamic_rate = min(dynamic_rate, 0.35)

    for y in years:
        age = y - year
        price = present_price * ((1 - dynamic_rate) ** age)
        price *= (1 - 0.02 * age)
        prices.append(price)

    prices[-1] = final_price

    fig, ax = plt.subplots()
    ax.plot(years, prices, marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Realistic Depreciation Curve")
    st.pyplot(fig)

    # Score
    score = (mech_factor * 100) - (penalty * 100) - (usage_penalty * 100)

    st.subheader("🔎 Condition Score")
    st.write(f"{round(score,1)}/100")
    st.progress(max(0, min(score/100, 1)))

    # Recommendation
    st.subheader("🧠 Recommendation")
    if score > 70:
        st.success("✅ Strong Buy")
    elif score > 50:
        st.warning("⚠️ Inspect before buying")
    else:
        st.error("❌ Avoid")

    # Maintenance Advice
    st.subheader("🛠️ Maintenance Advice")

    advice_given = False

    if tire_condition <= 2:
        st.warning("⚠️ Replace tyres immediately")
        advice_given = True
    elif tire_condition == 3:
        st.info("Tyres moderately worn")
        advice_given = True

    if brake_condition <= 2:
        st.warning("⚠️ Check brake pads")
        advice_given = True
    elif brake_condition == 3:
        st.info("Brake inspection needed")
        advice_given = True

    if engine_condition <= 2:
        st.warning("⚠️ Engine servicing required")
        advice_given = True
    elif engine_condition == 3:
        st.info("Engine average condition")
        advice_given = True

    if not advice_given:
        st.success("✅ Car is in excellent condition")