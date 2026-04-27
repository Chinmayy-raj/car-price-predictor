import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Predictor", page_icon="🚗")

# White UI
st.markdown("""
<style>
.stApp { background-color: white; color: black; }
</style>
""", unsafe_allow_html=True)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# VALID CAR BRANDS
# -----------------------------
valid_brands = [
    "maruti", "hyundai", "honda", "toyota", "ford", "bmw",
    "audi", "mercedes", "tata", "mahindra", "kia", "skoda",
    "volkswagen", "nissan", "renault", "chevrolet"
]

# -----------------------------
# HEADER
# -----------------------------
st.title("🚗 Car Price Intelligence System")
st.markdown("### 👋 Hi car enthusiast!")

# -----------------------------
# INPUTS
# -----------------------------
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

# -----------------------------
# CONDITION INPUT
# -----------------------------
st.subheader("🔍 Car Condition")

scratches = st.selectbox("Scratches", ["None", "Minor", "Major"])
paint_faded = st.selectbox("Paint Faded", ["No", "Yes"])
repainted = st.selectbox("Repainted", ["No", "Yes"])
accident_history = st.selectbox("Accident History", ["No", "Yes"])
service_history = st.selectbox("Service History", ["Yes", "No"])

engine_condition = st.slider("Engine", 1, 5)
brake_condition = st.slider("Brakes", 1, 5)
tire_condition = st.slider("Tyres", 1, 5)

# -----------------------------
# ENCODING
# -----------------------------
car_age = 2025 - year

fuel_map = {"Petrol":0, "Diesel":1, "CNG":2}
seller_map = {"Dealer":0, "Individual":1}
trans_map = {"Manual":0, "Automatic":1}

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):

    # ---------- VALIDATION ----------
    if not car_name:
        st.error("❌ Please enter a car name")
        st.stop()

    brand = car_name.split(" ")[0].lower()

    if not any(b in brand for b in valid_brands):
        st.error("❌ Invalid car name. Use real brands like Honda, BMW, Hyundai")
        st.stop()

    # ---------- MODEL ----------
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

    # ---------- CONDITION PENALTY ----------
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

    # ---------- DYNAMIC GRAPH ----------
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

    # ---------- SCORE ----------
    score = (mech_factor * 100) - (penalty * 100) - (usage_penalty * 100)

    st.subheader("🔎 Condition Score")
    st.write(f"{round(score,1)}/100")
    st.progress(max(0, min(score/100, 1)))

    # ---------- RECOMMEND ----------
    st.subheader("🧠 Recommendation")

    if score > 70:
        st.success("✅ Strong Buy")
    elif score > 50:
        st.warning("⚠️ Inspect before buying")
    else:
        st.error("❌ Avoid")

    # ---------- ADVICE ----------
  
        # -----------------------------
# 🛠️ MAINTENANCE ADVICE (FINAL)
# -----------------------------
st.subheader("🛠️ Maintenance Advice")

advice_given = False

# Tyres
if tire_condition <= 2:
    st.warning("⚠️ Tyres are heavily worn. Replace them immediately for safe driving and better grip.")
    advice_given = True
elif tire_condition == 3:
    st.info("🔧 Tyres are moderately worn. Consider replacing them soon for optimal performance.")
    advice_given = True

# Brakes
if brake_condition <= 2:
    st.warning("⚠️ Brake condition is poor. Inspect brake pads immediately and replace if worn out.")
    advice_given = True
elif brake_condition == 3:
    st.info("🔧 Brake performance is average. Schedule a check-up soon.")
    advice_given = True

# Engine
if engine_condition <= 2:
    st.warning("⚠️ Engine condition is not good. Immediate servicing is recommended to avoid breakdowns.")
    advice_given = True
elif engine_condition == 3:
    st.info("🔧 Engine performance is average. Regular servicing is advised.")
    advice_given = True

# Positive case
if not advice_given:
    st.success("✅ Your car is in great condition! Enjoy a smooth and safe driving experience 🚗💨")