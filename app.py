import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Predictor", page_icon="🚗")

# -----------------------------
# UI STYLE
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #000000; }
h1, h2, h3, h4, h5, h6, p, label { color: #000000 !important; }
input, textarea { color: black !important; background-color: #f9f9f9 !important; }
div[data-baseweb="select"] { background-color: #f9f9f9 !important; color: black !important; }
div[data-baseweb="select"] span { color: black !important; }
.stButton>button { background-color: #2563eb; color: white; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# VALID BRANDS
# -----------------------------
valid_brands = ["maruti","hyundai","honda","toyota","ford","bmw","audi","mercedes",
"tata","mahindra","kia","skoda","volkswagen","nissan","renault","chevrolet"
"jaguar","land","volvo","mini","fiat","isuzu","mitsubishi","porsche",
"subaru","suzuki","datsun","lamborghini","ferrari","rolls","bentley","aston","maserati","bugatti", "alfa","romeo","lexus","infiniti","acura","lincoln","cadillac","genesis","tesla","lucid","rivian","fisker","byd","nio","xpeng","wm","great","baic","changan","gac","geely","haval","mg","ora","sehol","seres","voyah","zhongtong","kinglong","yutong","foton","ashok","leyland","eicher","bajaj",]

# -----------------------------
# HEADER
# -----------------------------
st.title("Used Car Price Predictor")
st.markdown("#### Hi enthusiast!, we will guide you with your upcoming car.")

# -----------------------------
# INPUTS
# -----------------------------
car_name = st.text_input("Car Name (Please enter along with brand name)")

col1, col2 = st.columns(2)

with col1:
    year = st.slider("Year", 2000, 2025)
    present_price = st.number_input("Original Price (in lakhs)")
    kms_driven = st.number_input("Kilometers Driven")

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol","Diesel","CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer","Individual"])
    transmission = st.selectbox("Transmission", ["Manual","Automatic"])
    owner = st.selectbox("Owners", [1,2,3,4])

# -----------------------------
# CONDITION INPUT
# -----------------------------
st.subheader("🔍 Car Condition")

scratches = st.selectbox("Scratches", ["None","Minor","Major"])
paint_faded = st.selectbox("Paint Faded", ["No","Yes"])
repainted = st.selectbox("Repainted", ["No","Yes"])
accident_history = st.selectbox("Accident History", ["No","Yes"])
service_history = st.selectbox("Service History", ["Yes","No"])

engine_condition = st.slider("Engine",1,5)
brake_condition = st.slider("Brakes",1,5)
tire_condition = st.slider("Tyres",1,5)

# -----------------------------
# ENCODING
# -----------------------------
car_age = 2025 - year
fuel_map = {"Petrol":0,"Diesel":1,"CNG":2}
seller_map = {"Dealer":0,"Individual":1}
trans_map = {"Manual":0,"Automatic":1}

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict"):

    # Validation
    if not car_name:
        st.error("Enter car name")
        st.stop()

    brand = car_name.split(" ")[0].lower()
    if brand not in valid_brands:
        st.error("Invalid car brand")
        st.stop()

    # Model input
    input_data = np.array([[present_price,kms_driven,
                            fuel_map[fuel_type],
                            seller_map[seller_type],
                            trans_map[transmission],
                            owner,car_age]])

    base_price = model.predict(input_data)[0]

    # -----------------------------
    # CONDITION LOGIC
    # -----------------------------
    penalty = 0

    if scratches == "Minor": penalty += 0.03
    elif scratches == "Major": penalty += 0.08
    if paint_faded == "Yes": penalty += 0.05
    if repainted == "Yes": penalty += 0.04
    if accident_history == "Yes": penalty += 0.10
    if service_history == "No": penalty += 0.07

    mech_factor = (engine_condition + brake_condition + tire_condition)/15
    usage_penalty = min(kms_driven/100000,1)*0.15
    owner_penalty = owner*0.04

    final_price = base_price*(1-penalty-usage_penalty-owner_penalty)*mech_factor
    final_price = max(0, final_price)

    st.success(f"Estimated Price: ₹ {round(final_price,2)} lakhs")

    # -----------------------------
    # DYNAMIC GRAPH (FIXED)
    # -----------------------------
    st.subheader("Depreciation Curve")

    years = list(range(year,2026))
    prices = []

    for i,y in enumerate(years):

        age = y - year

        if age<=1: dep_rate=0.15
        elif age<=3: dep_rate=0.10
        elif age<=5: dep_rate=0.08
        else: dep_rate=0.06

        km_factor=min(kms_driven/100000,1)*0.05
        owner_factor=owner*0.015
        condition_factor=(1-mech_factor)*0.08

        extra=0
        if accident_history=="Yes": extra+=0.05
        if repainted=="Yes": extra+=0.03
        if paint_faded=="Yes": extra+=0.02

        final_rate=min(dep_rate+km_factor+owner_factor+condition_factor+extra,0.30)

        if i==0:
            price=present_price
        else:
            price=prices[-1]*(1-final_rate)

        prices.append(price)

    prices[-1]=final_price

    fig,ax=plt.subplots()
    ax.plot(years,prices,marker='o')
    ax.set_xlabel("Year")
    ax.set_ylabel("Price")
    ax.set_title("Car Depreciation")
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------
    # SCORE
    # -----------------------------
    score=(mech_factor*100)-(penalty*100)-(usage_penalty*100)

    st.subheader("🔎 Condition Score")
    st.write(f"{round(score,1)}/100")
    st.progress(max(0,min(score/100,1)))

    # -----------------------------
    # RECOMMENDATION
    # -----------------------------
    st.subheader("🧠 Recommendation")

    if score>70:
        st.success("Strong Buy")
    elif score>50:
        st.warning("Inspect before buying")
    else:
        st.error("Avoid")

    # -----------------------------
    # MAINTENANCE
    # -----------------------------
    st.subheader("🛠️ Maintenance Advice")

    advice=False

    if tire_condition<=2:
        st.warning("Replace tyres")
        advice=True
    if brake_condition<=2:
        st.warning("Check brakes")
        advice=True
    if engine_condition<=2:
        st.warning("Service engine")
        advice=True

    if not advice:
        st.success("Car in excellent condition")