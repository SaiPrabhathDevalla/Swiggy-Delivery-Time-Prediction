import pandas as pd
import pickle
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Swiggy ETA Predictor", layout="centered")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("swiggy_demographic.csv")
    df.dropna(inplace=True)
    df.drop(columns=["rider_id", "order_date"], inplace=True)
    return df

df = load_data()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

Model = load_model()

# ---------------- HEADER ----------------
st.image("https://m.economictimes.com/thumb/msid-113428241,width-1600,height-900,resizemode-4,imgsize-180876/swiggy-instamart.jpg")
st.title("Swiggy Delivery Time Prediction")
st.write("Enter all details below to predict Estimated Delivery Time (ETA)")

# ---------------- ALL INPUTS IN ONE CONTINUOUS FLOW ----------------

AGE = st.slider("Rider Age (years)", 
                int(df["age"].min()), 
                int(df["age"].max()), 
                int(df["age"].mean()))

RATINGS = st.slider("Rider Rating (2.5 - 5.0)", 
                    float(df["ratings"].min()), 
                    float(df["ratings"].max()), 
                    float(df["ratings"].mean()))

RESTAURANT_LATITUDE = st.number_input(
    "Restaurant Latitude",
    float(df["restaurant_latitude"].min()),
    float(df["restaurant_latitude"].max()),
    float(df["restaurant_latitude"].mean()),
    format="%.6f"
)

RESTAURANT_LONGITUDE = st.number_input(
    "Restaurant Longitude",
    float(df["restaurant_longitude"].min()),
    float(df["restaurant_longitude"].max()),
    float(df["restaurant_longitude"].mean()),
    format="%.6f"
)

DELIVERY_LATITUDE = st.number_input(
    "Delivery Latitude",
    float(df["delivery_latitude"].min()),
    float(df["delivery_latitude"].max()),
    float(df["delivery_latitude"].mean()),
    format="%.6f"
)

DELIVERY_LONGITUDE = st.number_input(
    "Delivery Longitude",
    float(df["delivery_longitude"].min()),
    float(df["delivery_longitude"].max()),
    float(df["delivery_longitude"].mean()),
    format="%.6f"
)

WEATHER = st.selectbox("Weather Condition", df["weather"].unique())
TRAFFIC = st.selectbox("Traffic Level", df["traffic"].unique())

VEHICLE_CONDITION = st.selectbox(
    "Vehicle Condition (0 = Poor, 1 = Average, 2 = Good)",
    sorted(df["vehicle_condition"].unique())
)

TYPE_OF_ORDER = st.selectbox("Type of Order", df["type_of_order"].unique())
TYPE_OF_VEHICLE = st.selectbox("Type of Vehicle", df["type_of_vehicle"].unique())

MULTIPLE_DELIVERIES = st.selectbox(
    "Multiple Deliveries Assigned",
    sorted(df["multiple_deliveries"].unique())
)

FESTIVAL = st.selectbox("Festival Day?", df["festival"].unique())
CITY_TYPE = st.selectbox("City Type", df["city_type"].unique())
CITY_NAME = st.selectbox("City Name", df["city_name"].unique())

ORDER_DAY = st.slider("Day of Month (1 - 31)", 1, 31, int(df["order_day"].mean()))
ORDER_MONTH = st.slider("Month (1 - 12)", 1, 12, int(df["order_month"].mean()))
ORDER_DAY_OF_WEEK = st.selectbox("Day of Week", df["order_day_of_week"].unique())

IS_WEEKEND = st.selectbox("Is Weekend? (0 = No, 1 = Yes)", [0, 1])

PICKUP_TIME_MINUTES = st.slider(
    "Restaurant Preparation Time (minutes)",
    float(df["pickup_time_minutes"].min()),
    float(df["pickup_time_minutes"].max()),
    float(df["pickup_time_minutes"].mean())
)

ORDER_TIME_HOUR = st.slider(
    "Order Placed Hour (0 - 23)",
    0, 23, int(df["order_time_hour"].mean())
)

ORDER_TIME_OF_DAY = st.selectbox("Time of Day", df["order_time_of_day"].unique())

DISTANCE = st.slider(
    "Distance Between Restaurant & Customer (km)",
    float(df["distance"].min()),
    float(df["distance"].max()),
    float(df["distance"].mean())
)

# ---------------- CREATE INPUT DATA ----------------
input_data = pd.DataFrame({
    "age": [AGE],
    "ratings": [RATINGS],
    "restaurant_latitude": [RESTAURANT_LATITUDE],
    "restaurant_longitude": [RESTAURANT_LONGITUDE],
    "delivery_latitude": [DELIVERY_LATITUDE],
    "delivery_longitude": [DELIVERY_LONGITUDE],
    "weather": [WEATHER],
    "traffic": [TRAFFIC],
    "vehicle_condition": [VEHICLE_CONDITION],
    "type_of_order": [TYPE_OF_ORDER],
    "type_of_vehicle": [TYPE_OF_VEHICLE],
    "multiple_deliveries": [MULTIPLE_DELIVERIES],
    "festival": [FESTIVAL],
    "city_type": [CITY_TYPE],
    "city_name": [CITY_NAME],
    "order_day": [ORDER_DAY],
    "order_month": [ORDER_MONTH],
    "order_day_of_week": [ORDER_DAY_OF_WEEK],
    "is_weekend": [IS_WEEKEND],
    "pickup_time_minutes": [PICKUP_TIME_MINUTES],
    "order_time_hour": [ORDER_TIME_HOUR],
    "order_time_of_day": [ORDER_TIME_OF_DAY],
    "distance": [DISTANCE]
})

# ---------------- PREDICTION ----------------
if st.button("Predict Delivery Time"):
    input_data = input_data[Model.feature_names_in_]
    result = Model.predict(input_data)
    st.success(f"Estimated Delivery Time: {result[0]:.2f} minutes")