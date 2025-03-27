import streamlit as st
import numpy as np
import joblib

MODEL_PATH = "mushroom_classifier.pkl"  
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

feature_names = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor", "gill-attachment",
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat"
]

feature_options = {
    "cap-shape": ["bell", "conical", "convex", "flat", "knobbed", "sunken"],
    "cap-surface": ["fibrous", "grooves", "scaly", "smooth"],
    "cap-color": ["brown", "buff", "cinnamon", "gray", "green", "pink", "purple", "red", "white", "yellow"],
    "bruises": ["no", "yes"],
    "odor": ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"],
    "gill-attachment": ["attached", "descending", "free", "notched"],
    "gill-spacing": ["close", "crowded", "distant"],
    "gill-size": ["broad", "narrow"],
    "gill-color": ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"],
    "stalk-shape": ["enlarging", "tapering"],
    "stalk-root": ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"],
    "stalk-surface-above-ring": ["fibrous", "scaly", "silky", "smooth"],
    "stalk-surface-below-ring": ["fibrous", "scaly", "silky", "smooth"],
    "stalk-color-above-ring": ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"],
    "stalk-color-below-ring": ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"],
    "veil-type": ["partial", "universal"],
    "veil-color": ["brown", "orange", "white", "yellow"],
    "ring-number": ["none", "one", "two"],
    "ring-type": ["cobwebby", "evanescent", "flaring", "large", "none", "pendant", "sheathing", "zone"],
    "spore-print-color": ["black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"],
    "population": ["abundant", "clustered", "numerous", "scattered", "several", "solitary"],
    "habitat": ["grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"]
}

st.title("Mushroom Classification App")
st.write("Select the characteristics of a mushroom to predict whether it's **Edible** or **Poisonous**.")

user_input = []
for feature in feature_names:
    selected_value = st.selectbox(f"{feature.replace('-', ' ').capitalize()}:", feature_options[feature])
    user_input.append(feature_options[feature].index(selected_value))  # Convert to numerical value

if st.button("Predict"):
    try:
        input_array = np.array([user_input]).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "Edible" if prediction == 0 else "Poisonous"
        st.success(f"**Prediction: {result}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
