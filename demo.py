import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# =======================
# Load Models & Data
# =======================
# Load ML model
model = pickle.load(open("svc.pkl", "rb"))

# Load training data for structure
df = pd.read_csv("training.csv")

# Prepare LabelEncoder
le = LabelEncoder()
le.fit(df['prognosis'])

# List of symptoms (all feature columns)
all_symptoms = df.drop('prognosis', axis=1).columns.tolist()

# Load CSV files for extra info
sym_des = pd.read_csv("symtoms_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
workout = pd.read_csv("workout_df.csv")
diets = pd.read_csv("diets.csv")
precautions = pd.read_csv("precautions_df.csv")

# =======================
# Helper Functions
# =======================
def predict_disease(symptoms_text):
    user_symptoms = [s.strip() for s in symptoms_text.split(",")]

    # Create 0/1 vector
    input_vector = [0] * len(all_symptoms)
    for symptom in user_symptoms:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1

    # Predict with model
    try:
        pred_encoded = model.predict([input_vector])[0]
        prediction = le.inverse_transform([pred_encoded])[0]
    except Exception as e:
        prediction = f"Prediction error: {e}"

    return prediction



def get_description(disease):
    desc = description[description['Disease'] == disease]['Description'].values
    return desc[0] if len(desc) > 0 else "No description available"

def get_medications(disease):
    meds = medications[medications['Disease'] == disease].iloc[:, 1:].values.flatten().tolist()
    meds = [m for m in meds if str(m).strip() not in ['nan', '', 'None']]
    return meds if len(meds) > 0 else ["No medications available"]


def get_workout(disease):
    wrk = workout[workout['disease'] == disease]['workout'].values.flatten()
    wrk = [w for w in wrk if str(w) != 'nan']
    return wrk if len(wrk) > 0 else ["No workout available"]

def get_diets(disease):
    diet = diets[diets['Disease'] == disease].iloc[:, 1:].values.flatten()
    diet = [d for d in diet if str(d) != 'nan']
    return diet if len(diet) > 0 else ["No diets available"]

def get_precautions(disease):
    prec = precautions[precautions['Disease'] == disease].iloc[:, 1:].values.flatten()
    prec = [p for p in prec if str(p) != 'nan']
    return prec if len(prec) > 0 else ["No precautions available"]
# =======================
# Streamlit UI
# =======================
st.set_page_config(page_title="Health Mate AI", layout="wide")

st.sidebar.title("Health Mate AI")

st.markdown("<h2 style='text-align: center;'>Enter Symptoms</h2>", unsafe_allow_html=True)

# Input box
symptoms_input = st.text_input("Enter symptoms (comma separated)", "")

# Buttons
col1, col2, col3, col4, col5,col6 = st.columns(6)
with col1:
    predict_btn = st.button("Predict Disease")
with col2:
    desc_btn = st.button("Description")
with col3:
    meds_btn = st.button("Medications")
with col4:
    workout_btn = st.button("Workout")
with col5:
    diets_btn = st.button("Diets")
with col6:
    precaution_btn = st.button("Precautions")
# Global placeholder for results
if "disease" not in st.session_state:
    st.session_state["disease"] = None

# Prediction
if predict_btn and symptoms_input:
    # Convert symptoms_input into features (implement preprocessing as in notebook)
    st.write("### Disease Prediction:")
    st.session_state["disease"] = predict_disease(symptoms_input)
    st.error(f"Predicted Disease: {st.session_state['disease']}")

# Description
if desc_btn:
    if st.session_state["disease"]:
        st.write("### Description of disease :")
        st.info(get_description(st.session_state["disease"]))
    else:
        st.warning("Please predict disease first.")




# Medications
if meds_btn:
    if st.session_state["disease"]:
        meds = get_medications(st.session_state["disease"])
        st.write("### Recommended Medications:")
        for m in meds:
            st.success(f"- {m}")   # ✅ shows one by one
    else:
        st.warning("Please predict disease first.")


# Workout
if workout_btn:
    if st.session_state["disease"]:
        wrk = get_workout(st.session_state["disease"])
        # orange heading
        st.markdown("### :orange[Suggested Workouts:]")
        for w in wrk:
            st.markdown(f":orange[- {w}]")
    else:
        st.warning("Please predict disease first.")



# Diets
if diets_btn:
    if st.session_state["disease"]:
        diet = get_diets(st.session_state["disease"])
        st.write("### Suggested Diets:")
        for d in diet:
            st.success(f"- {d}")
    else:
        st.warning("Please predict disease first.")

# Precautions
if precaution_btn:
    if st.session_state["disease"]:
        prec = get_precautions(st.session_state["disease"])
        st.write("### Suggested Precautions:")
        for p in prec:
            st.warning(f"- {p}")
    else:
        st.warning("Please predict disease first.")
