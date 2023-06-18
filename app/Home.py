import streamlit as st

st.set_page_config(page_title="Attendance System", layout="wide")
st.header("Attendance System using Facial Recognition")

with st.spinner("Loading Model and Connecting to redis DB..."):
    import face_rec

st.success("Model Loaded Succesfully")
st.success("Redis DB Succesfully connected")