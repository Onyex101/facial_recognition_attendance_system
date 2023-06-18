import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Registration", layout="centered")
st.subheader("Registration form")
# collect person name and role
# initialize registration form
registration_form = face_rec.RegistrationForm()
person_name = st.text_input(label="Name",placeholder="First & Last Name")
role = st.selectbox(label="Select your role",options=("Student","Teacher"))

# collect facial embedding of person
# callback function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24") # 3D numpy array
    # operations that can be performed on the array
    reg_img, embedding = registration_form.get_embeddings(img)
    # save data into local computer as .txt
    if embedding is not None:
        with open("face_embedding.txt",mode="ab") as f:
            np.savetxt(f,embedding)
    
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registration", video_frame_callback=video_frame_callback)

# save data to redis database
if st.button("Submit"):
    return_val = registration_form.save_data_redis(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered succesfully!")
    elif return_val == "name_false":
        st.error("Please enter Name: Name cannot be empty or spaces")
    elif return_val == "file_false":
        st.error("face_embedding.txt is not found. Please refresh page and execute again.")