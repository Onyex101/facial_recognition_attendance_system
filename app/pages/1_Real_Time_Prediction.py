import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title="Predictions")
st.subheader("Real-Time Attendance System")
# retrieve data from redis database
with st.spinner("Retrieveing Data from Redis DB..."):
    redis_face_db = face_rec.retrieve_data(name="academy:register")
    st.dataframe(redis_face_db)

st.success("Data Successfully retrieved from redis!")

# time
waitTime = 30 # time in seconds
setTime = time.time()
realTimePred = face_rec.RealTimePred() # real time prediction class
# real time prediction
# streamlit webrtc
# callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") # 3D numpy array
    # operations that can be performed on the array
    pred_img = realTimePred.face_prediction(img, redis_face_db, "facial_features",["name","role"])
    
    timenow = time.time()
    diftime = timenow - setTime
    if diftime >= waitTime:
        realTimePred.save_logs()
        setTime = time.time() # reste time
        print("Saved Log Data to redis database")
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realtimePrediction", video_frame_callback=video_frame_callback)