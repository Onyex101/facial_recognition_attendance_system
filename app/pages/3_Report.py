import streamlit as st
from Home import face_rec

st.set_page_config(page_title="Report", layout="wide")
st.subheader("Reporting")
# retrieve log data
# extract data from redis list
name = "attendance:logs"
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list


# tabs to show the info
tab1, tab2 = st.tabs(["Registered Data","Logs"])

with tab1:
    if st.button("Refresh Data"):
    # retrieve data from redis database
        with st.spinner("Retrieveing Data from Redis DB..."):
            redis_face_db = face_rec.retrieve_data(name="academy:register")
            st.dataframe(redis_face_db[["name","role"]])
            
with tab2:
    if st.button("Refresh Logs"):
        st.write(load_logs(name=name))

