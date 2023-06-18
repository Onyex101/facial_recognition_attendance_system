import numpy as np
import pandas as pd
import cv2
import os
import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time
from datetime import datetime

# Connect to Redis Client
hostname = "redis-12004.c13.us-east-1-3.ec2.cloud.redislabs.com"
port = 12004
password = "5w8GS5MT0PujpWOk8kkKL06Av1gapTTq"

r = redis.StrictRedis(host=hostname, port=port, password=password)


# Retrieve data from Database
def retrieve_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(
        lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role', 'facial_features']
    retrive_df[['name', 'role']] = retrive_df['name_role'].apply(
        lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[["name","role","facial_features"]]


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_l', root='insightface_model', providers=[
                       'CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

# ML Search Algorithm


def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_role=['name', 'role'], thresh=0.5):
    """
    cosine similarity base search algorithm
    """
    # step-1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    # step-2: Index face embeding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # step-3: Cal. cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step-4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # step-5: get the person name
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]

    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


### Real Time Prediction
# save logs for every minute
class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def save_logs(self):
        # create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        # drop duplicate information (distinct name)
        dataframe.drop_duplicates("name",inplace=True)
        # push data to redis database (redis list)
        # encode the data to string
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name != "Unknown":
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) > 0:
            r.lpush("attendance:logs",*encoded_data)
        self.reset_dict()

    
    def face_prediction(self,test_image, dataframe, feature_column,
                        name_role=['name', 'role'], thresh=0.5):
        # step-0: find the current time
        current_time = str(datetime.now())
        # step-1: take the test image and apply to insight face
        results = faceapp.get(test_image)
        test_copy = test_image.copy()

        # step-2: use for loop and extract each embedding and pass to ml_search_algorithm
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                        feature_column,
                                                        test_vector=embeddings,
                                                        name_role=name_role,
                                                        thresh=thresh)
            if person_name == 'Unknown':
                color = (0, 0, 255)  # bgr
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1-5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2+20),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            # save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


### Registration form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
        
    def reset(self):
        self.sample = 0
        
    def get_embeddings(self, frame):
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            text = f"samples = {self.sample}"
            # put text sample into frame
            cv2.putText(frame, text, (x1, y1-5),cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,0), 2)
            # facial features
            embeddings = res['embedding']
        return frame, embeddings
    
    def save_data_redis(self,name,role):
        # validate name
        if name is not None:
            if name.strip() != "":
                key = f"{name}@{role}"
            else:
                return "name_false"
        else:
            return "name_false"
        # check if face_embedding.txt exists
        if "face_embedding.txt" not in os.listdir():
            return "file_false"
        # load face_embedding.txt file
        x_array = np.loadtxt("face_embedding.txt",dtype=np.float32)
        # convert into array into proper shape
        recieved_samples = int(x_array.size/512)
        x_array = x_array.reshape(recieved_samples,512)
        x_array = np.asarray(x_array)
        # calculate mean embeddings
        x_mean = x_array.mean(axis=0).astype(np.float32)
        x_mean_bytes = x_mean.tobytes()
        # save to redis database
        r.hset(name='academy:register',key=key,value=x_mean_bytes)
        # remove file
        os.remove("face_embedding.txt")
        self.reset()
        return True