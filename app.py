
import uvicorn as uv
from fastapi import FastAPI
from iris_format import Iris_Fromat
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from pydantic import BaseModel



app = FastAPI()
# with tf.device('CPU'):
#     model = tf.keras.Sequential(
#         [tf.keras.layers.Dense(4, activation = 'relu', input_shape=(None,4)),
#         tf.keras.layers.Dense(3, activation='relu'),
#         tf.keras.layers.Dense(3, activation='softmax')]
#     )
#     model.load_weights(r"C:\Users\Siddhartha Devan V\jupyter ml\fast_api\tutorial\iris_classifier_1.h5")
# model = tf.keras.models.load_model(r"C:\Users\Siddhartha Devan V\jupyter ml\fast_api\tutorial\iris_classifier_1.0")

pick_in =  open('iris_mod.pkl', 'rb')
loaded_model = pickle.load(pick_in)

class Iris_Fromat(BaseModel):
    sepal_length: float 
    sepal_width: float 
    petal_length: float 
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# @app.get('/name')
# def get_name(name:str):
#     return {"welcome":f"{name}"}
# class dummy_model:
#     def predict(self, preds):
#         return np.array([1])

# loaded_model = dummy_model()

@app.post('/predict')
def predict_species(data:Iris_Fromat):
    print(type(data))
    # data = data.
    print(data)
    print("hi")

    sepal_length = data.sepal_length
    print("sepal_lenght", sepal_length)
    sepal_width = data.sepal_width
    petal_length = data.petal_length
    petal_width = data.petal_width

    print(loaded_model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]])))
    prediction = loaded_model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
    print("pred:", prediction)
    prediction = prediction.tolist()
    return {
        "predicted":prediction
    }

if __name__ == "__main__":
    uv.run(app, host = "127.0.0.1", port=8000)
    

