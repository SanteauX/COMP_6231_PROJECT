# Code that get conteneurized
import boto3
import os, shutil
import keras
import cv2
import numpy as np
import base64
import json

from cv2 import resize

__S3_BUCKET_NAME = 'comp6231-demo-s3-bucket'
__MODEL_FILENAME = 'model.h5'
s3 = boto3.client('s3')

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

lesion_ID_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}

def clean_local_temp_folder():
    folder = '/tmp'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            pass

def lambda_handler(event, context):

    if 'body' not in event.keys():
        return {
            "statusCode" : 403,
            "headers" : {
                "Content-Type" : "application/json"
            },
            "body" : {
                "result" : "failure",
                "reason" : "bad request"
            },
            "isBase64Encoded" : False
        }

    # see if model exists in S3 already
    modelExists = False

    try:
        s3 = boto3.resource('s3')
        s3.Object(__S3_BUCKET_NAME, __MODEL_FILENAME).load()
        modelExists = True
    except Exception as e:
        pass

    if not modelExists:
        return {
            "statusCode" : 404,
            "headers" : {
                "Content-Type" : "application/json"
            },
            "body" : {
                "result" : "failure",
                "reason" : "model doesn't exist in S3 bucket"
            },
            "isBase64Encoded" : False
        }

    model = None
    
    if modelExists:
        try:
            clean_local_temp_folder()
            print(f"downloading model from S3 bucket=[{__S3_BUCKET_NAME}], key={__MODEL_FILENAME}")
            s3.Bucket(__S3_BUCKET_NAME).download_file(__MODEL_FILENAME, '/tmp/model.h5')
            model = keras.models.load_model("/tmp/model.h5")
            clean_local_temp_folder()
        except Exception as e:
            raise

    X = []

    img = event['body']
    imgDecoded = base64.b64decode(img)
    decoded = cv2.imdecode(np.frombuffer(imgDecoded, np.uint8), -1)

    img2 = resize(decoded,(100,100))

    X.append(img2)
    X = np.array(X)

    result = None
    result = model.predict(X)

    resultDict = {}

    #iterate through lesion_ID_dict and lesion_type_dict. build response with model predictions
    for i, value in enumerate(result[0]):
        key = None
        for k,v in lesion_ID_dict.items():
            if i == v:
                key = k
        for k,v in lesion_type_dict.items():
            if key == k:
                key = v
        resultDict[key] = str(value)

    return {
        "statusCode" : 200,
        "headers" : {
            "Content-Type" : "application/json"
        },
        "body" : str(resultDict),
        "isBase64Encoded" : False
    }

    # step 1 connect to S3
    # step 2 get model
    # step 3 get model predicting
    # step 4 convert image to bytestream
    # step 5 run image against model for prediction
    # step 6 return prediciton back to API