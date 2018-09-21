# -*- coding: utf-8 -*-

import logging
import pickle
import pandas as pd
import numpy as np
from keras.models import model_from_json
from sklearn.feature_extraction.text import TfidfVectorizer


class model():
    def train_model():
        """Model training

        """
        raise NotImplemented()

    def save_pipeline():
        """Save the pipeline with the trained model

        """
        raise NotImplemented()

    def load_pipeline():
        """Loading the pipeline with the trained model

        """
        log = logging.getLogger('load_pipeline')

        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("model.h5") # CONFIDENTIAL
        log.info("Loaded model from disk")
        loaded_model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy'])

        return loaded_model

    def clean_and_predict(corpus):
        """Pre-processing and makes predictions on a batch

        """
        log = logging.getLogger('clean_and_predict')

        tags = pd.read_csv('clean_tags.csv', sep=';', header=-1)
        loaded_model = model()
        #Load TF-IDF
        loaded_vec = TfidfVectorizer(
            decode_error="replace",
            vocabulary=pickle.load(open("feature.pkl", "rb")))
        tfidf = loaded_vec.fit_transform(np.array([corpus], dtype=object))
        log.info("Prediction...")
        pred = loaded_model.predict(tfidf)
        pred[pred > 0.5] = 1
        _, index = np.where(pred == 1)
        y_pred = tags.iloc[index, :]
        log.debug("y_pred: {}" .format(y_pred))

        return y_pred

    def eval_model():
        """Evaluates the model predictions

        """
        raise NotImplemented()

