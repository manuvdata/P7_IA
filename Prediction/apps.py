from django.apps import AppConfig
import pandas as pd
from joblib import load
import os
from transformers import BertTokenizer, TFBertModel, BertConfig, BertForSequenceClassification, BertForTokenClassification
from transformers import *
import gdown

class PredictionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Prediction'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CLASSIFIER_FOLDER = os.path.join(BASE_DIR, 'Prediction/bert/')
    url = 'https://drive.google.com/uc?id=1211JqjGW5zYkJMO1bVjsrsS0Xd6mmX_V'
    output = 'manugdown.h5'
    model_save_path =gdown.download(url,output)
 
    num_classes=2
    bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    #model_save_path=os.path.join(CLASSIFIER_FOLDER, "bert_model.h5")
    trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    trained_model.load_weights(model_save_path)