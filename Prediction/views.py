from django.shortcuts import render

# Create your views here.
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from .apps import PredictionConfig
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from keras.models import Model
import tensorflow as tf
from keras.utils import np_utils
import numpy as np

# Create your views here.









def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clean_stopwords_shortwords(w):
    stopwords_list=stopwords.words('english')
    words = w.split() 
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 1]
    return " ".join(clean_words) 

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w=clean_stopwords_shortwords(w)
    w=re.sub(r'@\w+', '',w)
    return w

def encode_names(n, tokenizer):
   tokens = list(tokenizer.tokenize(n))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

def bert_encode(string_list, tokenizer, max_seq_length):
  num_examples = len(string_list)
  
  string_tokens = tf.ragged.constant([
      encode_names(n, tokenizer) for n in np.array(string_list)])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*string_tokens.shape[0]
  input_word_ids = tf.concat([cls, string_tokens], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor(shape=(None, max_seq_length))

  type_cls = tf.zeros_like(cls)
  type_tokens = tf.ones_like(string_tokens)
  input_type_ids = tf.concat(
      [type_cls, type_tokens], axis=-1).to_tensor(shape=(None, max_seq_length))

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(shape=(None, max_seq_length)),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs


def bad_tweets(tweets):
    sentences = [tweets]
    input_ids=[]
    attention_masks=[]


    for sent in sentences:
        sent = preprocess_sentence(sent)
        print(sent)
        bert_inp=PredictionConfig.bert_tokenizer.encode_plus(sent,add_special_tokens = True,max_length =102,pad_to_max_length = True,return_attention_mask = True,padding='max_length')
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids=np.asarray(input_ids)
    attention_masks=np.array(attention_masks)
    preds = PredictionConfig.trained_model.predict([input_ids,attention_masks],batch_size=32)
    pred_labels = int(np.argmax(preds.logits, axis=1))
    return pred_labels
    if pred_labels == 0 :
         print('bad tweets')
    if pred_labels == 1 :
         print('good tweets')



class Tweets_Model_Predict(APIView):
    #permission_classes = [IsAuthenticated]
    def post(self, request, format=None):
        data = request.data
        keys = []
        values = []
        for key in data:
            resultat = bad_tweets(data[key])
            if resultat == 1 :
                sum = 'Good Tweet'
            if resultat == 0 :
                sum = ' ATTENTION BAD TWEET'
        response_dict = {"Prediction du tweets": sum}


        return Response(response_dict, status=200)

@api_view(['GET', 'POST'])
def api_tweet(request):
    sum = 0
    response_dict = {}
    if request.method == 'GET':
        # Do nothing
        pass
    elif request.method == 'POST':
        # Add the numbers
        data = request.data
        for key in data:
            resultat = bad_tweets(data[key])
            if resultat == 1 :
                sum = 'Good Tweet'
            if resultat == 0 :
                sum = ' ATTENTION BAD TWEET'
            


        response_dict = {"Prediction du tweets": sum}
    return Response(response_dict, status=status.HTTP_201_CREATED)

