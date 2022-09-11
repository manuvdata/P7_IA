from django.urls import path
import Prediction.views as views

urlpatterns = [

    path('tweet/', views.Tweets_Model_Predict.as_view(), name = 'tweet'),
    path('tweet_class/', views.api_tweet, name = 'api_tweet'),

]
