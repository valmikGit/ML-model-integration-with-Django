from django.urls import path
from mlApiApp import views

urlpatterns = [
    path('', view=views.home),
    path('predict/', view=views.predict_Value)
]
