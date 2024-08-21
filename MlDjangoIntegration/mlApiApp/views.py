from django.shortcuts import render
from django.http import HttpRequest, HttpResponse
import pickle
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np
import pandas as pd

model = pickle.load(open(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML model integration with Django\MlDjangoIntegration\ml_Model\regressionmodel.pkl', 'rb'))
scaler = pickle.load(open(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\ML model integration with Django\MlDjangoIntegration\ml_Model\scaling.pkl', 'rb'))

@api_view(['POST'])
def predict_Value(request:HttpRequest):
    """
    Expected JSON data (raw JSON data/form data):
    {
        "input": list of input values
    }
    """
    print(request.data)
    input_Values = request.data.get('input')
    print(input_Values)
    if input_Values is None:
        return Response({
            'message': 'Input is None.'
        })
    print(np.array(list(input_Values)).reshape(1, -1))
    scaled_Input = scaler.transform(np.array(list(input_Values)).reshape(1, -1))
    print(scaled_Input)
    output = model.predict(scaled_Input)
    print(output)
    return Response({
        'input': input_Values,
        'output': output
    })

def home(request:HttpRequest):
    return HttpResponse('<h1>HOME<h1>')
