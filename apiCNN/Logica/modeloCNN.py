from numpy.random import seed
seed(1)
import cv2
from keras.models import model_from_json
from django.db import models
from django.urls import reverse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
from apiCNN import models
import os
from PIL import Image
from tensorflow.python.keras.models import Sequential
import pathlib
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
#import matplotlib.image as mpimg
import tarfile
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input, InputLayer
#import matplotlib.pyplot as plt

class modeloCNN():
    """Clase modelo SNN"""

    def cargarRNN(nombreArchivoModelo,nombreArchivoPesos):        
        # Cargar la Arquitectura desde el archivo JSON
        with open(nombreArchivoModelo+'.json', 'r') as f:model = model_from_json(f.read())

        # Cargar Pesos (weights) en el nuevo modelo
        model.load_weights(nombreArchivoPesos+'.h5')  

        print("Red Neuronal Cargada desde Archivo") 
        return model

    def predecir(self, image_id):

        print('CARGANDO MODELO...')
        label_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
        nombreArchivoModelo=r'apiCNN/Logica/arquitectura_optimizada'
        nombreArchivoPesos=r'apiCNN/Logica/pesos_optimizados'

        dbReg=models.Image.objects.get(id=image_id)
        image_path = str(dbReg.image)

        model=self.cargarRNN(nombreArchivoModelo,nombreArchivoPesos) 
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

        print('Imagen:')
        img = Image.open(image_path).convert('RGB')
        img= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
        arr = np.array(img)
        
        img=cv2.resize(img, (32, 32),interpolation = cv2.INTER_AREA)
        print(np.array(img).shape)
        arrTrans = np.array(img).reshape(1, 32, 32, 3)

        print('Predicciones:')
        resultados = model.predict(arrTrans)[0]
        print(resultados)
        maxElement = np.amax(resultados)
        print('certeza: ', str(round(maxElement*100, 4))+'%')
        result = np.where(resultados == np.amax(resultados))
        print('Max :', maxElement)
        index_sample_label=result[0][0]

        print(label_names)
        print('Etiqueta predicción: ', label_names[index_sample_label])
        
        dbReg.label = label_names[index_sample_label]
        dbReg.probability = maxElement
        dbReg.save()
        datos = dict()    
        datos['pred'] = label_names[index_sample_label]
        datos['prob'] = str(round(maxElement*100, 4))+'%'
        datos['url']= image_path    
        return datos

        #plt.imshow(img)
            
