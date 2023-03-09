#librerias

#Básicos

import numpy as np
import pandas as pd
import funciones
import warnings
warnings.filterwarnings('ignore')


#Visualización
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


#Modelos

import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import StandardScaler
import xgboost
import pickle
import joblib
import lightgbm as lgb


#carga de datos


ruta_analisis=r"C:\Users\Susana Ruiz\Documents\curso\machine learning\data\processed_files\analisis.csv"
analisis = pd.read_csv(ruta_analisis)




#procesado de datos




analisis=funciones.unir_columnas(analisis,'marca_modelo','Marca','Modelo',' ')

columna=['Mes_Venta','año_Venta']
analisis = funciones.pasar_a_str(analisis,columna)


analisis=funciones.unir_columnas(analisis,'Fecha','Mes_Venta','año_Venta','/')


analisis = funciones.convert_to_datetime(analisis, 'Fecha')

analisis = funciones.hacer_cluster(analisis)
analisis = funciones.borrar(analisis,'Marca')


columna=['color_coche']
analisis = funciones.borrar_null(analisis,columna)




cols = ['precio','Combustible', 'Año_del_vehiculo', 'kilometros','Caballos', 'puertas',

'Mes_Venta','cluster']
analisis = funciones.quitar_outliers(analisis,cols)



columnas=['Combustible']
prefijo='combus'
analisis=funciones.dummies(analisis,columnas,prefijo)


analisis = funciones.factorizar(analisis, 'Comunidad_autonoma')

analisis = funciones.factorizar(analisis, 'color_coche')

columna=['año_Venta','Mes_Venta']
analisis = funciones.pasar_a_float(analisis,columna)





columnas=['puertas','Comunidad_autonoma','color_coche','Super_95','Unidad','Modelo',
                            'Fecha','marca_modelo','Mes','Año']


analisis = funciones.borrar(analisis,columnas)



X_train, X_test, y_train, y_test = funciones.train_test_split_df(analisis, 'precio')




#entrenamiento del modelo


model_lgb=funciones.lgb_model_trabajo(X_train, y_train, X_test, y_test)




#guarda el modelo
funciones.guardar_modelo(model_lgb, "modelo_lgb.pkl")


