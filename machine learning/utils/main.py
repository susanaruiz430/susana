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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost
import pickle


#carga de datos


ruta_coches_usados=r"C:\Users\Susana Ruiz\Documents\curso\machine learning\data\raw_files\coches_segunda_mano.csv"
coches_usados = pd.read_csv(ruta_coches_usados)
ruta_gasolina=r"C:\Users\Susana Ruiz\Documents\curso\machine learning\data\processed_files\tabla_gasolina.csv"
tabla_gasolina = pd.read_csv(ruta_gasolina)


#procesado de datos


columnas=['url','company','price_financed','photos',
                    'dealer','country','shift','insert_date','version']
coches_usados = funciones.borrar(coches_usados,columnas)

columnas=['Marca', 'Modelo','precio','Combustible',
                        'Año_del_vehiculo','kilometros','Caballos','puertas','color_coche',
                        'profesional','Comunidad_autonoma','Año_de_venta']
coches_usados = funciones.cambio_de_nombre(coches_usados,columnas)

print('hola')
coches_usados = funciones.provincias(coches_usados,'Comunidad_autonoma')

columna='Comunidad_autonoma'
palabras=['Ceuta','Melilla']
coches_usados = funciones.borrar_filas(coches_usados,columna,palabras)


columna='profesional'
coches_usados = funciones.pasar_a_string(coches_usados,columna)


columna='profesional'
palabras='True'
coches_usados = funciones.borrar_filas(coches_usados,columna,palabras)



columna='Año_de_venta'
coches_usados = funciones.divide_fecha(coches_usados,columna)



columnas=['Año_de_venta','dia']
coches_usados = funciones.borrar(coches_usados,columnas)



columna=['Comunidad_autonoma','Caballos','Marca']
coches_usados = funciones.borrar_null(coches_usados,columna)


coches_usados = funciones.fillna_mode_by_group(coches_usados, 'color_coche', 'Modelo')


palabras=['IVECO-PEGASO','UMM','IVECO']
columna='Marca'
coches_usados = funciones.borrar_filas(coches_usados,columna,palabras)


coches_usados['color_coche']=coches_usados['color_coche'].apply(funciones.primera_palabra)


columna=['Diesel','Super_95']
tabla_gasolina = funciones.pasar_a_string(tabla_gasolina,columna)


columna=['Diesel','Super_95']
tabla_gasolina = funciones.pasar_a_float(tabla_gasolina,columna)

coches_usados = funciones.borrar(coches_usados,'profesional')


columna=['Mes_Venta','año_Venta']
coches_usados = funciones.pasar_a_int(coches_usados,columna)


columna1=['Mes','Año']
tabla_gasolina = funciones.pasar_a_int(tabla_gasolina,columna1)



analisis = funciones.unir_dataframes(coches_usados, tabla_gasolina,'inner',
                           ['año_Venta','Mes_Venta'],['Año','Mes'])


analisis=funciones.unir_columnas(analisis,'marca_modelo','Marca','Modelo',' ')

analisis = funciones.pasar_a_str(analisis,columna)


analisis=funciones.unir_columnas(analisis,'Fecha','Mes_Venta','año_Venta','/')


analisis = funciones.convert_to_datetime(analisis, 'Fecha')

analisis = funciones.hacer_cluster(analisis)

cols = ['precio','Combustible', 'Año_del_vehiculo', 'kilometros','Caballos', 'puertas',
'Mes_Venta','cluster']
analisis = funciones.quitar_outliers(analisis,cols)

columnas=['Combustible']
prefijo='combus'
analisis=funciones.dummies(analisis,columnas,prefijo)


analisis = funciones.factorizar(analisis, 'Comunidad_autonoma')

analisis = funciones.factorizar(analisis, 'color_coche')

analisis = funciones.borrar(analisis,'combus_Híbrido enchufable')


columna=['año_Venta','Mes_Venta']
analisis = funciones.pasar_a_float(analisis,columna)



columnas=['puertas','Comunidad_autonoma','color_coche','Super_95','Unidad','Modelo',
                            'Fecha','marca_modelo','Mes','Año']

analisis = funciones.borrar(analisis,columnas)

analisis_escalado = funciones.escalar(analisis)

#entrenamiento del modelo

X= analisis_escalado.drop(['precio'], axis=1)
y= analisis_escalado['precio']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                     y, test_size=0.2,random_state=42)


model_lgb = lgb.LGBMRegressor(learning_rate= 0.07, max_depth= -1, n_estimators=450, num_leaves= 10)
model_lgb.fit(X_train,y_train)
lgb_pred = model_lgb.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))
print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))

#guarda el modelo

with open('model_lgb.pkl', 'wb') as f:
    pickle.dump(model_lgb, f)