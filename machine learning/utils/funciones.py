#Básicos

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


#Visualización
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib




def borrar(df,columnas_a_eliminar):
    '''elimina las columnas de un dataframe,
    meter nombre de las columnas en una lista'''

    df.drop(columnas_a_eliminar,axis=1,inplace=True)
    return df



def cambio_de_nombre(df,columnas_a_renonbrar):
    '''renombras las columnas,
      meter los nombres en una lista'''
    
                    
    df.set_axis(columnas_a_renonbrar, axis=1,inplace=True)
    return df


def provincias(df,columna):

    ''' cambia el nombre de las provincias en comunidades autonomas, meter nombre de columna deseada'''

    for i in df[columna]:
        if i == 'Almería'or i == 'Granada'or i =='Córdoba' or i== 'Jaén' or i =='Sevilla' or i== 'Málaga' or i =='Cádiz' or i== 'Huelva':
            df[columna].replace(i,'Andalucía',inplace=True)
        elif i== 'Huesca' or i =='Zaragoza' or i=='Teruel':
            df[columna].replace(i,'Aragón',inplace=True)
        elif i=='Toledo' or i =='Guadalajara' or i=='Albacete' or i == 'Cuenca' or i =='Ciudad Real':
            df[columna].replace(i,'Castilla la Mancha',inplace=True)
        elif i=='León' or i =='Palencia' or i=='Salamanca' or i == 'Burgos' or i == 'Zamora' or i =='Valladolid' or i =='Soria' or i =='Segovia' or i == 'Ávila':
            df[columna].replace(i,'Castilla y León',inplace=True)
        elif i=='Barcelona' or i == 'Tarragona' or i =='Lérida' or i=='Gerona' or i=='Lleida' or i=='Girona':
            df[columna].replace(i,'Cataluña',inplace=True)
        elif i =='Caceres' or i =='Badajoz' or i =='Cáceres':
            df[columna].replace(i,'Extremadura',inplace=True)
        elif i== 'Álava' or i =='Guipúzcoa' or i =='Vizcaya':
            df[columna].replace(i,'País Vasco',inplace=True)
        elif i=='Castellón' or i =='Valencia' or i =='Alicante':
            df[columna].replace(i,'columna',inplace=True)
        elif i =='La coruña' or i== 'Lugo' or i=='Ourense' or i=='Pontevedra'or i == 'A Coruña' or i=='Orense':
            df[columna].replace(i,'Galicia',inplace=True)
        elif i == 'Tenerife' or i == 'Las Palmas':
            df[columna].replace(i,'Islas Canarias',inplace=True)
        elif i == 'Baleares':
            df[columna].replace(i,'Islas_Baleares',inplace=True)
        elif i == 'Madrid':
            df[columna].replace(i,'Madrid',inplace=True)
        elif i == 'Madrid':
            df[columna].replace(i,'Navarra',inplace=True)
        elif i == 'Madrid':
            df[columna].replace(i,'Asturias',inplace=True)
        elif i == 'La Rioja':
            df[columna].replace(i,'La rioja',inplace=True)                                              
        elif i == 'Murcia':
            df[columna].replace(i,'Murcia',inplace=True) 
        elif i == 'Cantabria':
            df[columna].replace(i,'Cantabria',inplace=True)

    return df


def borrar_filas(df,columna,palabras):
    '''borra filas que contengan las palabras(meter en lista)
    de la columna que le digas'''
    df_copy = df.copy() 


    for index,row in df_copy.iterrows():
        if row[columna] in palabras :
            df_copy=df_copy.drop(index)
    return df_copy

def pasar_a_string(df,columna):
    '''funcion que convierte columnas boleanas en string'''
    df[columna]=df[columna].astype(str)
    return df

def divide_fecha(df,columna):
    ''' divine una columna en 3 separadas por -
    esta pensado para fechas, tiene que ser en formato str
    ejemplo 2020-11-07 17:25:28'''
    df[['año_Venta', 'Mes_Venta','dia']] = df[columna].str.split('-',3,expand=True)
    return df


def borrar_null(df,columna):
    '''funcion que convierte columnas boleanas en string'''
    df = df.dropna(subset=columna)
    return df



def fillna_mode_by_group(df, column_to_fillna, column_to_group):
    """
    rellena los valores faltantes con la moda de su tipo en una columna distinta
    """
    groups = df.groupby(column_to_group)
    for group_name, group_data in groups:
        non_missing_values = group_data[column_to_fillna].dropna().unique()
        if len(non_missing_values) > 1:
            mode_value = group_data[column_to_fillna].mode().values[0]
            df.loc[group_data.index, column_to_fillna] = group_data[column_to_fillna].fillna(mode_value)
    return df


def primera_palabra(cadena):

    '''se queda con la primera palabra de una frase'''
    if isinstance(cadena, str):
        partes = cadena.split(" ")
        return partes[0]
    else:
        return None
    

def pasar_a_float(df,columna):
    '''funcion que convierte columnas float'''
    df[columna]=df[columna].astype(float)
    return df

def pasar_a_int(df,columna):
    '''funcion que convierte columnas int'''
    df[columna]=df[columna].astype(int)
    return df




def anañadir_columna(df,nombre_colum,loque_añado):
    df[nombre_colum]=loque_añado
    return df



def unir_dataframes(df1, df2, how,left_on, right_on):
    '''funcion que hace un marge con 2 columnas
    pudiendo elegir las columnas de union
    y el como'''
    analisis = pd.merge(df1, df2, how=how,left_on=left_on, right_on=right_on)
    return analisis


def unir_columnas(df,columna_nueva ,columna_vieja1,columna_vieja2,sep):

    ''' esta funcion uni columnas string 
    con este separador que le indiques'''
    df[columna_nueva]=df[columna_vieja1].str.cat(df[columna_vieja2],sep)
    return df


def pasar_a_str(df,columna):
    '''funcion que convierte columnas int'''
    df[columna]=df[columna].astype(str)
    return df


def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])
    return df


def hacer_cluster(df):
    from sklearn.cluster import KMeans
    df['marca_num'] = pd.factorize(df['Marca'])[0]
    precios = df[['precio', 'Caballos', 'marca_num']]
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(precios)
    df['cluster'] = kmeans.labels_
    return df


def quitar_outliers(df,cols):   
# Calculate quantiles and IQR
    Q1 = df[cols].quantile(0.25) # Same as np.percentile but maps (0,1) and not (0,100)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

# Return a boolean array of the rows with (any) non-outlier column values
    condition = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    return df


def dummies(df,columnas,pre):
    df = pd.get_dummies(df, columns=columnas, prefix=pre)
    return df


def factorizar(df, columna):
    df[columna] = pd.factorize(df[columna])[0]
    return df


def escalar(df):

    scaler=StandardScaler()

    scaler.fit_transform(df)
    df_escalado = scaler.transform(df)
    df_escalado= pd.DataFrame(df_escalado, columns=df.columns)
    return df_escalado


def eliminar_outliers(df, columnas):

    df_out = df.copy()
    for col in columnas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        filtro = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR).any(axis=1)
        df_out = df_out[filtro]
    return df_out


def train_test_split_df(df, target_column, test_size=0.2, random_state=42):

  
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
   
    return X_train, X_test, y_train, y_test



def lgb_model_trabajo(X_train, y_train, X_test, y_test):
    
    model_lgb = lgb.LGBMRegressor(learning_rate= 0.07, max_depth= -1, n_estimators=450, num_leaves= 10)
    
 
    model_lgb.fit(X_train,y_train)
    

    lgb_pred = model_lgb.predict(X_test)
    
 
    mae = metrics.mean_absolute_error(y_test, lgb_pred)
    mse = metrics.mean_squared_error(y_test, lgb_pred)
    rmse = np.sqrt(mse)
    
  
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    
    
    return model_lgb, lgb_pred


import pickle

def guardar_modelo(modelo, nombre_archivo):
    with open(nombre_archivo, 'wb') as archivo:
        pickle.dump(modelo, archivo)
    print(f"Modelo guardado correctamente en el archivo {nombre_archivo}")
