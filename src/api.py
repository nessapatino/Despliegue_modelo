from flask import Flask, jsonify, request
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
sys.path.append("..")

from utils.funciones_toolbox_ml_final import *
from utils.modulos import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer,OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, recall_score, balanced_accuracy_score, make_scorer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

import os


# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a la API de predicción del síndrome metabólico"



def categorize_BMI(bmi_list):
    df = pd.DataFrame(bmi_list, columns=['BMI'])
    bins = [-np.inf, 18.5, 24.9, 29.9, np.inf]
    labels = ['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad']
    df['BMI'] = pd.cut(df['BMI'], bins=bins, labels=labels)  # Reemplaza la columna 'BMI'
    return df[['BMI']]  # Retorna solo la columna transformada

def categorize_BloodGlucose(blood_glucose_list):
    df = pd.DataFrame(blood_glucose_list, columns=['BloodGlucose'])
    bins = [-np.inf, 99, 126, np.inf]
    labels = ['Normal', 'Prediabetes', 'Diabetes']
    df['BloodGlucose'] = pd.cut(df['BloodGlucose'], bins=bins, labels=labels)  # Reemplaza 'BloodGlucose'
    return df[['BloodGlucose']]  # Retorna solo la columna transformada

def categorize_Triglycerides(data):
    df = pd.DataFrame(data, columns=['Age', 'Triglycerides'])
    def categorize_row(row):
        age, triglycerides = row
        if 10 <= age <= 19:
            if triglycerides < 90:
                return 'Nivel normal'
            elif triglycerides < 150:
                return 'Niveles ligeramente altos'
            else:
                return 'Niveles altos'
        else:
            if triglycerides < 150:
                return 'Nivel normal'
            elif triglycerides < 200:
                return 'Niveles ligeramente altos'
            elif triglycerides < 500:
                return 'Niveles altos'
            else:
                return 'Niveles muy altos'

    df['Triglycerides'] = df[['Age', 'Triglycerides']].apply(categorize_row, axis=1)
    return df[['Triglycerides']]

# Función para categorizar el HDL
def categorize_HDL(data):
    df = pd.DataFrame(data, columns=['Sex', 'HDL'])
    def categorize_row(row):
        sex, hdl = row
        return 'Valor_Bajo' if (sex == 'Male' and hdl < 40) or (sex == 'Female' and hdl < 50) else 'Normal'

    df['HDL'] = df[['Sex', 'HDL']].apply(categorize_row, axis=1)
    return df[['HDL']]

# Función para categorizar la circunferencia de cintura (WaistCirc)
def categorize_WaistCirc(data):
    df = pd.DataFrame(data, columns=['Sex', 'WaistCirc'])
    def categorize_row(row):
        sex, waistcirc = row
        return 'Riesgo Elevado' if (sex == 'Female' and waistcirc > 88) or (sex == 'Male' and waistcirc > 102) else 'Normal'
    
    df['WaistCirc'] = df[['Sex', 'WaistCirc']].apply(categorize_row, axis=1)
    return df[['WaistCirc']]

# Definición de los transformadores para cada columna
bmi_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('func_transform', FunctionTransformer(categorize_BMI, validate=False)),
    ('one_hot', OneHotEncoder())
])

blood_glucose_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('func_transform', FunctionTransformer(categorize_BloodGlucose)),
    ('one_hot', OneHotEncoder())
])

triglycerides_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('func_transform', FunctionTransformer(categorize_Triglycerides)),
    ('one_hot', OneHotEncoder())
])

hdl_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('func_transform', FunctionTransformer(categorize_HDL)),
    ('one_hot', OneHotEncoder())
])

waist_circ_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('func_transform', FunctionTransformer(categorize_WaistCirc)),
    ('one_hot', OneHotEncoder())
])

income_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('power_transform', PowerTransformer())
])

age_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('power_transform', PowerTransformer())
])

sex_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder())
])

marital_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder())
])

race_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder())
])

albuminuria_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('power_transform', PowerTransformer())
])

uricacid_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('power_transform', PowerTransformer())
])

# ColumnTransformer para aplicar las transformaciones a las columnas específicas
preprocessor = ColumnTransformer(
    transformers=[
        ('bmi', bmi_transformer, ['BMI']),
        ('blood_glucose', blood_glucose_transformer, ['BloodGlucose']),
        ('triglycerides', triglycerides_transformer, ['Age','Triglycerides']),
        ('hdl', hdl_transformer, ['Sex','HDL']),
        ('waist_circ', waist_circ_transformer, ['Sex','WaistCirc']),
        ('income', income_transformer, ['Income']),
        ('age', age_transformer, ['Age']),
        ('sex', sex_transformer, ['Sex']),
        ('marital', marital_transformer, ['Marital']),
        ('race', race_transformer, ['Race']),
        ('albuminuria', albuminuria_transformer, ['Albuminuria']),
        ('uricacid', uricacid_transformer, ['UricAcid']),
    ], remainder='drop'
)

# Enruta la funcion al endpoint /api/v1/predict
@app.route("/api/v1/predict", methods=["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    model = load('mejor_modelo_XGBoost.joblib')
    seqn = float(request.args.get('seqn', None))
    Age = float(request.args.get('Age', None))
    Sex = request.args.get('Sex', None)
    Marital = request.args.get('Marital', None)
    Income = float(request.args.get('Income', None))
    Race = request.args.get('Race', None)
    WaistCirc = float(request.args.get('WaistCirc', None))
    BMI = float(request.args.get('BMI', None))
    Albuminuria = float(request.args.get('Albuminuria', None))
    UrAlbCr = float(request.args.get('UrAlbCr', None))
    UricAcid = float(request.args.get('UricAcid', None))
    BloodGlucose = float(request.args.get('BloodGlucose', None))
    HDL = float(request.args.get('HDL', None))
    Triglycerides = float(request.args.get('Triglycerides', None))
    
    
    if any(v is None for v in [seqn, Age, Sex, Marital, Income, Race, WaistCirc, BMI, Albuminuria, UrAlbCr, UricAcid, BloodGlucose, HDL, Triglycerides]):
        return "Args empty, the data are not enough to predict"
    else:
        data_dict = {
        'seqn': [seqn],
        'Age': [Age],
        'Sex': [Sex],
        'Marital': [Marital],
        'Income': [Income],
        'Race': [Race],
        'WaistCirc': [WaistCirc],
        'BMI': [BMI],
        'Albuminuria': [Albuminuria],
        'UrAlbCr': [UrAlbCr],
        'UricAcid': [UricAcid],
        'BloodGlucose': [BloodGlucose],
        'HDL': [HDL],
        'Triglycerides': [Triglycerides]
    }
        data = pd.DataFrame(data_dict)
        data_transformed = preprocessor.fit_transform(data)
        prediction = model.predict(data_transformed)
    
    return jsonify({'predictions': prediction[0]})
        
app.run()
