from flask import Flask, jsonify, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load

import os
os.chdir(os.path.dirname(__file__))

from funciones import *
from pipeline import pipe

app = Flask(__name__)
app.config['DEBUG'] = True

# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a la API de predicción del síndrome metabólico"

@app.route("/predict", methods=["GET"])
def show_form():
    return render_template("formulario.html")

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    model = load('mejor_modelo_XGBoost.joblib')
    
    # Listas de valores permitidos
    valid_sex = ['Male', 'Female']
    valid_marital = ['Married', 'Single', 'Divorced', 'Widowed']
    valid_race = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
    valid_albuminuria = [0,1,2]

    # Obtener y validar argumentos
    try:
        Age = int(request.form['Age'])
        Sex = request.form['Sex']
        Marital = request.form['Marital']
        Income = float(request.form['Income'])
        Race = request.form['Race']
        WaistCirc = float(request.form['WaistCirc'])
        BMI = float(request.form['BMI'])
        Albuminuria = int(request.form['Albuminuria'])
        UricAcid = float(request.form['UricAcid'])
        BloodGlucose = float(request.form['BloodGlucose'])
        HDL = float(request.form['HDL'])
        Triglycerides = float(request.form['Triglycerides'])

    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Tipos de entrada no válidos'}), 400

    # Verificar si los valores categóricos son válidos
    errors = {}
    if Sex not in valid_sex:
        errors['Sex'] = f"Entrada inválida. Valores permitidos: {valid_sex}"
    if Marital not in valid_marital:
        errors['Marital'] = f"Entrada inválida. Valores permitidos: {valid_marital}"
    if Race not in valid_race:
        errors['Race'] = f"Entrada inválida. Valores permitidos: {valid_race}"
    if Albuminuria not in valid_albuminuria:
        errors['Albuminuria'] = f"Entrada invalida. Valores permitidos: {valid_albuminuria}"

    # Si hay errores, retorna el mensaje con los errores
    if errors:
        return jsonify({'error': 'Entradas inválidas', 'details': errors}), 400

    # Crear dataframe y predecir
    data_dict = {
        'Age': [Age],
        'Sex': [Sex],
        'Marital': [Marital],
        'Income': [Income],
        'Race': [Race],
        'WaistCirc': [WaistCirc],
        'BMI': [BMI],
        'Albuminuria': [Albuminuria],
        'UricAcid': [UricAcid],
        'BloodGlucose': [BloodGlucose],
        'HDL': [HDL],
        'Triglycerides': [Triglycerides]
    }

    df = pd.DataFrame(data_dict)
    prediction = model.predict(df)[0]
    predict = 'El paciente presenta MetabolicSyndrome' if prediction == 1 else 'El paciente NO presenta MetabolicSyndrome'

    return jsonify({'prediction': predict})


@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if not os.path.exists("data/MetabolicSyndrome.csv"):
        return "Archivo de datos no encontrado. Por favor, cargue los datos antes de reintentar...",404
    
    df = pd.read_csv("data/MetabolicSyndrome.csv")
    target = 'MetabolicSyndrome'
    print('Entrenando')
    pipe(df,target)

    return "Modelo reentrenado y guardado exitosamente.", 200

if __name__ == "__main__":
    
    app.run(debug=True)





