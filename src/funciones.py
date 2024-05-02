import numpy as np
import pandas as pd

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
