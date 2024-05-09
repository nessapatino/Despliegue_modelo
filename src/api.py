from flask import Flask, request, render_template, render_template_string, send_file, jsonify
import pandas as pd
import pickle
import subprocess
from funciones import categorize_BMI, categorize_BloodGlucose, categorize_Triglycerides, categorize_HDL, categorize_WaistCirc
import os
os.chdir(os.path.dirname(__file__))

from pipeline import pipe


app = Flask(__name__)
app.config['DEBUG'] = True

root_path= "/home/SindromeMetabolico/Despliegue_modelo/src/"
# root_path = ""
# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello_with_image_and_button():
    # Devolver HTML que incluye el mensaje de bienvenida y un botón
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bienvenido a la API de predicción del síndrome metabólico...</title>
        <style>
            body {
                background-image: url('/imagen');
                background-repeat: no-repeat;
                background-attachment: fixed;  
                background-size: cover;
                background-color: transparent;
                color: white;
                font-family: sans-serif;
                text-align: center;
            }
            .content {
                padding: 50px;
            }
            h1 {
                font-size: 50px;
            }
            button {
                font-size: 25px;
                padding: 20px 30px;
                border: none;
                background-color: #007bff;
                color: black;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
        </style>
    </head>
    <body>
        <div class="content">
            <h1>PREDICCIÓN DE SÍNDROME METÁBOLICO</h1>
            <a href="/predict"><button>Ir al formulario de predicción</button></a>
        </div>
    </body>
    </html>
    """

@app.route("/imagen", methods=["GET"])
def obtener_imagen():
    # Suponiendo que la imagen está en la misma carpeta que tu archivo de Flask
    return send_file('sm3.webp', mimetype='image/webp')

@app.route("/predict", methods=["GET"])
def show_form():
    return render_template("formulario.html")

@app.route("/api/v1/predict", methods=["POST"])
def predict():

    with open('mejor_modelo_SVC.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Listas de valores permitidos
    valid_sex = ['Male', 'Female']
    valid_marital = ['Married', 'Single', 'Divorced', 'Widowed','Unk','Separated']
    valid_race = ['White', 'Black', 'Asian', 'Hispanic', 'Other','MexAmerican']
    valid_albuminuria = [0, 1, 2]

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
        return render_template_string("""
            <html>
            <body>
                <h1>Error en la entrada</h1>
                <p>Tipos de entrada no válidos</p>
            </body>
            </html>
        """), 400

    errors = {}
    if Sex not in valid_sex:
        errors['Sex'] = f"Entrada inválida. Valores permitidos: {valid_sex}"
    if Marital not in valid_marital:
        errors['Marital'] = f"Entrada inválida. Valores permitidos: {valid_marital}"
    if Race not in valid_race:
        errors['Race'] = f"Entrada inválida. Valores permitidos: {valid_race}"
    if Albuminuria not in valid_albuminuria:
        errors['Albuminuria'] = f"Entrada invalida. Valores permitidos: {valid_albuminuria}"

    if errors:
        error_message = "<br>".join([f"{k}: {v}" for k, v in errors.items()])
        return render_template_string(f"""
            <html>
            <body>
                <h1>Error en la entrada</h1>
                <p>{error_message}</p>
            </body>
            </html>
        """), 400

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
    predict_text = 'El paciente presenta MetabolicSyndrome' if prediction == 1 else 'El paciente NO presenta MetabolicSyndrome'

    return render_template_string(f"""
    <html>
        <head>
            <style>
                body {{
                    background-color: #92a8d1;
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Resultado de la Predicción</h1>
            <p>{predict_text}</p>
        </body>
    </html>
    """)

@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if not os.path.exists(root_path + "data/MetabolicSyndrome.csv"):
        return "Archivo de datos no encontrado. Por favor, cargue los datos antes de reintentar...",404
    
    df = pd.read_csv(root_path + "data/MetabolicSyndrome.csv")
    target = 'MetabolicSyndrome'
    print('Entrenando')
    pipe(df,target,categorize_BMI, categorize_BloodGlucose, categorize_Triglycerides, categorize_HDL, categorize_WaistCirc)

    return "Modelo reentrenado y guardado exitosamente.", 200

@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = 'https://sindromemetabolico.pythonanywhere.com/'
    servidor_web = '/var/www/sindromemetabolico_pythonanywhere_com_wsgi.py' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400
    

if __name__ == "__main__":
    app.run()