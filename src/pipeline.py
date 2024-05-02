import numpy as np
import pandas as pd

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


def pipe(df,target):

    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73, stratify=y)

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

    modelos = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight="balanced"),
        'XGBoost': XGBClassifier(verbosity=0, random_state=42, scale_pos_weight=70/30),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-100, class_weight='balanced'),
        'LogisticRegression': LogisticRegression(max_iter=10000, class_weight='balanced'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False, auto_class_weights='Balanced'),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        'SVC': SVC(random_state=42, class_weight='balanced'),
        'KNeighbors': KNeighborsClassifier(n_neighbors=4)
    }

    resultados = {}

    for nombre, modelo in modelos.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', modelo)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='recall')
        media_scores = scores.mean()
        resultados[nombre] = [media_scores]

    df_resultados = pd.DataFrame(resultados, index=['Media Recall']).T.sort_values(by='Media Recall', ascending=False)

    param_grid_rf = {
        "classifier__n_estimators": [50,100,200],
        "classifier__max_depth": [None,20,30],
        "classifier__min_samples_split": [2, 10, 20],
        "classifier__min_samples_leaf": [1,5,10],
        "classifier__max_features": ["sqrt","log2",None],
        "classifier__class_weight": ["balanced", None],
    }
    param_grid_xgb= {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0]
    }
    param_grid_lgb= {
        "classifier__max_depth": [-1,5,10],
        "classifier__num_leaves": [31, 50],
        "classifier__learning_rate": [0.1, 0.01],
        "classifier__n_estimators": [100, 200],
        "classifier__class_weight": ["balanced", None], 
        "classifier__min_child_samples": [20, 30],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }
    param_grid_tree = {
        "classifier__criterion": ["gini","entropy"],
        "classifier__splitter": ["best", "random"],
        "classifier__max_depth": [None,20,30,40],
        "classifier__min_samples_split":[2,10,20],
        "classifier__min_samples_leaf":[1,5,10],
        "classifier__max_features": ["sqrt","log2",None],
        "classifier__class_weight": ["balanced", None]
    }
    param_grid_lg= {
        "classifier__C":[0.01, 0.1, 1, 10],
        "classifier__max_iter":[1000,2000,5000],
        "classifier__class_weight":["balanced",None]
    }                
    param_grid_knn= {
        "classifier__n_neighbors":[3,4,5],
        "classifier__weights":['uniform', 'distance'],
        "classifier__metric":["manhattan","euclidean","chebyshev"]
    }
    param_grid_cat= {
        "classifier__iterations": [100, 300], 
        "classifier__learning_rate": [0.01, 0.05, 0.1],  
        "classifier__depth": [4, 6, 8],  
        "classifier__l2_leaf_reg": [1, 3, 5],  
        "classifier__bagging_temperature": [0, 1, 10],
        "classifier__auto_class_weights": ["Balanced"]
    }
    param_grid_svc= {
        "classifier__C":[0.01, 0.1, 1, 10, 100],
        "classifier__kernel":['linear', 'poly', 'rbf', 'sigmoid'],
        "classifier__gamma":['scale', 'auto'],
        "classifier__class_weight":["balanced",None]
    }

    modelos = {
        'Random_Forest': (RandomForestClassifier(random_state=42), param_grid_rf),
        'XGBoost': (XGBClassifier(verbosity=0, random_state=42, scale_pos_weight=70/30), param_grid_xgb),
        'LightGBM':(LGBMClassifier(random_state= 42, verbose = -100), param_grid_lgb),
        'DecisionTree':(DecisionTreeClassifier(random_state= 42), param_grid_tree),
        'LogisticRegression':(LogisticRegression(random_state=42), param_grid_lg),
        'KNeighbors':(KNeighborsClassifier(), param_grid_knn),
        'CatBoost':(CatBoostClassifier(random_state= 42, verbose= False), param_grid_cat),
        'SVC':(SVC(random_state= 42), param_grid_svc),
    }

    resultados = {}
    modelos_gs = {} 

    for nombre, (modelo, param_grid) in modelos.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', modelo)
        ])
        
        grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring="recall")
        grid_search.fit(X_train, y_train)
        resultados[nombre] = grid_search.best_score_
        modelos_gs[nombre] = grid_search 

    df_resultadosGS = pd.DataFrame.from_dict(resultados, orient='index', columns=['Media Recall'])
    df_resultadosGS.sort_values(by='Media Recall', ascending=False, inplace=True)

    mejor_modelo_nombre = df_resultadosGS.idxmax()[0]

    mejor_modelo = modelos_gs[mejor_modelo_nombre].best_estimator_

    dump(mejor_modelo, f'mejor_modelo_{mejor_modelo_nombre}.joblib')

if __name__ == "__main__":
    print('Cargando datos...')
    df = pd.read_csv("src/data/MetabolicSyndrome.csv")
    target = 'MetabolicSyndrome'
    pipe(df,target)