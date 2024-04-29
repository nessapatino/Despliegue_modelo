import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, f_oneway
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def tipifica_variables(df, umbral_categorica, umbral_continua):
    """
    Clasifica las variables de un DataFrame en función de sus características.

    Argumentos:
    - df: DataFrame que contiene las variables a clasificar.
    - umbral_categorica (int): Umbral para clasificar una variable como categórica.
    - umbral_continua (float): Umbral para clasificar una variable como continua.

    Retorna:
    DataFrame con dos columnas: 'nombre_variable' y 'tipo_sugerido'.
                      'nombre_variable': Nombre de la variable.
                      'tipo_sugerido': Tipo sugerido para la variable ('Categorica', 'Binaria', 'Numerica Discreta', 'Numerica Continua').
    """
    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(umbral_categorica, int) or not isinstance(umbral_continua, int):
        print("Error: df debe ser un DataFrame, 'umbral_categorica' un entero y 'umbral_continua' un entero.")
        return None
   
    df_tipificacion = pd.DataFrame([df.nunique(), df.nunique() / len(df) * 100, df.dtypes]).T.rename(columns={0: "Card", 1: "%_Card", 2: "Tipo"})
    df_tipificacion["Clasificada_como"] = "Categorica"
    df_tipificacion.loc[df_tipificacion.Card == 2, "Clasificada_como"] = "Binaria"
    df_tipificacion.loc[df_tipificacion["Card"] >= umbral_categorica, "Clasificada_como"] = "Numerica Discreta"
    df_tipificacion.loc[(df_tipificacion["%_Card"] >= umbral_continua) & (df_tipificacion["Card"] >= umbral_categorica), "Clasificada_como"] = "Numerica Continua"

    resultado = pd.DataFrame({
        "nombre_variable": df.columns,
        "tipo_sugerido": df_tipificacion["Clasificada_como"].values
    })
    return resultado

def describe_df(df):
    """
    Realiza un análisis descriptivo de un DataFrame.

    Argumentos:
    - df (DataFrame): El DataFrame a analizar.

    Retorna:
    DataFrame: Un DataFrame que contiene información sobre las características de cada columna del DataFrame de entrada.
        - Tipo: El tipo de datos de cada columna.
        - Porcentaje_Nulos: El porcentaje de valores nulos o missing en cada columna.
        - Valores_Unicos: El número de valores únicos en cada columna.
        - Porcentaje_Cardinalidad: El porcentaje de cardinalidad, es decir, la proporción de valores únicos respecto al total de valores, en cada columna.
    """
    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame")
        return None
    
    # Crea las variables
    tipos = df.dtypes
    nulos_porcentaje = df.isnull().mean() * 100
    unicos = df.nunique()
    cardinalidad_porcentaje = unicos / len(df) * 100

    # Crea el dataframe de salida
    df_describe = pd.DataFrame({
        "Tipo": tipos,
        "Porcentaje_Nulos": nulos_porcentaje,
        "Valores_Unicos": unicos,
        "Porcentaje_Cardinalidad": cardinalidad_porcentaje
    }).T

    return df_describe


def get_features_num_regression(df, target_col, umbral_corr, pvalue=None):
    """
    Esta función devuelve una lista con las columnas numéricas cuya correlación con 'target_col'
    es superior al valor absoluto de 'umbral_corr' y, si se proporciona 'pvalue', supera
    el test de hipótesis con significación mayor o igual a 1-pvalue.

    Argumentos:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que se considerará el target del modelo de regresión.
    - umbral_corr: Valor umbral para la correlación (debe estar entre 0 y 1).
    - pvalue: Valor p para el test de hipótesis (si se proporciona, debe ser un número entre 0 y 1).

    Retorna:
    - Lista de columnas numéricas que cumplen con los criterios especificados.
    """

    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(target_col, str):
        print("Error: df debe ser un DataFrame y 'target_col' debe ser una cadena.")
        return None

    if target_col not in df.columns:
        print(f"Error: {target_col} no está presente en el DataFrame.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None

    # Verifica que 'target_col' sea una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser una variable numérica continua.")
        return None

    # Comprueba valores nulos en el DataFrame
    if df.isnull().any().any():
        print("Error: El DataFrame contiene valores nulos. Imputa o elimina los valores nulos antes de continuar.")
        return None

    # Calcula la correlación y realiza las comprobaciones adicionales según los parámetros proporcionados
    correlated_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[col], df[target_col])

            if abs(correlation) > umbral_corr and (pvalue is None or p_value <= 1 - pvalue):
                correlated_columns.append(col)
    
    return correlated_columns

def plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0, pvalue=None):
    """
    Esta función pinta un pairplot del DataFrame considerando la columna designada por "target_col" y
    aquellas incluidas en "columns" que cumplen con ciertas condiciones de correlación.
    Además, devuelve una lista de las columnas que cumplen con estas condiciones.

    Argumentos:
    - df: DataFrame de pandas.
    - target_col: Nombre de la columna que se considerará el target del modelo de regresión.
    - columns: Lista de nombres de columnas numéricas a considerar (por defecto, la lista vacía).
    - umbral_corr: Valor umbral para la correlación (debe estar entre 0 y 1, por defecto 0).
    - pvalue: Valor p para el test de hipótesis (si se proporciona, debe ser un número entre 0 y 1).

    Retorna:
    - Lista de columnas numéricas que cumplen con las condiciones especificadas.
    """

    # Comprobaciones de los valores de entrada
    if not isinstance(df, pd.DataFrame) or not isinstance(target_col, str):
        print("Error: 'df' debe ser un DataFrame y 'target_col' debe ser una cadena.")
        return None

    if target_col not in df.columns:
        print(f"Error: {target_col} no está presente en el DataFrame.")
        return None

    if not (0 <= umbral_corr <= 1):
        print("Error: 'umbral_corr' debe estar entre 0 y 1.")
        return None

    if pvalue is not None and not (0 <= pvalue <= 1):
        print("Error: 'pvalue' debe ser None o un número entre 0 y 1.")
        return None

    # Comprueba valores nulos en el DataFrame
    if df.isnull().any().any():
        print("Error: El DataFrame contiene valores nulos. Imputa o elimina los valores nulos antes de continuar.")
        return None

    # Verifica que 'target_col' sea una variable numérica continua
    if not np.issubdtype(df[target_col].dtype, np.number):
        print("Error: 'target_col' debe ser una variable numérica continua.")
        return None

    # Si 'columns' está vacío, utiliza todas las columnas numéricas del DataFrame
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Filtra las columnas según las condiciones de correlación y p-value
    filtered_columns = []
    for col in columns:
        if col != target_col:
            correlation, p_value = pearsonr(df[col], df[target_col])

            if abs(correlation) > umbral_corr and (pvalue is None or p_value <= 1 - pvalue):
                filtered_columns.append(col)

    # Si la lista de columnas es grande, divide en varios pairplots
    num_plots = len(filtered_columns)
    num_subplots = min(num_plots, 5)

    for i in range(0, num_plots, num_subplots):
        cols_to_plot = filtered_columns[i:i + num_subplots] + [target_col]
        sns.pairplot(df[cols_to_plot])
        plt.show()


    return filtered_columns

def get_features_cat_regression(df, target_col, pvalue=0.05, cardinality_threshold=10):

    """
    Identifica columnas categóricas en un DataFrame que tienen una relación estadísticamente significativa
    con una columna objetivo numérica, utilizando T-Test para baja cardinalidad y Z-Test para alta cardinalidad.

    Argumentos:
    df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    target_col (str): El nombre de la columna objetivo numérica en el DataFrame.
    pvalue (float): El valor p umbral para determinar la significancia estadística.
    cardinality_threshold (int): El umbral de porcentaje para determinar alta cardinalidad en la columna objetivo.

    Retorna:
    - Lista de columnas categóricas que tienen una relación estadísticamente significativa con la columna objetivo.
    """
    
    if df.isnull().any().any():
        print("Advertencia: El DataFrame contiene valores NaN.")
        return None

    if target_col not in df.columns:
        print(f"La columna '{target_col}' no existe en el DataFrame.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"La columna '{target_col}' debe ser numérica.")
        return None

    cardinality_percentage = df[target_col].nunique() / len(df) * 100
    if cardinality_percentage < cardinality_threshold:
        print(f"La columna '{target_col}' no tiene alta cardinalidad. Cardinalidad como porcentaje del total: {cardinality_percentage:.2f}%")
        return None

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    significant_cols = []
    overall_mean = df[target_col].mean()

    for col in categorical_cols:
        unique_values = df[col].nunique()

        if unique_values > 1 and unique_values <= 30:
            for unique_value in df[col].unique():
                group = df[df[col] == unique_value][target_col].dropna()
                if group.var() == 0:  
                    continue  
                t_stat, p_val = stats.ttest_1samp(group, overall_mean, nan_policy='omit')
                if p_val < pvalue:
                    significant_cols.append(col)
                    break

        elif unique_values > 30:
            for unique_value in df[col].unique():
                subgroup = df[df[col] == unique_value][target_col]
                success_count = np.sum(subgroup > overall_mean)
                nobs = len(subgroup)

                if nobs == 0:  
                    continue

                z_stat, p_val = proportions_ztest(success_count, nobs, prop=0.5)
                if p_val < pvalue:
                    significant_cols.append(col)
                    break

    return significant_cols

def plot_features_cat_regression(df, target_col, columns=[], pvalue=0.05, with_individual_plot=False):
    """
    Genera histogramas agrupados de la variable target_col para cada uno de los valores de las variables
    categóricas en 'columns' que tienen una relación estadísticamente significativa con target_col,
    basado en un T-Test para baja cardinalidad y Z-Test para alta cardinalidad.
    
    Argumentos:
    df (pd.DataFrame): El DataFrame que contiene los datos a analizar.
    target_col (str): El nombre de la columna objetivo numérica en el DataFrame.
    columns (list): Lista de nombres de columnas categóricas a analizar. Si está vacía, se utilizarán todas las columnas numéricas.
    pvalue (float): El valor p umbral para determinar la significancia estadística.
    with_individual_plot (bool): Si es True, genera un histograma para cada columna categórica significativa.
    
    Retorna:
    - Lista de nombres de columnas categóricas que tienen una relación estadísticamente significativa con la columna objetivo.
    """
    
    if df.isnull().any().any():
        print("Advertencia: El DataFrame contiene valores NaN.")
        return None

    if not target_col or target_col not in df.columns:
        print("La columna target_col no está en el DataFrame o no se especificó correctamente.")
        return None

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("La columna target_col debe ser numérica.")
        return None

    if not columns:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        columns.extend(num_cols)
        df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')

    significant_cols = []

    for col in columns:
        if col not in df.columns:
            continue
        unique_values = df[col].nunique()

        if unique_values > 1:  
            for unique_value in df[col].unique():
                group = df[df[col] == unique_value][target_col].dropna()
                if group.var() == 0:  
                    continue
                if unique_values <= 30:  
                    t_stat, p_val = stats.ttest_1samp(group, df[target_col].mean(), nan_policy='omit')
                else:  
                    success_count = (group > df[target_col].mean()).sum()
                    nobs = len(group)
                    z_stat = (success_count - nobs * 0.5) / np.sqrt(nobs * 0.5 * (1 - 0.5))
                    p_val = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

                if p_val < pvalue:
                    significant_cols.append(col)
                    break 
                
    if with_individual_plot and significant_cols:
        for col in significant_cols:
            if col in df.columns:
                df.groupby(col)[target_col].plot(kind='hist', alpha=0.5, legend=True, title=f"Histograma de '{target_col}' por '{col}'")
                plt.xlabel(target_col)
                plt.ylabel('Frecuencia')
                plt.legend(title=col)
                plt.show()

    return significant_cols

def eval_model(target, predictions, problem_type, metrics):
    '''
    Evalúa el rendimiento de un modelo de machine learning, ya sea de regresión o clasificación, basándose en una lista de métricas específicas.

    Argumentos:
    target (array-like): Valores verdaderos del conjunto de datos.
    predictions (array-like): Predicciones generadas por el modelo.
    problem_type (str): Tipo de problema, 'regresión' o 'clasificación'.
    metrics (list of str): Lista de etiquetas de métricas para evaluar. Las métricas válidas para regresión incluyen 'RMSE', 'MAE', 'MAPE', 'GRAPH'. 
    Para clasificación, las métricas incluyen 'ACCURACY', 'PRECISION', 'RECALL', 'CLASS_REPORT', 'MATRIX', 'MATRIX_RECALL', 'MATRIX_PRED', 
    'PRECISION_X', 'RECALL_X', donde 'X' es una etiqueta de clase específica.

    Retorna:
    tupla: Una tupla que contiene los resultados de las métricas evaluadas en el orden en que se especificaron en el argumento `metrics`.

    Levanta:
    ValueError: Si el cálculo de 'MAPE' falla debido a valores de target con cero.

    Nota:
    La métrica 'GRAPH' muestra un gráfico pero no retorna un valor numérico, por lo tanto, no se incluye en la tupla de retorno.
    Las métricas 'PRECISION_X' y 'RECALL_X' requieren que 'X' sea reemplazado por la etiqueta específica de la clase a evaluar.
    '''
    results = []

    if problem_type == 'regresión':
        for metric in metrics:
            if metric == 'RMSE':
                rmse = np.sqrt(mean_squared_error(target, predictions))
                print(f'RMSE: {rmse}')
                results.append(rmse)
            elif metric == 'MAE':
                mae = mean_absolute_error(target, predictions)
                print(f'MAE: {mae}')
                results.append(mae)
            elif metric == 'MAPE':
                try:
                    mape = mean_absolute_percentage_error(target, predictions)
                    print(f'MAPE: {mape}')
                    results.append(mape)
                except ValueError as e:
                    print("Error al calcular MAPE:", e)
                    raise
            elif metric == 'GRAPH':
                plt.scatter(target, predictions)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.title('Comparison of True Values vs Predictions')
                plt.show()
                
    elif problem_type == 'clasificación':
        for metric in metrics:
            if metric == 'ACCURACY':
                accuracy = accuracy_score(target, predictions)
                print(f'Accuracy: {accuracy}')
                results.append(accuracy)
            elif metric.startswith('PRECISION_'):
                class_label = metric.split('_')[1]
                precision = precision_score(target, predictions, labels=[class_label], average=None)
                print(f'Precision for {class_label}: {precision[0]}')
                results.append(precision[0])
            elif metric.startswith('RECALL_'):
                class_label = metric.split('_')[1]
                recall = recall_score(target, predictions, labels=[class_label], average=None)
                print(f'Recall for {class_label}: {recall[0]}')
                results.append(recall[0])
            elif metric == 'PRECISION':
                precision = precision_score(target, predictions, average='macro')
                print(f'Precision: {precision}')
                results.append(precision)
            elif metric == 'RECALL':
                recall = recall_score(target, predictions, average='macro')
                print(f'Recall: {recall}')
                results.append(recall)
            elif metric == 'CLASS_REPORT':
                print(classification_report(target, predictions))
            elif metric == 'MATRIX':
                cm = confusion_matrix(target, predictions)
                ConfusionMatrixDisplay(cm).plot()
                plt.show()
            elif metric == 'MATRIX_RECALL':
                cm = confusion_matrix(target, predictions, normalize='true')
                ConfusionMatrixDisplay(cm).plot()
                plt.show()
            elif metric == 'MATRIX_PRED':
                cm = confusion_matrix(target, predictions, normalize='pred')
                ConfusionMatrixDisplay(cm).plot()
                plt.show()
    
    return tuple(results)


def is_discrete_numeric(series):
    """
    Determina si una Serie de pandas es numérica discreta.

    Argumentos:
    series (pandas.Series): Una Serie de pandas a ser verificada.

    Retorna:
    bool: True si la serie es numérica discreta (tiene <= 15 valores únicos y es numérica), False de lo contrario.
    """
    unique_values = series.unique()
    num_unique_values = len(unique_values)
    return num_unique_values <= 15 and pd.api.types.is_numeric_dtype(series)


def get_features_num_classification(df, target_col, pvalue=0.05):
    """
    Identifica columnas númericas en un DataFrame que tienen una relación estadísticamente significativa
    con una columna objetivo categórica, utilizando un test de ANOVA.

    Argumentos:
    df (pandas.DataFrame): El DataFrame de pandas que contiene el conjunto de datos.
    target_col (str): El nombre de la columna objetivo para clasificación.
    pvalue (float, opcional): El nivel de significancia para la selección de características. El valor predeterminado es 0.05.

    Retorna:
    Una lista de nombres de columna que representan características significativas para clasificación,
    según el test de ANOVA, donde la columna es numérica y su ANOVA con la columna designada por "target_col"
    supera el test de hipótesis con una significancia mayor o igual a 1-pvalue.
    """
    # Comprobar si el dataframe es de tipo DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'dataframe' no es un DataFrame de pandas.")
        return None
    
    # Comprobar si target_col es una columna del dataframe
    if target_col not in df.columns:
        print("Error: 'target_col' no es una columna válida del dataframe.")
        return None
    
    # Codificar target_col si es categórica
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        ordinal_encoder = OrdinalEncoder()
        df_encoded = df.copy()
        df_encoded[target_col] = ordinal_encoder.fit_transform(df[[target_col]])
    else:
        df_encoded = df
    
    # Comprueba valores nulos en el DataFrame
    if df_encoded.isnull().any().any():
        print("Error: El DataFrame contiene valores nulos. Imputa o elimina los valores nulos antes de continuar.")
        return None
    
    # Comprobar si pvalue es un valor float
    if not isinstance(pvalue, float):
        print("Error: El argumento 'pvalue' debe ser un valor float.")
        return None
    
    # Comprobar si pvalue está en el rango válido (0, 1)
    if not (0 < pvalue < 1.0):
        print("Error: El valor de 'pvalue' debe estar en el rango (0, 1).")
        return None
    
    # Si target_col es numérica, comprobar si es discreta y tiene baja cardinalidad
    if pd.api.types.is_numeric_dtype(df_encoded[target_col]):
        if not is_discrete_numeric(df_encoded[target_col]):
            print("Error: 'target_col' no es una variable numérica discreta.")
            return None
        
        cardinality = df_encoded[target_col].nunique()
        if cardinality > 0.1 * len(df_encoded):
            print("Error: 'target_col' no tiene una baja cardinalidad.")
            return None
    
    # Obtener las columnas numéricas del dataframe
    numeric_columns = df_encoded.select_dtypes(exclude=object).columns
    
    columnas_significativas = []
    columnas_no_significativas = []

    for col in numeric_columns:
        if col != target_col:
            groups = [group[col].dropna() for name, group in df.groupby(target_col)]
            f_value, p_valor = f_oneway(*groups)
            if p_valor <= pvalue:
                columnas_significativas.append(col)
            else:
                columnas_no_significativas.append(col)
    
    if columnas_no_significativas:
        print("Las siguientes columnas no pasaron el test de significancia:")
        print(columnas_no_significativas)
    
    print("\n Las siguientes columnas pasaron el test de significancia:")

    return columnas_significativas
       

def plot_features_num_classification(df, target_col="", columns=[], pvalue=0.05):

    """
    Genera pairplots para visualizar la relación entre variables numéricas en un DataFrame, 
    segmentadas por valores únicos de una columna target que tienen una relación estadísticamente significativa 
    utilizando un test de ANOVA.

    Argumentos:
    - df (pandas.DataFrame): El DataFrame que contiene los datos.
    - target_col (str): La columna que se utilizará para segmentar los datos en los pairplots. 
      Debe ser una columna del DataFrame df. Por defecto es una cadena vacía, lo que significa que no se realizará segmentación.
    - columns (list): Una lista de columnas del DataFrame df que se utilizarán en los pairplots. 
      Por defecto es una lista vacía, lo que significa que se utilizarán todas las columnas numéricas del DataFrame.
    - pvalue (float): El valor de p a utilizar para determinar la significancia de las variables numéricas en relación con target_col. 
      Por defecto es 0.05.

    Retorna:
    - Pairplots de acuerdo a las especificaciones anteriores
    - Columnas_significativas (list): Una lista de columnas numéricas significativas en relación con target_col, 
      de acuerdo con el valor de p especificado.
    

    Nota:
    - Si el número de valores únicos en target_col es mayor que 5, los pairplots se dividirán en múltiples subgráficos 
      segmentados por los valores únicos de target_col.
    - Se asume que la función get_features_num_classification está definida en otro lugar y devuelve las características numéricas 
      significativas en relación con target_col con un valor de p especificado.

    Raises:
    - TypeError: Si el argumento df no es un DataFrame de pandas.
    - ValueError: Si target_col no es una columna válida del DataFrame df, o si una o más columnas especificadas en columns 
      no están presentes en el DataFrame df, o si el argumento pvalue no es un valor float.

    Ejemplo:
    >>> plot_pairplots(dataframe, target_col="target", columns=["feature1", "feature2"], pvalue=0.05)
    """
    # Comprobar si el dataframe es de tipo DataFrame de pandas
    if not isinstance(df, pd.DataFrame):
        print("Error: El argumento 'dataframe' no es un DataFrame de pandas.")
        return None
    
    # Comprobar si la columna target_col está presente en el dataframe
    if target_col not in df.columns:
        print("Error: 'target_col' no es una columna válida del dataframe.")
        return None
    
    # Comprobar si la lista de columnas a considerar es válida
    if not all(col in df.columns for col in columns):
        print("Error: Una o más columnas especificadas no están presentes en el dataframe.")
        return None
    
    # Comprobar si la lista de columnas está vacía y si lo está, llenarla con las columnas numéricas
    if not columns:
        columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Comprobar si el valor de pvalue es válido
    if not isinstance(pvalue, float):
        print("Error: El argumento 'pvalue' debe ser un valor float.")
        return None
    
    # Obtener las columnas numéricas significativas usando la función get_features_num_classification
    significant_columns = get_features_num_classification(df, target_col, pvalue)
    
    # Filtrar las columnas de interés que sean significativas
    significant_columns = [col for col in columns if col in significant_columns]
    
    # Dividir el dataframe por valores únicos de target_col si el número de valores posibles es mayor que 5
    if len(df[target_col].unique()) > 5:
        unique_values = df[target_col].unique()
        num_plots = len(unique_values) // 5 + (1 if len(unique_values) % 5 != 0 else 0)
        
        for i in range(num_plots):
            start_index = i * 5
            end_index = min((i + 1) * 5, len(unique_values))
            target_subset = unique_values[start_index:end_index]
            
            # Filtrar el DataFrame por los valores únicos de target_col
            subset_df = df[df[target_col].isin(target_subset)]
            
            # Dividir las columnas significativas en grupos de máximo cinco columnas por pairplot
            num_cols = len(significant_columns)
            num_subplots = num_cols // 4 + (1 if num_cols % 4 != 0 else 0)
            
            for j in range(num_subplots):
                start_col_index = j * 4
                end_col_index = min((j + 1) * 4, num_cols)
                subset_columns = significant_columns[start_col_index:end_col_index]
                
                # Asegurar que target_col esté presente en el pairplot
                if target_col not in subset_columns:
                    subset_columns.append(target_col)
                
                # Pintar el pairplot
                sns.pairplot(subset_df, hue=target_col, vars=subset_columns)
                plt.show()
    else:
        # Pintar el pairplot con todas las variables significativas
        sns.pairplot(df, hue=target_col, vars=significant_columns)
        plt.show()
    
    return significant_columns


def get_features_cat_classification(dataframe, target_col, normalize=False, mi_threshold=0):
    """
    Devuelve una lista de características categóricas del dataframe que cumplen ciertos criterios de mutual information.

    Argumentos:
    - dataframe (DataFrame): El dataframe que contiene los datos.
    - target_col (str): El nombre de la columna que actúa como el objetivo para un modelo de clasificación.
    - normalize (bool, opcional): Indica si se debe normalizar el valor de la información mutua. Por defecto es False.
    - mi_threshold (float, opcional): Umbral para la información mutua. Por defecto es 0.

    Retorna:
    - list: Una lista de columnas categóricas que cumplen los criterios establecidos por los parámetros.
    """

    # Comprobación de que 'target_col' hace referencia a una variable categórica del dataframe.
    if target_col not in dataframe.columns:
        print(f"Error: '{target_col}' no es una columna válida en el dataframe.")
        return None
    
    if dataframe[target_col].dtype != 'object':
        print(f"Error: '{target_col}' no es una variable categórica en el dataframe.")
        return None

    # Comprobación de valores de 'normalize' y 'mi_threshold'.
    if not isinstance(normalize, bool):
        print("Error: 'normalize' debe ser un booleano.")
        return None

    if not isinstance(mi_threshold, (int, float)):
        print("Error: 'mi_threshold' debe ser un número.")
        return None

    if not 0 <= mi_threshold <= 1:
        print("Error: 'mi_threshold' debe estar entre 0 y 1.")
        return None

    # Encoding de las características categóricas si no están en formato numérico.
    encoder = OrdinalEncoder()
    encoded_data = dataframe.copy()
    encoded_data[dataframe.select_dtypes(include=['object']).columns] = encoder.fit_transform(dataframe.select_dtypes(include=['object']))

    # Cálculo de la información mutua.
    mi_scores = mutual_info_classif(encoded_data.drop(columns=[target_col]), encoded_data[target_col], discrete_features=True)

    # Normalización si se especifica.
    if normalize:
        mi_scores_normalized = mi_scores / np.sum(mi_scores)
        selected_features = encoded_data.drop(target_col, axis=1).columns[mi_scores_normalized >= mi_threshold].tolist()
    else:
        p = pd.DataFrame({"Features":encoded_data.drop(target_col, axis=1).columns, "resultados":mi_scores})
        mask = p["resultados"] >= mi_threshold
        selected_features = p[mask]["Features"].tolist()

    return selected_features


def plot_features_cat_classification(dataframe, target_col="", columns=[], mi_threshold=0.0, normalize=False):
    """
    1. Plot categorical features against classification target labels.

    Sección de Argumentos:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - target_col (str): Nombre de la columna que representa las etiquetas de clasificación. Por defecto, "".
    - columns (list): Lista de nombres de columnas categóricas a considerar. Por defecto, [].
    - mi_threshold (float): Umbral de información mutua para seleccionar características. Debe estar entre 0.0 y 1.0. Por defecto, 0.0.
    - normalize (bool): Indica si se deben normalizar las características seleccionadas. Por defecto, False.

    Sección de Retorna:
    - None

    """
    # Verificar si la lista de columnas está vacía
    if not columns:
        # Seleccionar automáticamente las variables categóricas del DataFrame
        cat_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
    else:
        cat_columns = columns
    
    # Comprobar si el umbral de información mutua es válido
    if mi_threshold < 0.0 or mi_threshold > 1.0:
        raise ValueError("El umbral de información mutua debe estar entre 0.0 y 1.0")
    
    # Verificar si la columna objetivo está presente
    if target_col != "":
        if target_col not in dataframe.columns:
            raise ValueError("La columna objetivo no está presente en el DataFrame")
        
    # Eliminar filas con valores nulos en las columnas categóricas
    dataframe = dataframe.dropna(subset=cat_columns)
    
    # Codificar las variables categóricas a numéricas
    encoder = LabelEncoder()
    for col in cat_columns:
        dataframe[col] = encoder.fit_transform(dataframe[col])
    
    # Calcular la información mutua entre las características categóricas y la columna objetivo
    mi_scores = mutual_info_classif(dataframe[cat_columns], dataframe[target_col])
    
    # Seleccionar características basadas en el umbral de información mutua
    selected_features = [col for col, mi_score in zip(cat_columns, mi_scores) if mi_score > mi_threshold]
    
    # Normalizar las características si se solicita
    if normalize:
        dataframe[selected_features] = dataframe[selected_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    # Visualizar la distribución de etiquetas para cada característica seleccionada
    for feature in selected_features:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, hue=target_col, data=dataframe)
        plt.title(f"Distribución de etiquetas para {feature}")
        plt.xlabel(feature)
        plt.ylabel("Conteo")
        plt.legend(title=target_col)
        plt.show()