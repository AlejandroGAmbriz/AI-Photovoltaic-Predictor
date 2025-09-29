def Prediction_RF(entryValues):
    """
    Realiza la predicción usando el modelo Random Forest guardado y normaliza los datos de entrada.

    Parámetros:
    entryValues : list
        Conjunto de valores climatológicos ingresados por el usuario.

    Retorno:
    float
        Predicción de WH basada en los valores ingresados.
    """
    import numpy as np
    import joblib as jl
    import pandas as pd

    try:
        # Cargar el modelo Random Forest y el scaler
        model = jl.load('Model_RF.pkl')
        scaler = jl.load('Scaler_RF.pkl')  # Cargar el scaler guardado

        # Nombres de las características
        featureNames = ["T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]

        # Convertir la lista de datos en un DataFrame con nombres de columnas
        entryValues_df = pd.DataFrame([entryValues], columns=featureNames)
        


        # Normalizar los datos de entrada utilizando el scaler cargado
        entryValues_scaled = scaler.transform(entryValues_df)

        # Realizar predicción con el modelo Random Forest
        prediction = model.predict(entryValues_scaled)

        return float(prediction[0])  # Enviar la predicción como un número flotante

    except Exception as e:
        print("Error al realizar la predicción:", e)
        return None