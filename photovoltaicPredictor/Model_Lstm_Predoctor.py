def Prediction_Lstm(climate_data):
    import numpy as np
    import pandas as pd
    import joblib
    from tensorflow.keras.models import load_model

    try:
        # === Cargar modelo y escaladores ===
        model = load_model('modelo_LSTM.keras', compile=False)
        scalers = [joblib.load(f'LSTM_scaler_{i}.save') for i in range(6)]

        # 2. Preprocesamiento de entrada
        features = ["T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]
        arr = np.array(climate_data)

        if arr.ndim == 1 and arr.shape[0] == 5:
            arr = np.tile(arr, (24, 1))  # repetir los mismos valores para 24 horas

        if arr.shape != (24, 5):
            raise ValueError(f"Entrada debe tener shape (24, 5), recibida: {arr.shape}")

        df_input = pd.DataFrame(arr, columns=features)

        # Normalizar cada columna con su escalador correspondiente
        scaled = np.zeros_like(df_input.values)
        for i in range(5):
            scaled[:, i] = scalers[i].transform(df_input.iloc[:, i].values.reshape(-1, 1)).flatten()

        x_in = scaled.reshape(1, 24, 5)

        # 3. Predicción
        y_scaled = model.predict(x_in, verbose=0)
        print("Predicción normalizada (salida del modelo):", y_scaled[0][0])

        # Desnormalizar la salida con el sexto escalador (WH)
        prediction = float(scalers[5].inverse_transform(y_scaled)[0][0])
        print("Predicción desnormalizada (WH):", prediction)

        return prediction

    except Exception as e:
        print(f"Error en predicción (LSTM_predictor): {e}")
        return None
