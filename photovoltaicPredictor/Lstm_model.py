def Lstm(df_nasa, df_cocoa):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import joblib
    import os

    # === Preparar datos ===
    df_total = df_nasa.copy()
    df_total['WH'] = df_cocoa['WH'].reset_index(drop=True)
    df_total = df_total.select_dtypes(exclude=['datetime'])
    data = df_total.values.astype(np.float32)

    # Normalización por columnas (como MATLAB)
    scalers = []
    data_norm = np.zeros_like(data)
    for i in range(data.shape[1]):
        scaler = MinMaxScaler()
        data_norm[:, i:i+1] = scaler.fit_transform(data[:, i:i+1])
        scalers.append(scaler)

    # Guardar escaladores
    os.makedirs('model', exist_ok=True)
    for i, scaler in enumerate(scalers):
        joblib.dump(scaler, f'LSTM_scaler_{i}.save')

    # Separar datos (80% entrenamiento, 20% validación)
    train_size = int(len(data_norm) * 0.8)
    train_data = data_norm[:train_size]
    val_data = data_norm[train_size - 24:]

    # Crear secuencias
    def create_sequences(data, seq_len=24):
        x, y = [], []
        for i in range(len(data) - seq_len):
            x.append(data[i:i+seq_len, :-1])  # entradas
            y.append(data[i+seq_len, -1])     # salida
        return np.array(x), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)

    # === Definir modelo LSTM profundo ===
    model = Sequential([
        tf.keras.layers.Input(shape=(24, 5)),
        LSTM(100, return_sequences=True),
        Dropout(0.1),
        LSTM(100, return_sequences=False),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 200], [0.01, 0.005, 0.001])
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse')

    # Entrenamiento
    model.fit(X_train, y_train, epochs=350, batch_size=16, validation_data=(X_val, y_val), verbose=0)

    # Guardar modelo
    model.save('modelo_LSTM.keras')

    # === Validación ===
    y_pred = model.predict(X_val).flatten()
    scaler_WH = scalers[-1]
    y_val_des = scaler_WH.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_des = scaler_WH.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_val_des, y_pred_des))
    mae = mean_absolute_error(y_val_des, y_pred_des)
    r2 = r2_score(y_val_des, y_pred_des)

    print(f"Validación - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"WH total validación: {np.sum(y_val_des):.2f} kWh")

    # Gráfico
    plt.figure()
    plt.plot(y_val_des, label='WH real')
    plt.plot(y_pred_des, label='WH predicho')
    plt.title('Validación - WH Real vs Predicho')
    plt.legend()
    plt.show()
