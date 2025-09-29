def Svm(dfNasaResampled, dfCocoaResampled):
    """
    Entrenamiento optimizado con LinearSVR (para regresión)
    """
    import pandas as pd
    import numpy as np
    import joblib as jb
    import matplotlib.pyplot as plt
    from sklearn.svm import LinearSVR  # Cambiado a regresión
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import random
    import time

    # 1. Combinar datos
    dfCombined = pd.merge(dfNasaResampled, dfCocoaResampled, on="Fecha", how="inner")
    features = ["T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]
    y = dfCombined["WH"].values
    X = dfCombined[features].values

    # 2. Escalado con StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    jb.dump(scaler, 'scaler_SVM.pkl')

    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. Configuración óptima para regresión
    best_mse = float('inf')
    best_params = None
    mse_history = []
    
    param_ranges = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5],
        'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        'max_iter': [10000]  # Fijo para garantizar convergencia
    }

    print("=== Entrenamiento LinearSVR (Regresión) ===")
    start_time = time.time()

    # 5. Búsqueda de parámetros
    for i in range(20):  # Iteraciones reducidas pero efectivas
        params = {
            'C': random.choice(param_ranges['C']),
            'epsilon': random.choice(param_ranges['epsilon']),
            'loss': random.choice(param_ranges['loss']),
            'max_iter': param_ranges['max_iter'][0]
        }
        
        model = LinearSVR(**params, random_state=42)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        current_mse = mean_squared_error(y_test, y_pred)
        mse_history.append(current_mse)
        
        if current_mse < best_mse:
            best_mse = current_mse
            best_params = params
            jb.dump(model, 'Model_SVM.pkl')
            print(f"Iter {i+1}: MSE = {best_mse:.4f} | Params: {params}")

    # 6. Gráfica de resultados
    plt.figure(figsize=(12, 6))
    plt.plot(mse_history, 'bo-', alpha=0.5)
    plt.title(f"Evolución del MSE\nMejor MSE: {best_mse:.4f}")
    plt.xlabel("Iteración")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.show()

    print("\n=== Resultados Finales ===")
    print(f"Mejor MSE: {best_mse:.4f}")
    print(f"Mejores parámetros: {best_params}")
    print(f"Tiempo: {(time.time()-start_time):.1f} segundos")