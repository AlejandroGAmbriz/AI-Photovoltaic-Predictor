def Random_Forest(dfNasaResampled, dfCocoaResampled):
    """
    Entrena un modelo Random Forest para predecir WH usando datos climatológicos de la NASA y COCOA.
    
    Parámetros:
    dfNasaResampled : DataFrame
        Datos climatológicos de la NASA con columnas ["Fecha", "T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]
    dfCocoaResampled : DataFrame
        Datos de COCOA que deben contener las columnas ["Fecha", "WH"] 
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import joblib as jb
    import numpy as np
    from time import time

    print("Inicio de entrenamiento Random Forest sin features derivados")
    
    # ====== 1. PREPROCESAMIENTO solo con las 5 variables originales ======
    # Selecciona únicamente las columnas base sin crear nuevas features
    X = dfNasaResampled.drop(columns=["Fecha"]).loc[:, ["T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]]
    y = dfCocoaResampled["WH"]
    
    # Estadísticas descriptivas para chequeo
    print("\nEstadísticas descriptivas de los datos:")
    print(X.describe())
    print("\nRango de WH:", y.min(), "-", y.max())
    
    # Factor de escala si WH es muy grande
    y_scaling_factor = 1
    if y.max() > 1000:
        y_scaling_factor = 1000
        y = y / y_scaling_factor
        print(f"\nAplicado factor de escala a WH: 1/{y_scaling_factor}")
    
    # Limpieza básica de NaNs e infinitos
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    y = y[X.index]
    
    # ====== 2. DIVISIÓN Y ESCALADO ======
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ====== 3. BÚSQUEDA DE HIPERPARÁMETROS ======
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 0.5, 0.7],
        'bootstrap': [True]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=50, cv=5,
        scoring='neg_mean_squared_error',
        verbose=2, random_state=42, n_jobs=-1
    )
    
    print("\nIniciando búsqueda de hiperparámetros...")
    start_time = time()
    search.fit(X_train_scaled, y_train)
    print(f"\nBúsqueda completada en {(time()-start_time)/60:.1f} minutos")
    
    # ====== 4. EVALUACIÓN Y GUARDADO ======
    best_model = search.best_estimator_
    
    y_pred_train = best_model.predict(X_train_scaled) * y_scaling_factor
    y_pred_test = best_model.predict(X_test_scaled) * y_scaling_factor
    y_train_orig = y_train * y_scaling_factor
    y_test_orig = y_test * y_scaling_factor
    
    print("\n=== RESULTADOS FINALES ===")
    print(f"MSE_train: {mean_squared_error(y_train_orig, y_pred_train):.4f}")
    print(f"MSE_test: {mean_squared_error(y_test_orig, y_pred_test):.4f}")
    print(f"R2_train: {r2_score(y_train_orig, y_pred_train):.4f}")
    print(f"R2_test: {r2_score(y_test_orig, y_pred_test):.4f}")
    print(f"MAE_train: {mean_absolute_error(y_train_orig, y_pred_train):.4f}")
    print(f"MAE_test: {mean_absolute_error(y_test_orig, y_pred_test):.4f}")
    print("\nMejores parámetros:", search.best_params_)
    
    jb.dump(best_model, "Model_RF.pkl")
    jb.dump(scaler, "Scaler_RF.pkl")
    print("\nModelo guardado correctamente")
