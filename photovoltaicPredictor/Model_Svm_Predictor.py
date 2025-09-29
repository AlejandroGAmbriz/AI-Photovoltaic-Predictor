def Svm_Predictor(entryValues):
    """
    Predicción con modelo LinearSVR
    """
    import joblib as jb
    import numpy as np

    try:
        if len(entryValues) != 5:
            raise ValueError("Se requieren 5 valores: [T2M, RH2M, WS10M, CLRSKY, Lluvia]")
        
        model = jb.load('Model_SVM.pkl')
        scaler = jb.load('scaler_SVM.pkl')
        
        scaled_input = scaler.transform([entryValues])
        return float(model.predict(scaled_input)[0])
        
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return None