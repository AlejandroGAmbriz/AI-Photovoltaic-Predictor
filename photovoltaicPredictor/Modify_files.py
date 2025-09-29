def Modify_and_recive_files(allDataFile):
    import pandas as pd
    """
    Parámetros:
    file : str
        Ruta del archivo Excel o CSV que contiene los datos.
    """
    print("Función Modify_and_recive_files ejecutada")
    print(f'Archivo recibido: {allDataFile}')

    # Lectura del archivo
    if allDataFile.endswith(".csv"):
        df = pd.read_csv(allDataFile)
    elif allDataFile.endswith((".xlsx", ".xls")):
        df = pd.read_excel(allDataFile)
    else:
        raise ValueError("Formato de archivo no válido. Debe ser CSV o Excel.")

    # Verificar si el archivo tiene las columnas esperadas
    required_columns = ["Fecha", "T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia", "WH"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("El archivo no tiene las columnas necesarias.")

    # Crear DataFrame para NASA (todas las columnas excepto "WH")
    dfNasaResampled = df[["Fecha", "T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]].copy()  # Cambio clave aquí
    
    # Crear DataFrame para COCOA (solo la columna "Fecha" y "WH")
    dfCocoaResampled = df[["Fecha", "WH"]].copy()
    
    # Aplicar la función de mitigación de ruido
    dfNasaResampled = dfNasaResampled.applymap(Mitigacion_ruido)
    dfCocoaResampled = dfCocoaResampled.applymap(Mitigacion_ruido)

    print('Nasa DF:', '\n', dfNasaResampled.head(), '\nCocoa DF:', '\n', dfCocoaResampled.head())
    return dfNasaResampled, dfCocoaResampled

def Mitigacion_ruido(value):
    import pandas as pd
    """
    Función para eliminar el ruido en los datos.
    Reemplaza valores -999 o NaN con 0.
    """
    if value == -999 or pd.isna(value):
        return 0
    return value