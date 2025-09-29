import tkinter as tk
import os
import joblib as jl
from tkinter import filedialog
from Modify_files import Modify_and_recive_files
from Lstm_model import Lstm
from Model_Lstm_Predoctor import Prediction_Lstm
from Rf_model import Random_Forest
from Model_Rf_Predictor import Prediction_RF
from Svm_Model import Svm
from Model_Svm_Predictor import Svm_Predictor

DataBaseFile = None
dfCocoaResampled = None
dfNasaResampled = None
rootSelector = None

def load_model(model_name):
    "Carga el modelo correspondiente y abre la ventana de prediccion"
    
    global rootSelector
    
    if model_name == "LSTM":
        #Cargar modelo LSTM preentrenado
        try:
            model = load_model('modelo_LSTM.keras')
            print('Modelo LSTM cargado correctamente')
        except Exception as e:
            print(f"Error al cargar modelo LSTM: {e}")
            return
    elif model_name == "SVM":
        try:
            model = jl.load('Model_SVM.pkl')
            print("Modelo SVM cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo SVM: {e}")
            return
    elif model_name == "Random Forest":
        try:
            model = jl.load("Model_RF.pkl")
            print("Modelo Random Forest cargado correctamente")
        except Exception as e:
            print(f"Error al cargar modelo Random Forest: {e}")
            return
    else:
        print("Modelo no reconocido")
        return
     # Cerrar la ventana de selección de modelos
    if rootSelector:
        rootSelector.destroy()
        rootSelector = None

      # Abrir la ventana de predicción y pasar el modelo cargado
    Root_predictor(model_name, model)
    
    
def Charge_file(IdButton): #La funcion Charge_file  permite seleccionar un archivo
                            #obtiene la ruta y el nombre del arvhico seleccioando
                            #y asigna la ruta a DataBaseFile
                            
    global  DataBaseFile
    
    #obtiene ruta del archivo
    filePath = filedialog.askopenfilename(filetypes =[ ('Archivo Excel', '*.xlsx'), 
                                                      ('Archivo CSV', '*.csv')])
    fileName = os.path.basename(filePath)#Obtiene nombre del archivo
    
    entryFile.delete(0, tk.END)
    entryFile.insert(0, fileName)
    DataBaseFile = filePath#Recibe la ruta del archivo 
       
#Ventana de seleccion de modelo
def Root_model_selector ():#obtenemos los DF de la funcion Modify_and_recive_files
    
    global dfNasaResampled, dfCocoaResampled , rootSelector
    print('root model selector ejecutado')
    root.destroy()#Elimina la ventana Principal
    
    dfNasaResampled, dfCocoaResampled = Modify_and_recive_files(DataBaseFile)
    print('nasa: ', dfNasaResampled, 'cocoa: ', dfCocoaResampled)
    rootSelector = tk.Tk()
    
    rootSelector.title('Predictor Fotovoltaico')
    rootSelector.geometry('400x300')
    
    #Se envian los DF al modelo seleccionado
    buttonLstm = tk.Button (rootSelector, text = 'LSTM', command = lambda: [Lstm(dfNasaResampled,dfCocoaResampled) ,
                                                        Root_predictor('LSTM')])
    buttonLstm.pack(pady = '20')    
    
    buttonSvm = tk.Button(rootSelector, text = 'SVM', command = lambda: [Svm(dfNasaResampled,dfCocoaResampled) , 
                                                        Root_predictor('SVM')])
    buttonSvm.pack(pady = '20')
    
    buttonRf = tk.Button(rootSelector,text = 'Random Forest', command = lambda:
                                                            [Random_Forest(dfNasaResampled, dfCocoaResampled), 
                                                             Root_predictor('Random Forest')])
    buttonRf.pack(pady = '20')
    
    rootSelector.mainloop()

#Ventana para ingreso de datos del usuario y prediccion del modelo
def Root_predictor(IdButton, model = None):
    
    global rootSelector
    
    print('root predictor ejecutado') 
   #Elimina la ventana Root_model_selector
    if rootSelector:
        
        rootSelector.destroy()
        rootSelector = None
   
    rootPredictor = tk.Tk()
    rootPredictor.title('Predictor Fotovoltaico')
    rootPredictor.geometry('400x300')
   
    # Etiquetas y campos de entrada
    
    labels = ["T2M", "RH2M", "WS10M", "CLRSKY", "Lluvia"]
    inputs = []

    for i, label in enumerate(labels):
        tk.Label(rootPredictor, text=label).grid(row=i, column=0, padx=5, pady=5)
        entry = tk.Entry(rootPredictor)
        entry.grid(row=i, column=1, padx=5, pady=5)
        inputs.append(entry)
        
    # Label para mostrar el resultado de la predicción

    resultado_label = tk.Label(rootPredictor, text="Predicción de WH: ")
    resultado_label.grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)

    # Función para capturar valores e invocar la predicción
    def realizar_prediccion(IdButton):
        try:
            # Capturar los valores ingresados
            entryValues = []
            for entry in inputs:
                try:
                    value = float(entry.get())  # Intentar convertir el valor a flotante
                except ValueError:
                    value = 0  # Si no es numérico, asignar un valor por defecto
                entryValues.append(value)

            print("Valores ingresados:", entryValues)

            # Llamar a la función de predicción con los valores ingresados y el modelo elegido
            if IdButton == 'LSTM':
                prediccion = Prediction_Lstm(entryValues)
            elif IdButton == 'SVM':
                prediccion = Svm_Predictor(entryValues)
            elif IdButton == 'Random Forest':
                prediccion = Prediction_RF(entryValues)
                
            # Mostrar el resultado de la predicción
            if prediccion is not None:
                if prediccion < 0:
                    resultado_label.config(text=f"Predicción de WH: {0}")
                else:
                    resultado_label.config(text=f"Predicción de WH: {prediccion:.2f}")
            else:
                resultado_label.config(text="Error en la predicción (Main).")
                
        except Exception as e:
            print("Error durante la predicción:", e)
            resultado_label.config(text="Error en los datos ingresados.")

    # Botón para capturar valores y predecir
    tk.Button(rootPredictor, text="Capturar Valores y Predecir", command= lambda: realizar_prediccion(IdButton)).grid(
        row=len(labels), column=0, columnspan=2, pady=10)
       
def Root_already_trained():
    
    print('root already trained ejecutado')
    root.destroy()#Elimina la ventana principal 
    
    rootAleradyTrained = tk.Tk()
    rootAleradyTrained.title('Predictor Fotovoltaico')
    rootAleradyTrained.geometry('400x300')
    
    buttonTrainedLSTM = tk.Button(rootSelector, text = 'LSTM',command = lambda: load_model('LSTM'))
    buttonTrainedLSTM.pack(pady = 20)
    buttonTrainedSVM = tk.Button(rootSelector, text = 'SVM',command = lambda: load_model('SVM'))
    buttonTrainedSVM.pack(pady = 20)
    buttonTrainedRF = tk.Button(rootSelector, text = 'Random Forest',command = lambda: load_model('Random Forest'))
    buttonTrainedRF.pack(pady = 20)
    
    rootAleradyTrained.mainloop()
    
#Ventana inicial, se reciben arvhivos y se envian
root = tk.Tk()#Se crea la ventana principal

root.title('Predictor Fotovoltaico')
root.geometry('400x300')

entryFile = tk.Entry(root, width = 50)
entryFile.pack(pady = 20)

buttonFileLoad = tk.Button(root, text = 'Cargar Base de Datos',command = 
                           lambda: Charge_file('buttonFileLoad'))
buttonFileLoad.pack(pady = 20)

buttonNext = tk.Button(root, text='Siguiente', command = lambda:Root_model_selector())
buttonNext.pack(pady = 10)

buttonAlreadyTrain = tk.Button(root, text= 'Modelos preentrenados', command = lambda: Root_already_trained())
buttonAlreadyTrain.pack(pady = 10)
    
root.mainloop() #mantiene abierta la ventana y su interactividad
