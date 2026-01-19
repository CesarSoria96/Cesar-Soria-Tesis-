import os
import glob
from detector_zenodo_ultra_v3 import DetectorZenodoUltraV3

# --- CONFIGURACIÓN ---
CARPETA_AUDIOS = r'data\suplantados'
MODELO_PATH = "modelo_zenodo_ultra_v3.joblib"

# 1. Inicializar y cargar el detector (una sola vez para eficiencia)
detector = DetectorZenodoUltraV3()
detector.cargar_modelo(MODELO_PATH) #

# 2. Obtener lista de todos los audios .wav en la carpeta
lista_audios = glob.glob(os.path.join(CARPETA_AUDIOS, "*.wav"))

if not lista_audios:
    print(f"❌ No se encontraron archivos .wav en: {CARPETA_AUDIOS}")
else:
    print(f"🚀 Iniciando análisis de {len(lista_audios)} muestras...")
    print(f"{'Archivo':<20} | {'Veredicto':<10} | {'Prob. Real':<12} | {'Prob. IA':<10}")
    print("-" * 65)

    # 3. Bucle de procesamiento
    for ruta_audio in lista_audios:
        nombre_archivo = os.path.basename(ruta_audio)
        
        # Ejecutar predicción multidominio
        resultado = detector.predecir(ruta_audio) #

        if "error" not in resultado:
            veredicto = "Humano" if not resultado['is_deepfake'] else "IA"
            p_real = f"{resultado['probability_real']*100:.2f}%"
            p_ia = f"{resultado['probability_deepfake']*100:.2f}%"
            
            # Imprimir fila de la tabla
            print(f"{nombre_archivo:<20} | {veredicto:<10} | {p_real:<12} | {p_ia:<10}")
        else:
            print(f"{nombre_archivo:<20} | ❌ ERROR: {resultado['error']}")

    print("-" * 65)
    print("✅ Análisis por lotes completado.")