import sounddevice as sd
import soundfile as sf
import os
import time

# --- CONFIGURACIÓN ---
DURACION = 4  # Segundos por audio
FRECUENCIA = 16000  # Hz requeridos por el modelo ECAPA-TDNN
CARPETA_SALIDA = r'data\suplantados'
TOTAL_GRABACIONES = 20

# Asegurar que la carpeta existe
os.makedirs(CARPETA_SALIDA, exist_ok=True)

print(f"🚀 Iniciando sesión de grabación: {TOTAL_GRABACIONES} muestras de {DURACION}s.")
print(f"📂 Los archivos se guardarán en: {CARPETA_SALIDA}\n")

for i in range(1, TOTAL_GRABACIONES + 1):
    archivo_nombre = os.path.join(CARPETA_SALIDA, f'audio{i}.wav')
    
    print(f"--- 🎤 Grabación {i}/{TOTAL_GRABACIONES} ---")
    print("⏳ Prepárate... (1s)")
    time.sleep(1)

    
    print("🔴 GRABANDO...")
    # Grabación a 16kHz y canal mono
    audio = sd.rec(int(DURACION * FRECUENCIA), samplerate=FRECUENCIA, channels=1, dtype='float32')
    sd.wait()
    print("✅ Finalizado.")
    
    # Guardar el archivo en formato WAV PCM
    sf.write(archivo_nombre, audio, FRECUENCIA)
    print(f"💾 Guardado como: {archivo_nombre}\n")
    
    # Breve pausa para que el locutor descanse entre tomas
    if i < TOTAL_GRABACIONES:
        time.sleep(1)

print("✨ Sesión completada con éxito. Ya tienes tus 20 muestras listas.")