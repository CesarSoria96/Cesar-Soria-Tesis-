import os
import glob
import numpy as np
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

# --- CONFIGURACIÓN ---
RUTA_TERCEROS = r'data\suplantados'
PLANTILLA_PATH = "usuarios_biometria/Cesar.npy"
UMBRAL = 0.65 #

# Evitar advertencias de Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# 1. Cargar el motor biométrico ECAPA-TDNN
verificador = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# 2. Cargar plantilla maestra (Centroide de 192 dimensiones)
if not os.path.exists(PLANTILLA_PATH):
    raise FileNotFoundError(f"❌ No se encontró la plantilla en: {PLANTILLA_PATH}")

plantilla = np.load(PLANTILLA_PATH)
print(f"📂 Plantilla maestra cargada. Iniciando validación por lotes...")

# 3. Localizar audios de terceros
lista_audios = glob.glob(os.path.join(RUTA_TERCEROS, "*.wav"))

if not lista_audios:
    print(f"❌ No se encontraron archivos .wav en: {RUTA_TERCEROS}")
else:
    print(f"🚀 Procesando {len(lista_audios)} muestras frente a la identidad registrada.")
    print(f"\n{'Archivo':<25} | {'Similitud':<10} | {'Resultado'}")
    print("-" * 55)

    # 4. Bucle de verificación biométrica
    for ruta_audio in lista_audios:
        nombre_archivo = os.path.basename(ruta_audio)
        
        # Carga y estandarización (16 kHz, Mono)
        waveform, sample_rate = torchaudio.load(ruta_audio)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Extracción del vector biométrico (Embedding)
        embedding = verificador.encode_batch(waveform).squeeze().detach().cpu().numpy()
        
        # Normalización L2 para asegurar justicia en la comparación
        embedding = embedding / np.linalg.norm(embedding)

        # Cálculo de Similitud Coseno
        # Fórmula: Similitud = 1 - Distancia Coseno
        similitud = 1 - cosine(plantilla, embedding)
        
        veredicto = "✅ COINCIDE" if similitud >= UMBRAL else "❌ RECHAZADO"
        
        print(f"{nombre_archivo:<25} | {similitud:.4f}    | {veredicto}")

    print("-" * 55)
    print("✅ Validación de identidad completada.")