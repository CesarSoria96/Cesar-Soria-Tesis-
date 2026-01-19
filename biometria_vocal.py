import os
import numpy as np
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

# ==============================
# CONFIGURACIÓN DEL SISTEMA
# ==============================
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Cargar modelo de Speaker Recognition
verificador = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Carpeta donde se guardan las plantillas biométricas
CARPETA_USUARIOS = "usuarios_biometria"
os.makedirs(CARPETA_USUARIOS, exist_ok=True)

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def procesar_audio(ruta_audio):
    """Carga un audio, lo convierte a mono y lo normaliza a 16 kHz"""
    waveform, sample_rate = torchaudio.load(ruta_audio)

    # Convertir a mono si es necesario
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resamplear a 16kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    return waveform


def extraer_embedding(ruta_audio):
    """Devuelve el embedding normalizado del audio"""
    waveform = procesar_audio(ruta_audio)
    embedding = verificador.encode_batch(waveform).squeeze().detach().cpu().numpy()
    # Normalización L2
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def registrar_usuario(nombre_usuario, lista_audios):
    """
    Crea una plantilla biométrica para un usuario promediando múltiples audios.
    - nombre_usuario: str -> identificador del usuario
    - lista_audios: list[str] -> rutas de archivos de audio del usuario
    """
    embeddings = [extraer_embedding(audio) for audio in lista_audios]
    plantilla_final = np.mean(embeddings, axis=0)  # promedio
    plantilla_final = plantilla_final / np.linalg.norm(plantilla_final)  # normalización

    ruta_guardado = os.path.join(CARPETA_USUARIOS, f"{nombre_usuario}.npy")
    np.save(ruta_guardado, plantilla_final)
    print(f"✅ Plantilla biométrica de {nombre_usuario} guardada en {ruta_guardado}")


def verificar_usuario(nombre_usuario, ruta_audio, umbral=0.65):
    """
    Verifica si un audio corresponde a un usuario registrado.
    - nombre_usuario: str -> identificador del usuario
    - ruta_audio: str -> archivo de audio a verificar
    - umbral: float -> valor mínimo de similitud para aceptar
    """
    ruta_plantilla = os.path.join(CARPETA_USUARIOS, f"{nombre_usuario}.npy")
    if not os.path.isfile(ruta_plantilla):
        raise FileNotFoundError(f"❌ No existe plantilla para el usuario {nombre_usuario}")

    # Cargar plantilla y extraer embedding del audio de prueba
    plantilla = np.load(ruta_plantilla)
    embedding_prueba = extraer_embedding(ruta_audio)

    # Calcular similitud coseno
    similitud = 1 - cosine(plantilla, embedding_prueba)

    print(f"\n🔍 Similitud: {similitud:.4f}")
    if similitud >= umbral:
        print(f"✅ La voz corresponde al usuario {nombre_usuario}")
        return True
    else:
        print(f"❌ La voz NO corresponde al usuario {nombre_usuario}")
        return False

# ==============================
# EJEMPLO DE USO
# ==============================
if __name__ == "__main__":
    import glob
    
    # Obtener todos los archivos de audio de la carpeta "legitimos"
    carpeta_legitimos = "data/legitimos"
    audios_legitimos = glob.glob(os.path.join(carpeta_legitimos, "*.wav"))
    
    if not audios_legitimos:
        print(f"❌ No se encontraron archivos .wav en la carpeta {carpeta_legitimos}")
        print("📁 Asegúrate de que la carpeta 'legitimos' existe y contiene archivos de audio .wav")
    else:
        print(f"📂 Encontrados {len(audios_legitimos)} archivos de audio en {carpeta_legitimos}")
        
        # Mostrar menú de opciones
        print("\n=== SISTEMA DE BIOMETRÍA VOCAL ===")
        print("1. Registrar usuario con audios de la carpeta 'legitimos'")
        print("2. Verificar usuario con un audio específico")
        print("3. Listar usuarios registrados")
        
        opcion = input("\nSelecciona una opción (1-3): ").strip()
        
        if opcion == "1":
            # Registro de usuario
            nombre_usuario = input("📝 Nombre del usuario a registrar: ").strip()
            if len(audios_legitimos) >= 2:
                print(f"🎯 Registrando usuario '{nombre_usuario}' con {len(audios_legitimos)} audios...")
                registrar_usuario(nombre_usuario, audios_legitimos)
            else:
                print("⚠️ Se necesitan al menos 2 audios para crear una plantilla robusta")
        
        elif opcion == "2":
            # Verificación de usuario
            nombre_usuario = input("👤 Nombre del usuario a verificar: ").strip()
            ruta_audio = input("🎙️ Ruta del audio de prueba: ").strip()
            
            if os.path.isfile(ruta_audio):
                umbral = float(input("🎯 Umbral de similitud (0.65 recomendado): ") or "0.65")
                verificar_usuario(nombre_usuario, ruta_audio, umbral)
            else:
                print(f"❌ No se encontró el archivo: {ruta_audio}")
        
        elif opcion == "3":
            # Listar usuarios registrados
            usuarios_registrados = glob.glob(os.path.join(CARPETA_USUARIOS, "*.npy"))
            if usuarios_registrados:
                print(f"\n📋 Usuarios registrados ({len(usuarios_registrados)}):")
                for i, usuario in enumerate(usuarios_registrados, 1):
                    nombre = os.path.basename(usuario).replace(".npy", "")
                    print(f"   {i}. {nombre}")
            else:
                print("📋 No hay usuarios registrados aún")
        
        else:
            print("❌ Opción no válida")

    print("\n👤 Sistema biométrico de voz listo para pruebas.")
