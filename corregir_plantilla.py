import numpy as np
import os

# Ruta del archivo .npy
ruta = "plantilla_biometrica1.npy"

# Validar existencia del archivo
if not os.path.isfile(ruta):
    raise FileNotFoundError(f"❌ No se encontró el archivo: {ruta}")

# Cargar el contenido
plantilla = np.load(ruta)
print(f"📂 Forma original de la plantilla: {plantilla.shape}")

# Validar dimensión
if plantilla.ndim == 2:
    # Calcular el promedio entre los vectores
    plantilla_promedio = np.mean(plantilla, axis=0)
    print(f"✅ Se promedió la plantilla. Nueva forma: {plantilla_promedio.shape}")
elif plantilla.ndim == 1:
    plantilla_promedio = plantilla
    print(f"ℹ️ Ya es una plantilla 1-D. No se requiere corrección.")
else:
    raise ValueError("❌ Formato inesperado en la plantilla.")

# Guardar sobreescribiendo
np.save("plantilla_biometrica.npy", plantilla_promedio)
print("💾 Plantilla corregida guardada exitosamente.")
