# Cesar-Soria-Tesis-

Debido a su peso >800MB se deben descargar los siguientes archivos desde los siguientes links:
1. Colocarlo en la ruta principal: https://1drv.ms/u/c/486c58d5cece92c6/IQDPK9Vx_-E2RYD1GcZS9wPTAU_nCICmUIbBrmzM1fNw-Ho?e=tmqW2F
2. Colocar la carpeta completa dentro de la carpeta pretrained_models : https://1drv.ms/f/c/486c58d5cece92c6/IgB_n7_r4NBFQYScGlCDXYX5AUWVrq_qOOvxno_kjkaA77U?e=HyOQir

Debemos crear un ambiente venv con la version de Python 3.11.9 de preferencia 
Antes de iniciar las pruebas se deben instalar las librerías con el comando: pip install -r requirements.txt 

El primer archivo que vamos a ejecutar para obtener la plantilla biométrica con la cual hacemos todas las pruebas es biometria_vocal.py
Allí elegimos el nombre con el cual queremos guardar la plantilla con la opción 1 

Para probar el modelo de identidad biométrica ejecutamos el archivo verificarvoz.py 
Es necesario recalcar que por defecto el nombre de la plantilla que lee es Cesar.npy por lo que si se guarda la plantilla con otro nombre debemos cambiarlo en la ruta
Una vez ejecutado comparará con el modelo todos los archivos ubicados en la carpeta RUTA_TERCEROS = r'data\suplantados' (Si se requiere comparar con más audios se pueden colocarlos) y nos dará un valor decimal que interpreta el porcentaje de similitud con las muestras 

Para probar el modelo antispoofing debemos ejecutar detector_IA.py el cual revisará todos los audios de la carpeta ubicada en data/suplantados y nos dará una estimación en % de la probabilidad de uso de IA, de la misma manera podemos agregar más audios a la carpeta o cambiar la ruta si lo requerimos.
