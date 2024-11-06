import cv2
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import time
import os

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
########################
carpeta = r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F4\F4C"  # Cambia esto por la ruta de tu carpeta

# Crear una lista para almacenar las rutas de las imágenes
paths = []

# Recorrer todos los archivos en la carpeta
for archivo in os.listdir(carpeta):
    # Verificar si el archivo es un archivo JPEG
    if archivo.endswith(".jpg") or archivo.endswith(".jpeg"):
        # Crear la ruta completa del archivo y agregarla a la lista
        paths.append(os.path.join(carpeta, archivo))

#######################
plt.figure(figsize=(15, 8))
#................................1
h=int(0)
for k in paths:

    raw_img1 = cv2.imread(k)
    '''
    ###########
    image_float = raw_img1w.astype(np.float32)

    # Encontrar los valores mínimo y máximo de los píxeles
    min_val = np.min(image_float)
    max_val = np.max(image_float)

    # Normalizar la imagen
    normalized_image = (image_float - min_val) * (255 / (max_val - min_val))

    # Convertir de nuevo a uint8
    normalized_image = normalized_image.astype(np.uint8)

    # Convertir de BGR a RGB
    raw_img1  = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB)
   
    image_float = raw_img1w.astype(np.float32) / 255.0

    # Aplicar una transformación de contraste
    # Aquí usamos una transformación cuadrática para aumentar el contraste
    contrasted_image = np.clip(image_float ** 1.2, 0, 1)

    # Convertir de nuevo a uint8
    contrasted_image = (contrasted_image * 255).astype(np.uint8)

    # Mostrar la imagen ajustada
    raw_img1=cv2.cvtColor(contrasted_image, cv2.COLOR_BGR2RGB)
    '''
#######################
  
    start_time = time.time()
    depth1 = model.infer_image(raw_img1) # HxW raw depth map in numpy
    end_time1a = time.time() - start_time

    start_time = time.time()
    depth_normalized1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    end_time1b = time.time() - start_time

    umb1=20
 
    img1=depth_normalized1.copy()

    start_time = time.time()
    for i in range(depth_normalized1.shape[0]):
        for j in range(depth_normalized1.shape[1]):
            if (depth_normalized1[i][j]<umb1):
                img1[i][j]=0
    end_time1c = time.time() - start_time

    start_time = time.time()
    _, bw_img1 = cv2.threshold(img1, umb1, 255, cv2.THRESH_BINARY)
    end_time1d = time.time() - start_time

    #gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
    start_time = time.time()                    
    color_img1 = cv2.applyColorMap(bw_img1, cv2.COLORMAP_JET)
    end_time1e = time.time() - start_time

    # Mostrar la imagen original y el mapa de profundidad

    start_time = time.time() 
    for i in range(depth_normalized1.shape[0]):
        for j in range(depth_normalized1.shape[1]):
            if (color_img1[i][j][2]==128):
                color_img1[i][j]=[255,255,255]
            else:
                color_img1[i][j]=[0,0,0]
    end_time1f = time.time() - start_time


#for j in range(5):
# Imagen original
    plt.subplot(len(paths), 4, h+1)
    plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original 1')
    plt.axis('off')
    h=h+1
    # Mapa de profundidad
    plt.subplot(len(paths), 4, h+1)
    plt.imshow(depth_normalized1, cmap='jet')
    plt.title('Mapa de Profundidad 1')
    plt.axis('off')
    h=h+1
    plt.subplot(len(paths), 4, h+1)
    plt.imshow(img1, cmap='jet')
    plt.title('Mapa de Profundidad 1')
    plt.axis('off')
    h=h+1
    plt.subplot(len(paths), 4, h+1)
    plt.imshow(color_img1)
    plt.title('Mapa de Profundidad 1')
    plt.axis('off')
    h=h+1

plt.show()

#print('Tiempo de inferencia de la Primera imagen es:',end_time1a)
#print('Tiempo de inferencia de la Segunda imagen es:',end_time2a)
#('Tiempo de umbralizacion de la Primera imagen es:',end_time1b+end_time1c)
#('Tiempo de umbralizacion de la Segunda imagen es:',end_time2b+end_time2c)
#('Tiempo de binarizacion de la Primera imagen es:',end_time1d+end_time1e+end_time1f)
#('Tiempo de binarizacion de la Segunda imagen es:',end_time2d+end_time2e+end_time2f)