import cv2
import torch
import pandas
import numpy as np
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import time

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

raw_img1 = cv2.imread(r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\F1C\C12.jpg")
raw_img2 = cv2.imread(r"E:\repos\yolov8_training\Depth-Anything-V2\DATASETCHUTE\F1\F1C\C18.jpg")

start_time = time.time()
depth1 = model.infer_image(raw_img1) # HxW raw depth map in numpy
end_time1a = time.time() - start_time

start_time = time.time()
depth2 = model.infer_image(raw_img2)
end_time2a = time.time() - start_time

start_time = time.time()
depth_normalized1 = cv2.normalize(depth1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
end_time1b = time.time() - start_time

start_time = time.time()
depth_normalized2 = cv2.normalize(depth2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
end_time2b = time.time() - start_time

umb1=11
umb2=11
img1=depth_normalized1.copy()
img2=depth_normalized2.copy()

start_time = time.time()
for i in range(depth_normalized1.shape[0]):
    for j in range(depth_normalized1.shape[1]):
        if (depth_normalized1[i][j]<umb1):
            img1[i][j]=0
end_time1c = time.time() - start_time

start_time = time.time()
for i in range(depth_normalized2.shape[0]):
    for j in range(depth_normalized2.shape[1]):
        if (depth_normalized2[i][j]<umb2):
            img2[i][j]=0
end_time2c = time.time() - start_time

start_time = time.time()
_, bw_img1 = cv2.threshold(img1, umb1, 255, cv2.THRESH_BINARY)
end_time1d = time.time() - start_time

start_time = time.time()
_, bw_img2 = cv2.threshold(img2, umb2, 255, cv2.THRESH_BINARY)
end_time2d = time.time() - start_time

#gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
start_time = time.time()                    
color_img1 = cv2.applyColorMap(bw_img1, cv2.COLORMAP_JET)
end_time1e = time.time() - start_time

start_time = time.time()  
color_img2 = cv2.applyColorMap(bw_img2, cv2.COLORMAP_JET)
end_time2e = time.time() - start_time
# Mostrar la imagen original y el mapa de profundidad

start_time = time.time() 
for i in range(depth_normalized1.shape[0]):
    for j in range(depth_normalized1.shape[1]):
        if (color_img1[i][j][2]==128):
            color_img1[i][j]=[255,255,255]
        else:
            color_img1[i][j]=[0,0,0]
end_time1f = time.time() - start_time

start_time = time.time()
for i in range(depth_normalized2.shape[0]):
    for j in range(depth_normalized2.shape[1]):
        if (color_img2[i][j][2]==128):
            color_img2[i][j]=[255,255,255]
        else:
            color_img2[i][j]=[0,0,0]
end_time2f = time.time() - start_time

plt.figure(figsize=(15, 8))
#................................1
# Imagen original
plt.subplot(5, 4, 1)
plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 1')
plt.axis('off')

# Mapa de profundidad
plt.subplot(5, 4, 2)
plt.imshow(depth_normalized1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 3)
plt.imshow(img1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 4)
plt.imshow(color_img1)
plt.title('Mapa de Profundidad 1')
plt.axis('off')
#................................2
plt.subplot(5, 4, 5)
plt.imshow(cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 2')
plt.axis('off')

# Mapa de profundidad
plt.subplot(5, 4, 6)
plt.imshow(depth_normalized2, cmap='jet')
plt.title('Mapa de Profundidad 2')
plt.axis('off')

plt.subplot(5, 4, 7)
plt.imshow(img2, cmap='jet')
plt.title('Mapa de Profundidad 2')
plt.axis('off')

plt.subplot(5, 4, 8)
plt.imshow(color_img2)
plt.title('Mapa de Profundidad 2')
plt.axis('off')

#................................3
# Imagen original
plt.subplot(5, 4, 9)
plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 1')
plt.axis('off')

# Mapa de profundidad
plt.subplot(5, 4, 10)
plt.imshow(depth_normalized1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 11)
plt.imshow(img1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 12)
plt.imshow(color_img1)
plt.title('Mapa de Profundidad 1')
plt.axis('off')

#................................4
# Imagen original
plt.subplot(5, 4, 13)
plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 1')
plt.axis('off')

# Mapa de profundidad
plt.subplot(5, 4, 14)
plt.imshow(depth_normalized1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 15)
plt.imshow(img1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 16)
plt.imshow(color_img1)
plt.title('Mapa de Profundidad 1')
plt.axis('off')

#................................5
# Imagen original
plt.subplot(5, 4, 17)
plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 1')
plt.axis('off')

# Mapa de profundidad
plt.subplot(5, 4, 18)
plt.imshow(depth_normalized1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 19)
plt.imshow(img1, cmap='jet')
plt.title('Mapa de Profundidad 1')
plt.axis('off')

plt.subplot(5, 4, 20)
plt.imshow(color_img1)
plt.title('Mapa de Profundidad 1')
plt.axis('off')



plt.show()

print('Tiempo de inferencia de la Primera imagen es:',end_time1a)
print('Tiempo de inferencia de la Segunda imagen es:',end_time2a)
print('Tiempo de umbralizacion de la Primera imagen es:',end_time1b+end_time1c)
print('Tiempo de umbralizacion de la Segunda imagen es:',end_time2b+end_time2c)
print('Tiempo de binarizacion de la Primera imagen es:',end_time1d+end_time1e+end_time1f)
print('Tiempo de binarizacion de la Segunda imagen es:',end_time2d+end_time2e+end_time2f)