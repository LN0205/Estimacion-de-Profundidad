import cv2
import torch
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
encoder = 'vits'  # or 'vitb', 'vitl', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
raw_img1 = cv2.imread('rollos2.png')
raw_img2 = cv2.imread('rollos1.png')

start_time = time.time()
depth1 = model.infer_image(raw_img1)
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

umb1, umb2 = 66, 42

start_time = time.time()
img1 = np.where(depth_normalized1 < umb1, 0, depth_normalized1)
end_time1c = time.time() - start_time

start_time = time.time()
img2 = np.where(depth_normalized2 < umb2, 0, depth_normalized2)
end_time2c = time.time() - start_time

start_time = time.time()
_, bw_img1 = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
end_time1d = time.time() - start_time

start_time = time.time()
_, bw_img2 = cv2.threshold(img2, 30, 255, cv2.THRESH_BINARY)
end_time2d = time.time() - start_time

start_time = time.time()
color_img1 = cv2.applyColorMap(bw_img1, cv2.COLORMAP_JET)
end_time1e = time.time() - start_time

start_time = time.time()
color_img2 = cv2.applyColorMap(bw_img2, cv2.COLORMAP_JET)
end_time2e = time.time() - start_time

start_time = time.time()
color_img1[(color_img1[:, :, 2] != 128)] = [0, 0, 0]
color_img1[(color_img1[:, :, 2] == 128)] = [255, 255, 255]
end_time1f = time.time() - start_time

start_time = time.time()
color_img2[(color_img2[:, :, 2] != 128)] = [0, 0, 0]
color_img2[(color_img2[:, :, 2] == 128)] = [255, 255, 255]
end_time2f = time.time() - start_time

plt.figure(figsize=(15, 8))
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(raw_img1, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 1')
plt.axis('off')
plt.subplot(2, 4, 2)
plt.imshow(depth_normalized1, cmap='jet')
plt.title('Mapa de inferencia')
plt.axis('off')
plt.subplot(2, 4, 3)
plt.imshow(img1, cmap='jet')
plt.title('Mapa de umbralizacion')
plt.axis('off')
plt.subplot(2, 4, 4)
plt.imshow(color_img1)
plt.title('Mapa de binarizacion')
plt.axis('off')
plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(raw_img2, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original 2')
plt.axis('off')
plt.subplot(2, 4, 6)
plt.imshow(depth_normalized2, cmap='jet')
plt.title('Mapa de inferencia')
plt.axis('off')
plt.subplot(2, 4, 7)
plt.imshow(img2, cmap='jet')
plt.title('Mapa de umbralizacion')
plt.axis('off')
plt.subplot(2, 4, 8)
plt.imshow(color_img2)
plt.title('Mapa de binarizacion')
plt.axis('off')
plt.show()
print('Tiempo de inferencia de la Primera imagen es:',end_time1a)
print('Tiempo de inferencia de la Segunda imagen es:',end_time2a)
print('Tiempo de umbralizacion de la Primera imagen es:',end_time1b+end_time1c)
print('Tiempo de umbralizacion de la Segunda imagen es:',end_time2b+end_time2c)
print('Tiempo de binarizacion de la Primera imagen es:',end_time1d+end_time1e+end_time1f)
print('Tiempo de binarizacion de la Segunda imagen es:',end_time2d+end_time2e+end_time2f)
print('Tiempo de total de la Primera imagen es:',end_time1a+end_time1b+end_time1c+end_time1d+end_time1e+end_time1f)
print('Tiempo de total de la Segunda imagen es:',end_time2a+end_time2b+end_time2c+end_time2d+end_time2e+end_time2f)