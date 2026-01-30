import cv2
import torch
import fastcv
import numpy as np

img = cv2.imread("artifacts/test.jpg")
noise = np.random.rand(*img.shape)
img[noise < 0.05] = 0 
img[noise > 0.95] = 255 
cv2.imwrite("artifacts/output_noise.jpg", img)
print("saved noisy image")
img_tensor = torch.from_numpy(img).cuda()
blurred_tensor = fastcv.medianBlur(img_tensor, 7)

blurred_image = blurred_tensor.cpu().numpy()
cv2.imwrite("artifacts/output_blur.jpg", blurred_image)

print("saved blurred image.")