import cv2
import numpy as np
import os

IMAGE_PATH = 'IMAGE_PATH'
OUTPUT_DIR = 'sinais'

LOWER_HSV = np.array([160, 30, 30])
UPPER_HSV = np.array([180, 255, 255])

MIN_AREA = 2000

ERODE_KERNEL = (1, 1)  
CLOSE_KERNEL = (15, 15)

os.makedirs(OUTPUT_DIR, exist_ok=True)
def save_debug(name, img):
    cv2.imwrite(name, img)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Imagem nao encontrada em {IMAGE_PATH}")
orig = img.copy()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
save_debug('mask.png', mask)

kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados: {len(contours)}")

debug = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug, contours, -1, (0, 255, 0), 1)
cv2.imwrite("mask_dilated_debug.png", debug)

rois = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    rois.append((x, y, w, h))

print(f"Regioes candidatas: {len(rois)}")

rois = sorted(rois, key=lambda r: (r[1], r[0]))
for i, (x, y, w, h) in enumerate(rois, 1):
    crop = orig[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(OUTPUT_DIR, f'regiao_{i}.png'), crop)
    cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)

ecg_boxes = 'ecg_with_boxes.png'
cv2.imwrite(ecg_boxes, orig)
print(f"Cortadas {len(rois)} regioes em '{OUTPUT_DIR}'")
