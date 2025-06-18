import cv2
import numpy as np
import os

#Ainda nao funciona, mas a estrutura parece promissor

IMAGE_PATH = 'IMAGE_PATH'
OUTPUT_DIR = 'crops'

LOWER_HSV = np.array([160, 30, 30])
UPPER_HSV = np.array([180, 255, 255])

MIN_AREA = 200

ERODE_KERNEL = (5, 5)  
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

kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, ERODE_KERNEL)
eroded = cv2.erode(mask, kernel_erode, iterations=1)

kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL)
opened = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close)
save_debug('opened.png', opened)

contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contornos encontrados: {len(contours)}")

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
print("Ajuste LOWER_HSV, UPPER_HSV, MIN_AREA e kernels conforme debug (mask.png e opened.png).")
