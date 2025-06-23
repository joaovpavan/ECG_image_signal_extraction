import argparse
import sys
from dataclasses import dataclass
from math import isnan, sqrt, asin, pi
from typing import Iterable, Iterator, List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
import cv2

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def index(self) -> int:
        return round(self.x)

    def __repr__(self):
        return f"({self.x:.1f}, {self.y:.1f})"

def euclidean_distance(x: float, y: float) -> float:
    return sqrt(x**2 + y**2)

def angle_from_offsets(x: float, y: float) -> float:
    dist = euclidean_distance(x, y)
    if dist < 1e-6:
        return 0.0
    return asin(y/dist) * 180 / pi

def flatten(lst):
    return [item for sublist in lst for item in sublist]

def find_contiguous_regions(one_dim_image: np.ndarray) -> list:
    regions = []
    start = None
    for idx, val in enumerate(one_dim_image):
        if val > 0 and start is None:
            start = idx
        elif val == 0 and start is not None:
            regions.append((start, idx - 1))
            start = None
    if start is not None:
        regions.append((start, len(one_dim_image) - 1))
    return regions

def find_contiguous_region_centers(one_dim_image: np.ndarray) -> list:
    return [int((start + end) / 2) for start, end in find_contiguous_regions(one_dim_image)]

def distance_between_points(a: Point, b: Point) -> float:
    return euclidean_distance(a.x - b.x, a.y - b.y)

def angle_between_points(a: Point, b: Point) -> float:
    dx = b.x - a.x
    dy = b.y - a.y
    return angle_from_offsets(dx, dy)

def angle_similarity(angle1: float, angle2: float) -> float:
    diff = abs(angle1 - angle2)
    return (180 - min(diff, 360 - diff)) / 180

def get_point_locations(image: np.ndarray) -> List[List[Point]]:
    columns = image.T
    locations = []
    for col_idx, col_data in enumerate(columns):
        row_centers = find_contiguous_region_centers(col_data)
        points = [Point(float(col_idx), float(row)) for row in row_centers]
        locations.append(points)
    return locations

def path_score(current: Point, candidate: Point, cand_angle: float) -> float:
    DISTANCE_WEIGHT = 0.6
    ANGLE_WEIGHT = 0.4

    dist = distance_between_points(candidate, current)
    angle = angle_between_points(candidate, current)
    angle_diff = 1 - angle_similarity(angle, cand_angle)

    return dist * DISTANCE_WEIGHT + angle_diff * ANGLE_WEIGHT

def get_adjacent_points(points_by_col, path_data, current_col: int, lookback: int) -> list:
    left_bound = max(0, current_col - lookback)
    candidates = []

    for col_idx in range(left_bound, current_col):
        for point in points_by_col[col_idx]:
            if point in path_data:
                score, _, angle = path_data[point]
                candidates.append((score, point, angle))

    return candidates

def interpolate_points(a: Point, b: Point) -> Iterator[Point]:
    if abs(a.x - b.x) < 1e-6:
        yield a
        return

    slope = (b.y - a.y) / (b.x - a.x)
    start_x = min(a.x, b.x)
    end_x = max(a.x, b.x)

    for x in np.arange(start_x, end_x, 1.0):
        y = a.y + slope * (x - a.x)
        yield Point(x, y)

def points_to_signal(points: List[Point], width: int, height: int) -> np.ndarray:
    if not points:
        return np.array([np.nan] * width)

    points = sorted(points, key=lambda p: p.x)
    signal = np.full(width, np.nan, dtype=float)

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]

        if 0 <= start.index < width:
            signal[start.index] = start.y

        for pt in interpolate_points(start, end):
            idx = pt.index
            if 0 <= idx < width and np.isnan(signal[idx]):
                signal[idx] = pt.y

    if points and 0 <= points[-1].index < width:
        signal[points[-1].index] = points[-1].y

    return height - signal

def extract_signal(binary_img: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    height, width = binary_img.shape
    points_by_col = get_point_locations(binary_img)
    all_points = flatten(points_by_col)

    if not all_points:
        return None

    if debug:
        plt.figure(figsize=(12, 6))
        plt.imshow(binary_img, cmap='gray')
        plt.scatter([p.x for p in all_points], [p.y for p in all_points], 
                   s=2, c='red', alpha=0.5)
        plt.title("Candidate Points")
        plt.show()

    dp_table = {}
    lookback = 3

    for point in points_by_col[0]:
        dp_table[point] = (0.0, None, 0.0)

    for col_idx in range(1, len(points_by_col)):
        for current_point in points_by_col[col_idx]:
            candidates = get_adjacent_points(points_by_col, dp_table, current_point.index, lookback)

            if not candidates:
                dp_table[current_point] = (0.0, None, 0.0)
                continue

            best_score = float('inf')
            best_prev = None
            best_angle = 0.0

            for cand_score, cand_point, cand_angle in candidates:
                cost = path_score(current_point, cand_point, cand_angle)
                total_score = cand_score + cost

                if total_score < best_score:
                    best_score = total_score
                    best_prev = cand_point
                    best_angle = angle_between_points(cand_point, current_point)

            dp_table[current_point] = (best_score, best_prev, best_angle)

    end_region = max(0, width - width//10)
    end_candidates = []

    for col in points_by_col[end_region:]:
        for point in col:
            if point in dp_table:
                end_candidates.append((dp_table[point][0], point))

    if not end_candidates:
        return None

    _, best_end = min(end_candidates, key=lambda x: x[0])

    path = []
    current = best_end
    while current is not None:
        path.append(current)
        _, current, _ = dp_table.get(current, (0, None, 0))

    path.reverse()

    if debug:
        plt.figure(figsize=(12, 6))
        plt.imshow(binary_img, cmap='gray')
        plt.scatter([p.x for p in all_points], [p.y for p in all_points], 
                   s=2, c='gray', alpha=0.3)
        plt.plot([p.x for p in path], [p.y for p in path], 'r-', linewidth=1.5)
        plt.title("Selected Path")
        plt.show()

    return points_to_signal(path, width, height)

def remove_grid_adaptively(img_array: np.ndarray, initial_threshold=180, debug=False) -> np.ndarray:
    best_img = img_array.copy()
    min_signal_energy = float('inf')
    best_thresh = initial_threshold

    for thresh in range(initial_threshold - 30, initial_threshold + 30, 5):
        _, binary = cv2.threshold(img_array, thresh, 255, cv2.THRESH_BINARY_INV)
        cleaned = cv2.medianBlur(binary, 3)
        signal = extract_signal(cleaned)
        if signal is not None:
            energy = np.nansum(np.diff(signal)**2)
            if energy < min_signal_energy:
                min_signal_energy = energy
                best_img = cleaned
                best_thresh = thresh

    if debug:
        print(f"Melhor limiar: {best_thresh}")
        plt.imshow(best_img, cmap='gray')
        plt.title("Imagem com grade removida")
        plt.axis('off')
        plt.show()

    return best_img

def process_lotes(root_dir: str, threshold: int = 180, debug: bool = False):
    saida_dir = root_dir.rstrip("/\\") + "_extraidos"
    os.makedirs(saida_dir, exist_ok=True)

    for lote in sorted(os.listdir(root_dir)):
        caminho_lote = os.path.join(root_dir, lote)
        if not os.path.isdir(caminho_lote):
            continue

        print(f"\nProcessando lote: {lote}")
        pasta_saida_lote = os.path.join(saida_dir, lote)
        pasta_csv = os.path.join(pasta_saida_lote, 'csv')
        os.makedirs(pasta_csv, exist_ok=True)

        for nome_img in sorted(os.listdir(caminho_lote)):
            if not nome_img.lower().endswith('.png'):
                continue

            caminho_img = os.path.join(caminho_lote, nome_img)
            img = Image.open(caminho_img).convert('L')
            img_array = np.array(img)

            processed_img = remove_grid_adaptively(img_array, initial_threshold=threshold, debug=debug)
            signal = extract_signal(processed_img, debug=debug)

            if signal is None:
                print(f"  [X] Nenhum sinal extraído: {nome_img}")
                continue

            nome_base = os.path.splitext(nome_img)[0]
            csv_path = os.path.join(pasta_csv, f"{nome_base}.csv")
            pd.DataFrame({"sinal": signal}).to_csv(csv_path, index=False)
            print(f"  [✔] CSV salvo: {csv_path}")

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(signal, 'b-')
            ax.set_title(f'Sinal extraído - {nome_img}')
            ax.set_xlim(0, len(signal))
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(pasta_saida_lote, f"{nome_base}_sinal.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  [✔] Gráfico salvo: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai sinais de várias imagens em lotes")
    parser.add_argument("diretorio", help="Diretório contendo pastas com imagens")
    parser.add_argument("--threshold", type=int, default=180, help="Limiar de binarização inicial")
    parser.add_argument("--debug", action="store_true", help="Mostrar visualizações de depuração")
    args = parser.parse_args()

    process_lotes(args.diretorio, threshold=args.threshold, debug=args.debug)
