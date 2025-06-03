import argparse
import sys
from dataclasses import dataclass
from math import isnan, sqrt, asin, pi
from typing import Iterable, Iterator, List, Optional, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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

def map_list(lst, func):
    return [func(x) for x in lst]

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

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

def points_to_signal(points: List[Point], width: int) -> np.ndarray:
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
        
    return signal

def extract_signal(binary_img: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    height, width = binary_img.shape
    points_by_col = get_point_locations(binary_img)
    all_points = flatten(points_by_col)
    
    if not all_points:
        return None
        
    # Debug: show all candidate points
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
    
    return points_to_signal(path, width)

def load_and_binarize(image_path: str, threshold=180, invert=False) -> np.ndarray:
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    if invert:
        binary = (img_array > threshold).astype(np.uint8) * 255
    else:
        binary = (img_array < threshold).astype(np.uint8) * 255
    
    return binary

def main():
    parser = argparse.ArgumentParser(description='Extract ECG signal from image using Viterbi algorithm')
    parser.add_argument('image_path', type=str, help='Path to input image file')
    parser.add_argument('--threshold', type=int, default=180, 
                        help='Binarization threshold (0-255), default=180')
    parser.add_argument('--invert', action='store_true', 
                        help='Invert image (for dark backgrounds)')
    parser.add_argument('--debug', action='store_true', 
                        help='Show debugging visualizations')
    args = parser.parse_args()

    try:
        binary_img = load_and_binarize(args.image_path, args.threshold, args.invert)
        height, width = binary_img.shape
        
        signal = extract_signal(binary_img, args.debug)
        
        if signal is None:
            print("Error: No signal detected. Try adjusting --threshold or --invert.")
            return
            
        signal = height - signal
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        ax1.imshow(binary_img, cmap='gray')
        valid_indices = np.where(~np.isnan(signal))[0]
        if len(valid_indices) > 0:
            ax1.plot(valid_indices, height - signal[valid_indices], 'r-', linewidth=1.5, alpha=0.7)
        ax1.set_title('Image with Extracted Signal Overlay')
        ax2.plot(signal, 'b-')
        ax2.set_title('Extracted ECG Signal')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Signal Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim(0, width)
        
        valid_signal = signal[~np.isnan(signal)]
        if len(valid_signal) > 0:
            signal_min = np.min(valid_signal)
            signal_max = np.max(valid_signal)
            
            signal_range = signal_max - signal_min
            padding = 1 * signal_range
            
            ax2.set_ylim(signal_min - padding, signal_max + padding)
        
        plt.tight_layout()
        plt.show()
        
        print("Success! Signal extracted with length:", len(signal))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()