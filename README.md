# Signal Extraction Project

This is a really preliminary repository containing an attempt at signal extraction from images, particularly focused on ECG (electrocardiogram) signals. The code implements a Viterbi-based algorithm to trace and extract signal lines from scanned or digital ECG images.

## Inspiration

This project was mostly inspired by the paper:

**"Electrocardiogram digitization and processing using computer vision techniques"**  
Available at: https://www.sciencedirect.com/science/article/abs/pii/S0169260722002723?via%3Dihub

## Usage

```
python test.py path/to/image.png [--threshold VALUE] [--invert] [--debug]
```

### Parameters:
- `image_path`: Path to input ECG image file
- `--threshold`: Binarization threshold (0-255), default=180
- `--invert`: Invert signal and background binary values
- `--debug`: Show debugging visualizations for algorithm steps

## Status

This repository is in a preliminary stage and the implementation is experimental. Further improvements and optimizations are planned.
