# Homework 1 – Image Enhancement

**Computer Vision, Fall 2026** – University of Verona
*Master's Degree in Computer Engineering for Intelligent Systems*

An image enhancement pipeline built exclusively with classical computer vision methods (OpenCV and NumPy) applied to low-contrast images.
Each original image is synthetically degraded in a reproducible manner (contrast reduction plus additive Gaussian noise); four methods are subsequently evaluated and compared against the ground truth using PSNR, SSIM, and MAE metrics.

---

## Compared Methods

1. **Baseline**: the degraded image without any processing.
2. **Linear Contrast Stretching**: applied to the $L$ luminance channel of the CIE-Lab color space (2–98 percentiles).
3. **Histogram Equalization**: applied to the $L$ luminance channel of the CIE-Lab color space.
4. **Denoise + CLAHE**:
   - color Non-Local Means (NLM) denoising;
   - contrast stretching on $L$ (1–99 percentiles);
   - CLAHE applied to $L$;
   - Gaussian blur on the $a$ and $b$ color channels with chromatic gain adjustment.

> **Note:** the enhancement methods process only the degraded image and have no prior knowledge of the degradation parameters.

---

## Requirements & Setup

- **Python**: 3.11
- **Dependencies**: NumPy 2.4.6, Matplotlib 3.11.2, OpenCV (`opencv-python`) 5.0.0.93, pandas, scikit-image (listed in `requirements.txt`).

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Linux / macOS (Bash)

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

The `venv/` directory is excluded from Git tracking (add it to `.gitignore`).

---

## Execution

Run the script **from the repository root**, because all paths are relative:

```bash
python main.py
```

The script reads the inputs from `image/`, degrades them, applies all four enhancement methods, and writes the degraded images, the processed results, the comparison figures and the metrics CSV to `results/`.

---

## Parameters

Constants defined at the top of `main.py` (identical across all images):

| Parameter | Value | Description |
|---|---|---|
| `ALPHA` | 0.5 | Contrast scaling factor |
| `CENTER` | 128 | Mid-gray reference point |
| `NOISE_SIGMA` | 10 | Noise standard deviation |
| `SEED` | 42 | Random seed for reproducibility |

**Degradation model:** $g = \text{clip}(\alpha \cdot (f - \text{CENTER}) + \text{CENTER} + n,\ 0,\ 255)$, where $n \sim \mathcal{N}(0, \sigma^2)$ independently per channel.

Degraded images are saved in lossless PNG format to avoid secondary JPEG compression artifacts.

**Method 4 configuration (Denoise + CLAHE):** color NLM with $h = h_{\text{color}} = 20$ (search window 21, template window 7), percentile stretching on $L$ (1–99), CLAHE (`clip_limit` = 1.5, tile grid $8 \times 8$), $5 \times 5$ Gaussian kernel ($\sigma = 2$) on $a$ and $b$, `chroma_gain` = 1.5.

---

## Directory Structure

```text
.
├── image/                           # Original images (img, img2, img3A, img4, img5 in PNG format)
├── results/                         # Output directory
│   ├── degraded/                    # Synthetically degraded images (PNG)
│   ├── processed/                   # Output of each enhancement method per image
│   ├── figures/                     # Visual comparison plots
│   └── risultati_metriche.csv       # PSNR, SSIM, and MAE scores (sep: ';', decimal: ',')
├── funzioni.py                      # Degradation, enhancement, and metric functions
├── main.py                          # Main execution script
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## Evaluation Metrics

- **PSNR (dB)**: measured against the original image (higher is better).
- **SSIM**: computed with scikit-image across all three channels with `data_range=255` (higher is better).
- **MAE**: mean absolute error relative to the original (lower is better).

The metrics quantify structural and pixel-wise fidelity to the original, not perceptual or subjective preference.

---

## Dataset Information

The five test images are sourced from Unsplash and distributed under the Unsplash License (free commercial and non-commercial use and redistribution, no mandatory attribution).

| File | Description | Source | License |
|---|---|---|---|
| img.png | White wall with sharp shadows | Unsplash | Unsplash License |
| img2.png | Orange sunrise over a foggy valley | Unsplash | Unsplash License |
| img3A.png | Night path with fog and streetlights | Unsplash | Unsplash License |
| img4.png | Yellow wall with sharp shadows | Unsplash | Unsplash License |
| img5.png | Misty mountain landscape | Unsplash | Unsplash License |
