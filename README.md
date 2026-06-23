# Image Compression using Discrete Cosine Transform (DCT)

## Overview
This project implements an image compression system using the Discrete Cosine Transform (DCT) technique. DCT is widely used in standards like JPEG to reduce the storage size of digital images while maintaining acceptable visual quality by converting spatial image data into the frequency domain.

## Objectives
- Design and implement an image compression system using DCT.
- Compress digital images (grayscale) and reduce storage size.
- Reconstruct images using the Inverse Discrete Cosine Transform (IDCT).
- Evaluate the compression performance using metrics such as Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and an estimated compression ratio.

## Scope
- Focuses on DCT-based compression for digital grayscale images.
- Divides images into 8x8 blocks, applies DCT, quantizes to remove high-frequency details, and reconstructs the image.
- Computes performance metrics.
- Excludes video compression or deep-learning-based techniques.

## Prerequisites
Ensure you have Python 3 installed. Install the necessary libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

You can run the script via the command line. Ensure you have an input image available.

### Basic Compression
To compress an image with the default quality (50) and see the results:
```bash
python dct_compression.py -i your_image.jpg
```

### Adjusting Quality
You can specify a custom quality factor between 1 (maximum compression, lowest quality) and 100 (minimum compression, highest quality):
```bash
python dct_compression.py -i your_image.jpg -q 20
```

### Multi-Quality Comparison
To generate a side-by-side comparison of the image compressed at various quality levels, use the `--compare` flag:
```bash
python dct_compression.py -i your_image.jpg --compare
```

## Functions Breakdown
- **Preprocessing:** The image is converted into grayscale if it isn't already.
- **Block Division:** The image is padded (if necessary) and divided into 8x8 blocks.
- **DCT & Quantization:** Each block undergoes 2D DCT and quantization using a standard JPEG luminance quantization matrix scaled by the specified quality factor.
- **Reconstruction:** Quantized blocks are dequantized and subjected to Inverse DCT to reconstruct the original image.
- **Metrics Calculation:** Calculates MSE, PSNR, and provides a rough estimate of the compression ratio based on non-zero frequency coefficients.

## Outputs
- **Result Display:** A matplotlib window will show the original image, reconstructed image, and the amplified difference map. It also saves this result as `compression_result.png`.
- **Comparison Output:** If `--compare` is used, a detailed plot comparing different quality levels is saved as `quality_comparison.png`.
