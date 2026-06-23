import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import os

import argparse

# ──────────────────────────────────────────
# CONFIGURATION — Default values
# ──────────────────────────────────────────
DEFAULT_BLOCK_SIZE   = 8             # standard JPEG block size
DEFAULT_QUALITY      = 50            # 1 (max compress) to 100 (min compress)

# ──────────────────────────────────────────
# QUANTIZATION MATRIX (standard JPEG luminance)
# ──────────────────────────────────────────
QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def get_scaled_quant_matrix(quality: int) -> np.ndarray:
    """Scale the quantization matrix based on quality factor (1–100)."""
    if quality <= 0:   quality = 1
    if quality > 100:  quality = 100
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    q = np.floor((QUANT_MATRIX * scale + 50) / 100)
    q = np.clip(q, 1, 255)
    return q.astype(np.float32)

# ──────────────────────────────────────────
# CORE DCT FUNCTIONS
# ──────────────────────────────────────────

def apply_2d_dct(block: np.ndarray) -> np.ndarray:
    """Apply 2D DCT to an 8×8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def apply_2d_idct(block: np.ndarray) -> np.ndarray:
    """Apply 2D Inverse DCT to an 8×8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize(dct_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients — this is where compression happens."""
    return np.round(dct_block / q_matrix)

def dequantize(q_block: np.ndarray, q_matrix: np.ndarray) -> np.ndarray:
    """Reverse quantization to recover approximate DCT coefficients."""
    return q_block * q_matrix

# ──────────────────────────────────────────
# COMPRESSION PIPELINE
# ──────────────────────────────────────────

def compress_image(image: np.ndarray, quality: int) -> tuple:
    """
    Full compression pipeline (VECTORIZED for speed):
      1. Pad image to multiple of DEFAULT_BLOCK_SIZE
      2. Reshape into 4D tensor of 8×8 blocks
      3. Apply 2D DCT, Quantize, reshape back
    Returns quantized coefficients and padding info.
    """
    q_matrix = get_scaled_quant_matrix(quality)
    h, w = image.shape

    # Pad so dimensions are multiples of DEFAULT_BLOCK_SIZE
    pad_h = (DEFAULT_BLOCK_SIZE - h % DEFAULT_BLOCK_SIZE) % DEFAULT_BLOCK_SIZE
    pad_w = (DEFAULT_BLOCK_SIZE - w % DEFAULT_BLOCK_SIZE) % DEFAULT_BLOCK_SIZE
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge').astype(np.float32)

    ph, pw = padded.shape
    
    # Reshape into 4D array: (num_blocks_h, block_h, num_blocks_w, block_w)
    blocks = padded.reshape(ph // DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
                            pw // DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE)
    
    # Subtract 128 for DC offset
    blocks = blocks - 128.0
    
    # Apply 2D DCT to all blocks at once using scipy (applies DCT on axis 1 and 3)
    blocks = dct(blocks, axis=1, norm='ortho')
    blocks = dct(blocks, axis=3, norm='ortho')
    
    # Quantize all blocks at once
    blocks = np.round(blocks / q_matrix[None, :, None, :])
    
    # Reshape back to 2D
    compressed = blocks.reshape(ph, pw)
    
    return compressed, q_matrix, (pad_h, pad_w)

def decompress_image(compressed: np.ndarray, q_matrix: np.ndarray,
                     original_shape: tuple, padding: tuple) -> np.ndarray:
    """
    Full decompression pipeline (VECTORIZED for speed):
      Reshape into 4D tensor, Dequantize, apply 2D IDCT, reshape back, remove padding
    """
    ph, pw = compressed.shape
    
    # Reshape into 4D array: (num_blocks_h, block_h, num_blocks_w, block_w)
    blocks = compressed.reshape(ph // DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE,
                                pw // DEFAULT_BLOCK_SIZE, DEFAULT_BLOCK_SIZE)
    
    # Dequantize all blocks at once
    blocks = blocks * q_matrix[None, :, None, :]
    
    # Apply 2D IDCT to all blocks at once
    blocks = idct(blocks, axis=1, norm='ortho')
    blocks = idct(blocks, axis=3, norm='ortho')
    
    # Add 128 for DC offset
    blocks = blocks + 128.0
    
    # Reshape back to 2D
    reconstructed = blocks.reshape(ph, pw)

    # Remove padding and clip pixel values
    h, w = original_shape
    pad_h, pad_w = padding
    out_h = ph - pad_h if pad_h > 0 else ph
    out_w = pw - pad_w if pad_w > 0 else pw
    reconstructed = reconstructed[:out_h, :out_w]
    return np.clip(reconstructed, 0, 255).astype(np.uint8)

# ──────────────────────────────────────────
# QUALITY METRICS
# ──────────────────────────────────────────

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    orig_f = original.astype(np.float64)
    rec_f  = reconstructed.astype(np.float64)

    mse_val  = np.mean((orig_f - rec_f) ** 2)
    psnr_val = psnr(original, reconstructed, data_range=255)

    # Estimate compression ratio using non-zero DCT coefficients
    # (simplified; real ratio needs entropy coding)
    return {
        "MSE":  round(mse_val, 4),
        "PSNR": round(psnr_val, 4),
    }

def estimate_compression_ratio(compressed: np.ndarray, original: np.ndarray) -> float:
    """Rough estimate: ratio of non-zero coefficients vs total pixels."""
    nonzero    = np.count_nonzero(compressed)
    total_orig = original.size
    if nonzero == 0:
        return float('inf')
    return round(total_orig / nonzero, 2)

# ──────────────────────────────────────────
# DISPLAY RESULTS
# ──────────────────────────────────────────

def show_results(original: np.ndarray, reconstructed: np.ndarray,
                 metrics: dict, compression_ratio: float, quality: int):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"DCT Image Compression  |  Quality={quality}  |  "
                 f"CR≈{compression_ratio}x  |  "
                 f"PSNR={metrics['PSNR']} dB  |  MSE={metrics['MSE']}",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f"Reconstructed (Q={quality})")
    axes[1].axis('off')

    diff = np.abs(original.astype(np.int16) - reconstructed.astype(np.int16))
    axes[2].imshow(diff, cmap='hot', vmin=0, vmax=50)
    axes[2].set_title("Difference Map (amplified)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("compression_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Result saved as compression_result.png")

# ──────────────────────────────────────────
# MULTI-QUALITY COMPARISON
# ──────────────────────────────────────────

def compare_qualities(image: np.ndarray, qualities=[10, 30, 50, 75, 95]):
    """Show the effect of different quality levels side by side."""
    fig, axes = plt.subplots(2, len(qualities), figsize=(4*len(qualities), 8))

    for idx, q in enumerate(qualities):
        comp, qmat, pad = compress_image(image, q)
        recon = decompress_image(comp, qmat, image.shape, pad)
        m = compute_metrics(image, recon)
        cr = estimate_compression_ratio(comp, image)

        axes[0][idx].imshow(recon, cmap='gray', vmin=0, vmax=255)
        axes[0][idx].set_title(f"Q={q}\nPSNR={m['PSNR']}dB", fontsize=9)
        axes[0][idx].axis('off')

        diff = np.abs(image.astype(np.int16) - recon.astype(np.int16))
        axes[1][idx].imshow(diff, cmap='hot', vmin=0, vmax=50)
        axes[1][idx].set_title(f"CR≈{cr}x\nMSE={m['MSE']}", fontsize=9)
        axes[1][idx].axis('off')

    fig.suptitle("Comparison Across Quality Levels\n(Top: Reconstructed | Bottom: Error Map)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("quality_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved as quality_comparison.png")

# ──────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Compression using Discrete Cosine Transform (DCT)")
    parser.add_argument("-i", "--image", type=str, default=None, help="Path to the input image")
    parser.add_argument("-q", "--quality", type=int, default=DEFAULT_QUALITY, help="Compression quality (1-100)")
    parser.add_argument("--compare", action="store_true", help="Generate multi-quality comparison")
    
    args = parser.parse_args()

    image_path = args.image
    quality = args.quality

    if image_path is None:
        print("No image provided. Launching the Web UI...")
        import subprocess
        import sys
        import webbrowser
        import time
        
        # Start the Flask app
        subprocess.Popen([sys.executable, "app.py"])
        
        # Wait a moment for server to start, then open browser
        time.sleep(1)
        webbrowser.open("http://127.0.0.1:5000")
        return

    # -- Load image --
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        print("Please provide a valid image path using -i or --image.")
        return

    # Convert to grayscale as per requirement
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("[ERROR] Could not read image. Check the file path and format.")
        return

    print(f"Image loaded: {image_path}  Shape={image.shape}  Size={image.size} pixels")

    # -- Compress --
    print(f"\nCompressing at quality={quality} ...")
    compressed, q_matrix, padding = compress_image(image, quality)

    # -- Decompress --
    print("Decompressing ...")
    reconstructed = decompress_image(compressed, q_matrix, image.shape, padding)

    # -- Metrics --
    metrics = compute_metrics(image, reconstructed)
    cr      = estimate_compression_ratio(compressed, image)

    print("\n-- RESULTS --------------------------")
    print(f"  Quality Factor    : {quality}")
    print(f"  MSE               : {metrics['MSE']}")
    print(f"  PSNR              : {metrics['PSNR']} dB")
    print(f"  Est. Comp. Ratio  : {cr}x")
    print("-------------------------------------\n")

    # -- Display --
    show_results(image, reconstructed, metrics, cr, quality)

    # -- Multi-quality comparison --
    if args.compare:
        print("Generating multi-quality comparison ...")
        compare_qualities(image)

if __name__ == "__main__":
    main()