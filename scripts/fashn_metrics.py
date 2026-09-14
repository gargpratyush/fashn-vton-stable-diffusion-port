"""Pixel and timing metrics shared by experiments and publications."""
import statistics

import numpy as np


def distribution(values):
    values = [float(value) for value in values]
    if not values or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty timing samples")
    mean = statistics.mean(values)
    deviation = statistics.stdev(values) if len(values) > 1 else None
    return {"n": len(values), "values": values, "mean": mean, "min": min(values), "max": max(values),
            "sample_stddev": deviation,
            "coefficient_of_variation": deviation / mean if deviation is not None and mean else None}


def pixel_metrics(expected, actual):
    if expected.dtype != np.uint8 or actual.dtype != np.uint8 or expected.shape != actual.shape or not expected.size:
        raise ValueError("Expected matching nonempty uint8 images")
    delta = actual.astype(np.float64) - expected
    mse = float(np.mean(delta * delta))
    return {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(delta))),
            "max_abs": float(np.max(np.abs(delta))), "changed_channels": int(np.count_nonzero(delta)),
            "signed_error_mean": float(np.mean(delta)), "error_population_variance": float(np.var(delta)),
            "psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None, "exact": mse == 0}


def worst_region(expected, actual, size=64):
    if (expected.shape != actual.shape or expected.ndim != 3 or expected.shape[2] != 3
            or not expected.size or size < 1):
        raise ValueError("Expected matching nonempty RGB images and positive region size")
    height, width = expected.shape[:2]
    size = min(size, height, width)
    best = None
    for y in sorted(set(range(0, height - size + 1, size // 2 or 1)) | {height - size}):
        for x in sorted(set(range(0, width - size + 1, size // 2 or 1)) | {width - size}):
            delta = actual[y:y + size, x:x + size].astype(np.float64) - expected[y:y + size, x:x + size]
            error = float(np.mean(delta * delta))
            if best is None or error > best["mse"]:
                best = {"x": x, "y": y, "width": size, "height": size, "mse": error}
    return best
