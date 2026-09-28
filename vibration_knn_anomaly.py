#!/usr/bin/env python3
"""Extract vibration features and detect anomalies with k-nearest neighbors."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract vibration features and plot KNN anomaly scores."
    )
    parser.add_argument("--timestamps", required=True, help="Path to a 1D timestamps .npy file")
    parser.add_argument("--data", required=True, help="Path to vibration data .npy file")
    parser.add_argument("--output", default="knn_temporal_anomalies.png", help="Output plot path")
    parser.add_argument(
        "--features-output",
        help="Optional .npy output path for the extracted feature matrix",
    )
    parser.add_argument("--fs", type=float, default=32000, help="Sampling frequency in Hz")
    parser.add_argument("--channel", type=int, default=0, help="Vibration channel to analyze")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=6,
        help="Neighbors to query, including the sample itself (default: 6)",
    )
    parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.8,
        help="Score quantile separating the normal baseline from anomalies",
    )
    parser.add_argument("--nperseg", type=int, default=4096, help="Welch segment length")
    parser.add_argument("--show", action="store_true", help="Display the plot after saving it")
    return parser.parse_args()


def extract_features(raw_data, fs, channel, nperseg):
    """Extract nine time- and nine frequency-domain features per sample."""
    n_samples = raw_data.shape[0]
    features = np.empty((n_samples, 18), dtype=np.float64)

    for index in range(n_samples):
        if index and index % 500 == 0:
            print(f"   Processed {index} / {n_samples} samples...")

        signal = np.asarray(raw_data[index, :, channel], dtype=np.float64)
        if not np.isfinite(signal).all():
            raise ValueError(f"Sample {index}, channel {channel} contains non-finite values")

        abs_signal = np.abs(signal)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        rms_val = np.sqrt(np.mean(signal**2))
        peak_to_peak = np.ptp(signal)
        peak_val = np.max(abs_signal)
        mean_abs = np.mean(abs_signal)

        impulse = peak_val / mean_abs if mean_abs > 0 else 0.0
        shape = rms_val / mean_abs if mean_abs > 0 else 0.0
        crest = peak_val / rms_val if rms_val > 0 else 0.0
        skewness = skew(signal) if std_val > 0 else 0.0
        kurt = kurtosis(signal, fisher=False) if std_val > 0 else 3.0
        time_features = [
            mean_val,
            std_val,
            rms_val,
            peak_to_peak,
            impulse,
            shape,
            crest,
            skewness,
            kurt,
        ]

        frequencies, psd = welch(
            signal,
            fs=fs,
            nperseg=min(nperseg, signal.size),
        )
        total_power = np.sum(psd)
        if total_power > 0:
            psd_norm = psd / total_power
            spectral_mean = np.sum(frequencies * psd_norm)
            spectral_std = np.sqrt(np.sum((frequencies - spectral_mean) ** 2 * psd_norm))
            if spectral_std > 0:
                spectral_skew = np.sum(
                    (frequencies - spectral_mean) ** 3 * psd_norm
                ) / spectral_std**3
                spectral_kurt = np.sum(
                    (frequencies - spectral_mean) ** 4 * psd_norm
                ) / spectral_std**4
            else:
                spectral_skew = 0.0
                spectral_kurt = 0.0
        else:
            spectral_mean = 0.0
            spectral_std = 0.0
            spectral_skew = 0.0
            spectral_kurt = 0.0

        peak_power = np.max(psd)
        freq_rms = np.sqrt(np.mean(psd**2))
        psd_mean = np.mean(psd)
        freq_features = [
            spectral_mean,
            spectral_std,
            spectral_skew,
            spectral_kurt,
            frequencies[np.argmax(psd)],
            peak_power,
            total_power,
            peak_power / freq_rms if freq_rms > 0 else 0.0,
            freq_rms / psd_mean if psd_mean > 0 else 0.0,
        ]
        features[index] = time_features + freq_features

    return features


def main():
    args = parse_args()
    if args.fs <= 0:
        raise ValueError("--fs must be greater than zero")
    if args.nperseg <= 0:
        raise ValueError("--nperseg must be greater than zero")
    if args.n_neighbors < 2:
        raise ValueError("--n-neighbors must be at least 2")
    if not 0 < args.threshold_quantile < 1:
        raise ValueError("--threshold-quantile must be between 0 and 1")

    print("1. Loading raw dataset...")
    timestamps = np.load(args.timestamps)
    raw_data = np.load(args.data, mmap_mode="r")
    if timestamps.ndim != 1:
        raise ValueError(f"Timestamps must be one-dimensional; got shape {timestamps.shape}")
    if raw_data.ndim != 3:
        raise ValueError(
            "Vibration data must have shape (samples, points, channels); "
            f"got shape {raw_data.shape}"
        )
    if raw_data.shape[0] != timestamps.size:
        raise ValueError(
            f"Timestamp count ({timestamps.size}) does not match sample count "
            f"({raw_data.shape[0]})"
        )
    if not 0 <= args.channel < raw_data.shape[2]:
        raise ValueError(
            f"--channel must be between 0 and {raw_data.shape[2] - 1}"
        )
    if raw_data.shape[0] < args.n_neighbors:
        raise ValueError(
            f"At least {args.n_neighbors} samples are required for "
            f"--n-neighbors={args.n_neighbors}"
        )
    if raw_data.shape[1] == 0:
        raise ValueError("Each vibration sample must contain at least one point")

    print(f"Data shape: {raw_data.shape}")
    print(f"Total samples to process: {raw_data.shape[0]}")

    print(f"\n2. Extracting features for channel {args.channel}...")
    features = extract_features(raw_data, args.fs, args.channel, args.nperseg)
    print(f"Feature extraction complete. Matrix shape: {features.shape}")
    if args.features_output:
        features_path = Path(args.features_output)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(features_path, features)
        print(f"Saved feature matrix to {args.features_output}")

    print("\n3. Scaling features and fitting KNN model...")
    scaled = StandardScaler().fit_transform(features)
    neighbors = NearestNeighbors(
        n_neighbors=args.n_neighbors,
        metric="euclidean",
        algorithm="auto",
    )
    neighbors.fit(scaled)
    distances, _ = neighbors.kneighbors(scaled)
    scores = distances[:, 1:].mean(axis=1)

    print("\n4. Calculating threshold and plotting...")
    threshold_index = min(
        int(args.threshold_quantile * scores.size),
        scores.size - 1,
    )
    threshold_score = np.sort(scores)[threshold_index]
    normal_mask = scores <= threshold_score
    anomaly_mask = scores > threshold_score
    print(f"{args.threshold_quantile:.0%} threshold score cutoff: {threshold_score:.4f}")

    plt.figure(figsize=(12, 6))
    plt.scatter(
        timestamps[normal_mask],
        scores[normal_mask],
        color="blue",
        alpha=0.5,
        label="Normal baseline (lower KNN scores)",
        s=10,
    )
    plt.scatter(
        timestamps[anomaly_mask],
        scores[anomaly_mask],
        color="red",
        alpha=0.5,
        label="Potential anomalies (higher KNN scores)",
        s=10,
    )
    plt.axhline(
        y=threshold_score,
        color="black",
        linestyle="--",
        label=f"{args.threshold_quantile:.0%} threshold ({threshold_score:.3f})",
    )
    plt.title(f"KNN Anomaly Scores Over Time (Channel {args.channel})")
    plt.xlabel("Timestamp")
    plt.ylabel("Mean Euclidean distance to nearest neighbors")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=30)
    plt.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    if args.show:
        plt.show()
    plt.close()
    print(f"Saved plot to {output_path}")
    print("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
