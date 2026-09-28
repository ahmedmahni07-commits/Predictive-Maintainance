# Predictive-Maintainance
Machine learning pipeline for predictive maintenance and fault diagnosis in three-phase induction motors using vibration spectrum analysis.

## KNN vibration anomaly detection

`vibration_knn_anomaly.py` extracts nine time-domain and nine Welch-PSD
frequency-domain features for each vibration sample, scales the features, and
plots KNN anomaly scores against their timestamps. The vibration `.npy` file is
memory-mapped so the full raw array is not loaded into memory.

Install the required Python packages with `pip install numpy scipy scikit-learn matplotlib`.

```bash
python vibration_knn_anomaly.py \
  --timestamps timestamps_array.npy \
  --data raw_vibration_data_5246_320000_2.npy \
  --output knn_temporal_anomalies.png \
  --features-output extracted_18_features_fwd.npy
```

The input data must have shape `(samples, points, channels)`, and timestamps
must be a one-dimensional `.npy` array with one value per sample. Defaults are
32 kHz sampling frequency, channel 0, six queried neighbors (including each
sample itself), and an 80th-percentile score cutoff. Use `--help` to see
available options; `--show` opens the saved plot interactively.
