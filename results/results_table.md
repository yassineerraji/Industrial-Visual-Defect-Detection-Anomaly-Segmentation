| Model | Supervision | Evaluation | Image AUROC | Image AP | F1 | FPR | Pixel AUROC | Dice | Latency CPU (ms) | Latency MPS (ms) | Checkpoint (MB) | Peak RSS CPU (MB) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | Normal only | single model on full split | 0.432 | 0.748 | 0.367 | 0.333 | 0.678 | 0.065 | 592.3 | 40.0 | 7.5 | 1664 |
| PatchCore | Normal only | single model on full split | 0.713 | 0.917 | 0.583 | 0.000 | 0.856 | 0.338 | 367.3 | 54.8 | 19.8 | 1703 |
| U-Net | Pixel labels | pooled out-of-fold predictions (grouped CV) | 0.852 | 0.953 | 0.879 | 0.292 | 0.593 | 0.208 | 427.4 | 55.6 | 57.8 | 1615 |
