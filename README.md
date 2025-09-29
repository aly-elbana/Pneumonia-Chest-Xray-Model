## Pneumonia Chest X‑Ray Classification

A PyTorch project that classifies chest X‑ray images into three classes: `Bacteria_PNEUMONIA`, `NORMAL`, and `Virus_PNEUMONIA`. The default architecture uses DenseNet‑121 as a frozen feature extractor with a custom classifier head. Alternative classifier head (lightweight CNN) is also available.

### Overview

- Data directory is `chest_xray/` with `train/`, `val/`, and `test/` splits; each split contains `Bacteria_PNEUMONIA`, `NORMAL`, and `Virus_PNEUMONIA` folders.
- Two training scripts are provided: `train.py` (baseline) and `train2.py` (adds LR scheduler and optional early stopping).
- Evaluation across saved checkpoints is in `model_test.py`. Single‑image inference from `test_images/` is in `model_in_RL.py`.
- Saved models are stored under `saved_models_final/` and `saved_models_cnn/`.

### Dataset

- Recommended dataset: Chest X‑Ray Images (Pneumonia).
  - Kaggle: `https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia`
- After download, place it under `chest_xray/` with the following structure:

```
chest_xray/
  train/
    Bacteria_PNEUMONIA/
    NORMAL/
    Virus_PNEUMONIA/
  val/
    Bacteria_PNEUMONIA/
    NORMAL/
    Virus_PNEUMONIA/
  test/
    Bacteria_PNEUMONIA/
    NORMAL/
    Virus_PNEUMONIA/
```

If your original dataset only has `PNEUMONIA` as a single label, see `notebooks/main.ipynb`, which contains a cell to reorganize files into `Virus_` and `Bacteria_` based on filenames.

### Installation

Create a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- On Windows without CUDA, the code will default to CPU. For GPU acceleration, install the CUDA‑enabled PyTorch builds from the official guide.
- If you experience slow DataLoader performance on Windows, set `num_workers=0` in loaders.

### Training

Baseline script:

```bash
python train.py
```

Alternative with LR scheduler and early stopping:

```bash
python train2.py
```

Both scripts save the best checkpoint automatically as `best_model_epoch*_acc*_loss*_{timestamp}.pth` into either `saved_models_final/` or `saved_models_cnn/` depending on the path used.

### Testing on the test split

```bash
python model_test.py
```

This scans all `.pth` files in `saved_models_final/`, evaluates on `./chest_xray/test`, and reports the best.

CLI options:

```bash
python model_test.py --models_dir ./saved_models_final --data_root ./chest_xray/test --batch_size 32 --num_workers 0
```

### Single‑image inference

Place images under `test_images/` and run:

```bash
python model_in_RL.py
```

Predicted class names will be printed for each image file.

CLI options:

```bash
python model_in_RL.py --images_dir ./test_images --model_path ./saved_models_final/last_model_checkpoint.pth
```

### Project structure

```
.
├── chest_xray/
├── notebooks/
│   └── main.ipynb             # data organization & quick exploration
├── model.py                   # DenseNet feature extractor + fully‑connected classifier
├── model_cnn.py               # DenseNet feature extractor + lightweight CNN classifier
├── model_in_RL.py             # single‑image inference from test_images
├── model_test.py              # evaluate saved checkpoints on test split
├── train.py                   # baseline training (saves best + last)
├── train2.py                  # training with LR scheduler and early stopping
├── saved_models_final/        # saved checkpoints (best/last)
├── saved_models_cnn/
├── test_images/
├── requirements.txt
└── README.md
```

### Practical tips

- Start with `AdamW` at `lr=1e-3`, then reduce LR when validation plateaus.
- Tune the composite score weight `lambda_loss` to balance accuracy vs. loss when selecting best checkpoints.
- Consider class weighting in `CrossEntropyLoss` if classes are imbalanced.

### License

For research and educational use. Please review and respect the dataset license on Kaggle before redistribution.
