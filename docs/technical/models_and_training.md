# Input pipeline, models and training

**Code:** `src/drone_rf/{preprocessing,dataset,models,train,baselines}.py`, configs in `configs/`
**Tests:** `tests/test_dataset.py`, `tests/test_models.py`, `tests/test_train_smoke.py`
**Owner:** workstreams B and C (unassigned)

## Purpose

Load spectrograms as tensors, define the two required networks and the reference baselines, and train
them reproducibly while keeping the test set out of every decision.

## Inputs and outputs

| | Format | Example |
|---|---|---|
| Input | YAML config | `configs/binary_cnn.yaml` |
| Input | Splits file + image folder | `data/manifest/splits.csv`, `data/selected/` |
| Output | Run directory | `runs/a1_scratch_cnn/` with `config.yaml`, `meta.json`, `epochs.jsonl`, `best.pt`, `metrics_val.json`, `predictions_val.csv` |

## How it works

**Image to tensor.** PNG → 8-bit grayscale → resized to 224×224 (bilinear with antialiasing) →
float32 `[1, 224, 224]` in [0, 1]. Resized images are held in RAM so each PNG is decoded once.

**Normalization lives inside the model.** The scratch CNN subtracts the training set's pixel mean and
divides by its standard deviation (computed from training images only and stored in the checkpoint).
The ResNet18 wrapper copies the single channel three times and applies ImageNet's mean and standard
deviation, which is what its pretrained weights expect.

**Model A, `ScratchCNN`** (output of `python -m drone_rf.cli model-summary`):

```
input            (1, 1, 224, 224)
stage1           (1, 32, 112, 112)     Conv3x3 -> BatchNorm -> ReLU -> MaxPool2
stage2           (1, 64, 56, 56)
stage3           (1, 128, 28, 28)
stage4           (1, 256, 14, 14)
global_avg_pool  (1, 256)
logits           (1, 2)
parameters: 388,834 (all trainable, random initialization)
```

**Model B, `ResNet18Classifier`:** torchvision ResNet18, final layer replaced by a 2-output layer.
11,177,538 parameters; with the backbone frozen only 1,026 (the new head) are trainable.

**Training (`train.train_run`).**
1. Set seeds; load the splits file; assert it is group-safe; record its hash.
2. Build train and validation datasets from the rows whose images are on disk, filtered by the SNR
   levels in the config.
3. Optional stage 1 (`head_epochs > 0`, ResNet only): backbone frozen, BatchNorm statistics fixed,
   train the head.
4. Main stage: all parameters trainable. AdamW, cross-entropy loss, optional cosine schedule.
5. After every epoch: validation loss and metrics, one JSON line in `epochs.jsonl`.
6. Keep the checkpoint with the best validation macro-F1 (ties: lower validation loss). Stop a stage
   after `patience` epochs without improvement.
7. Reload the best checkpoint, write validation metrics and predictions, write `meta.json`.

**Baselines (`baselines.run_baselines`, run A0).** `majority` predicts the most common training
class. `brightness` is logistic regression on the mean pixel value. `energy` is logistic regression
on 39 numbers: seven global intensity statistics plus 16-bin mean profiles along each image axis. All
are fitted on the training split only.

**Augmentations (`preprocessing.AugmentConfig`), off by default:** small brightness offset (gain),
one blanked band per axis, small circular shift, and pixel-domain Gaussian noise. No rotations or
flips. Pixel noise is not the same as the dataset's RF noise and is named accordingly.

## How to run it

```
python -m drone_rf.cli baselines --config configs/binary_cnn.yaml --split val
python -m drone_rf.cli train --config configs/binary_cnn.yaml
python -m drone_rf.cli train --config configs/resnet18.yaml
python -m drone_rf.cli train --config configs/noise_robustness.yaml
```

| Config | Run | What it is |
|---|---|---|
| `binary_cnn.yaml` | A1 | Scratch CNN, SNR 10–20 dB, no augmentation |
| `binary_cnn_aug.yaml` | A2 | A1 plus one augmentation bundle |
| `resnet18.yaml` | B1 | Pretrained ResNet18, same data as A1 |
| `noise_robustness.yaml` | C1 | Scratch CNN trained on all SNR levels |
| `noise_robustness_resnet18.yaml` | C1b | Pretrained ResNet18 trained on all SNR levels |
| `smoke.yaml` | — | SYNTHETIC fixture run |

## Design decisions

| Decision | Alternatives considered | Why we chose this |
|---|---|---|
| 224×224 input | 640×640 native; 256; 320 | Training is CPU-only here; 224 is ResNet18's standard size. Risk: bursts only a few pixels wide at 640 get fainter. Larger input is a planned ablation |
| Four conv stages, 389k parameters | Deeper or wider network | Small enough to train on CPU in minutes, large enough to have a 14×14 map before pooling |
| Global average pooling | Flatten + large dense layer | Far fewer parameters; a burst can be anywhere in the image |
| Two logits + cross-entropy | One logit + BCE | Same thing for two classes; matches the ResNet head and torchvision conventions |
| Grayscale replicated to 3 channels for ResNet | Replace the first conv with a 1-channel one | Keeps every pretrained weight intact |
| Head-only stage, then full fine-tune at lower rate | Fine-tune everything from the start | The new head starts random; training it first avoids large gradients wrecking pretrained features |
| Select by validation macro-F1 | Accuracy; loss | Classes are imbalanced (about 2.2 : 1); macro-F1 weights both classes equally |
| No class weighting by default | Weighted loss | Imbalance is moderate; available as `class_weights: true` if recall on background suffers |
| Normalization stored in the model | Separate transform object | A checkpoint cannot be used with the wrong statistics |
| `require_files: true` | Fail if any image is missing | Lets the same frozen split work with a partial image download |

## Failure modes and how they are handled

| Failure | Handling |
|---|---|
| No images found for a split | Clear error naming `image_dir` and `splits_csv` |
| Splits file edited after training | `evaluate` compares hashes and refuses |
| Pretrained weights cannot be downloaded | torchvision raises; the run is blocked. A random ResNet18 is logged as `none (random init)`, never as pretrained |
| Quarantined row passed to the dataset | `ValueError` |

## Tests

- Tensor shape, dtype, range and finiteness; deterministic validation images.
- Labels follow the manifest column, not the file name.
- Normalization statistics are unchanged when validation/test rows are removed.
- Both models output `[batch, 2]`; gradients reach the first layer; freezing leaves exactly the head trainable and stops gradients to the backbone.
- A tiny CNN memorises 16 fixture images (pipeline sanity).
- Loss falls on the learnable fixture set; checkpoint reload gives identical predictions.
- Training writes no test-set file; a random ResNet18 is not logged as pretrained.
- `pytest -m integration` downloads the real ImageNet weights and checks they differ from random.

## Known limitations

- CPU-only on the development machine; large sweeps are impractical there.
- Images are preloaded into RAM: about 50 KB per image at 224×224.
- The comparison "scratch CNN vs pretrained ResNet18" changes architecture and initialization
  together. A ResNet18 with random weights (`pretrained: false`) would separate the two.
