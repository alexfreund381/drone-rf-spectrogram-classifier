| run | model | pretrained_weights | split | n_images | n_groups | accuracy | balanced_accuracy | macro_f1 | recall_drone | precision_drone | false_positive_rate | group_macro_f1 | best_epoch | wall_seconds | split_hash | macro_f1_ci95_low | macro_f1_ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0_brightness | baseline_brightness | n/a | val | 395 | 98 | 0.8253 | 0.7668 | 0.7867 | 0.9611 | 0.8072 | 0.4275 | 0.8503 |  |  | ddfcbce03785a3af | 0.7001 | 0.8663 |
| a0_energy | baseline_energy | n/a | val | 395 | 98 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |  |  | ddfcbce03785a3af | 1.0000 | 1.0000 |
| a0_majority | baseline_majority | n/a | val | 395 | 98 | 0.6506 | 0.5000 | 0.3942 | 1.0000 | 0.6506 | 1.0000 | 0.4096 |  |  | ddfcbce03785a3af | 0.3521 | 0.4299 |
| a1_scratch_cnn | scratch_cnn | none (random init) | val | 395 | 98 | 0.9975 | 0.9981 | 0.9972 | 0.9961 | 1.0000 | 0.0000 | 1.0000 | 13 | 2580.2722 | ddfcbce03785a3af | 0.9917 | 1.0000 |
| b1_resnet18_pretrained | resnet18 | ResNet18_Weights.IMAGENET1K_V1 | val | 395 | 98 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 5 | 1429.4208 | ddfcbce03785a3af | 1.0000 | 1.0000 |
