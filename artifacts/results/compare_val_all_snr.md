| run | model | pretrained_weights | split | n_images | n_groups | accuracy | balanced_accuracy | macro_f1 | recall_drone | precision_drone | false_positive_rate | group_macro_f1 | best_epoch | wall_seconds | split_hash | macro_f1_ci95_low | macro_f1_ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0_brightness | baseline_brightness | n/a | val | 1671 | 98 | 0.4800 | 0.5546 | 0.4800 | 0.3517 | 0.7585 | 0.2424 | 0.2547 |  |  | ddfcbce03785a3af | 0.4260 | 0.5234 |
| a0_energy | baseline_energy | n/a | val | 1671 | 98 | 0.6709 | 0.7594 | 0.6704 | 0.5188 | 1.0000 | 0.0000 | 0.6821 |  |  | ddfcbce03785a3af | 0.6087 | 0.7166 |
| a0_majority | baseline_majority | n/a | val | 1671 | 98 | 0.6840 | 0.5000 | 0.4062 | 1.0000 | 0.6840 | 1.0000 | 0.4096 |  |  | ddfcbce03785a3af | 0.3711 | 0.4362 |
| a1_scratch_cnn | scratch_cnn | none (random init) | val | 1671 | 98 | 0.7127 | 0.7727 | 0.7084 | 0.6098 | 0.9535 | 0.0644 | 0.8581 | 13 | 2580.2722 | ddfcbce03785a3af | 0.6488 | 0.7487 |
| b1_resnet18_pretrained | resnet18 | ResNet18_Weights.IMAGENET1K_V1 | val | 1671 | 98 | 0.6750 | 0.7390 | 0.6719 | 0.5652 | 0.9335 | 0.0871 | 0.7500 | 5 | 1429.4208 | ddfcbce03785a3af | 0.6135 | 0.7117 |
