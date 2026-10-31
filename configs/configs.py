# 18 tổ hợp = 3 LR x 3 WEIGHT_DECAY x 2 DROPOUT (full grid), ID đã xáo trộn (seed 42).
# Baseline hiện tại: LR=1e-3, WEIGHT_DECAY=0, DROPOUT=0.3. KHÔNG sửa file này giữa chừng thí nghiệm.

EXPERIMENT_CONFIGS = {
    1: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0, "DROPOUT": 0.5},
    2: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0, "DROPOUT": 0.3},  # <-- baseline
    3: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.3},
    4: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.5},
    5: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.5},
    6: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0, "DROPOUT": 0.5},
    7: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.3},
    8: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0, "DROPOUT": 0.3},
    9: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.3},
    10: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.5},
    11: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0, "DROPOUT": 0.5},
    12: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.5},
    13: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.3},
    14: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.3},
    15: {"BATCH_SIZE": 64, "LR": 0.001, "WEIGHT_DECAY": 0.001, "DROPOUT": 0.5},
    16: {"BATCH_SIZE": 64, "LR": 0.0003, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.3},
    17: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0, "DROPOUT": 0.3},
    18: {"BATCH_SIZE": 64, "LR": 0.0001, "WEIGHT_DECAY": 0.0001, "DROPOUT": 0.5},
}

# Chia việc (4+4+4+3+3):
# Kiet: 1-4 | Trang: 5-8 | Phuc: 9-12 | Dat: 13-15 | Tu: 16-18
