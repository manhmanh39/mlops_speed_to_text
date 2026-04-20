import json
import logging
import os

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cấu hình đường dẫn (Hải chỉnh lại cho khớp với folder trên máy nhé)
EXTRACTED_DIR = "data_train_70.6"
WAV_DIR = os.path.join(EXTRACTED_DIR, 'wavs')
META_CSV = os.path.join(EXTRACTED_DIR, 'metadata.csv')
OUTPUT_FILE = "models/baseline_distribution.json"


def main():
    if not os.path.exists(META_CSV):
        logger.error(f"Không tìm thấy file metadata tại: {META_CSV}")
        return

    # Đọc metadata
    meta = pd.read_csv(META_CSV, sep='|', header=None, names=['filename', 'transcript'])

    durations = []
    logger.info(f"Đang xử lý {len(meta)} file audio để lấy Baseline...")

    for filename in tqdm(meta['filename']):
        path = os.path.join(WAV_DIR, filename)
        if os.path.exists(path):
            try:
                # Chỉ load duration để tiết kiệm RAM và thời gian
                duration = librosa.get_duration(path=path)
                durations.append(float(duration))
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file {filename}: {e}")

    # Tính toán các thông số thống kê cơ bản
    baseline_data = {
        "durations": durations,
        "mean": float(np.mean(durations)),
        "std": float(np.std(durations)),
        "count": len(durations)
    }

    # Lưu file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(baseline_data, f, indent=4)

    logger.info(f"--- Đã trích xuất xong! Baseline lưu tại: {OUTPUT_FILE} ---")


if __name__ == "__main__":
    main()
