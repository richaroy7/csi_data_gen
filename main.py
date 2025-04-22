import pandas as pd
import numpy as np
import os

# --- Configuration ---
ANNOTATION_FILE = 'annotation.csv'
CSI_NPY_PATH = 'wifi_csi\\amp'
CSI_FILE_COLUMN = 'label'
USER_COUNT_COLUMN = 'number_of_users'
OUTPUT_CSV_FILE = 'full_combined_data.csv'
MAX_CSI_FEATURES = 3000

combined_data = []

# 1. Load Annotations
annotations_df = pd.read_csv(ANNOTATION_FILE)

# 2. Process Entire Dataset
for index, row in annotations_df.iterrows():
    csi_filename = row[CSI_FILE_COLUMN]
    num_users = row[USER_COUNT_COLUMN]
    presence = 1 if num_users > 0 else 0

    npy_filepath = os.path.join(CSI_NPY_PATH, f"{csi_filename}.npy")

    if not os.path.exists(npy_filepath):
        print(f"Warning: CSI file not found at {npy_filepath} for annotation at index {index}")
        continue

    try:
        csi_data = np.load(npy_filepath)
    except Exception as e:
        print(f"Error loading .npy file {npy_filepath} for annotation at index {index}: {e}")
        continue

    if csi_data is not None:
        flattened_csi = csi_data.flatten()[:MAX_CSI_FEATURES]

        row_dict = row.to_dict()
        row_dict['presence'] = presence
        for i, feature in enumerate(flattened_csi):
            row_dict[f'csi_feature_{i+1}'] = feature

        combined_data.append(row_dict)

# 3. Create DataFrame and Save
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"Combined full dataset saved to {OUTPUT_CSV_FILE}")
