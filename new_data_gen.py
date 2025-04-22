import pandas as pd
import numpy as np
import os

# --- Configuration ---
ANNOTATION_FILE = 'annotation.csv'
CSI_NPY_PATH = 'wifi_csi\\amp'
CSI_FILE_COLUMN = 'label'
USER_COUNT_COLUMN = 'number_of_users'
ACTIVITY_COLUMNS = [
    'user_1_activity', 'user_2_activity', 'user_3_activity',
    'user_4_activity', 'user_5_activity', 'user_6_activity'
]
OUTPUT_CSV_FILE = 'user1_combined_activity_data.csv'
MAX_CSI_FEATURES = 1000

combined_data = []

# 1. Load Annotations
annotations_df = pd.read_csv(ANNOTATION_FILE)

# 2. Filter for number_of_users == 1
filtered_df = annotations_df[annotations_df[USER_COUNT_COLUMN] == 1]

# 3. Process Each Entry
for index, row in filtered_df.iterrows():
    csi_filename = row[CSI_FILE_COLUMN]
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

        # Combine non-empty activity columns into a single 'activity' field
        activity_values = [
            str(row[col]).strip() for col in ACTIVITY_COLUMNS
            if pd.notna(row.get(col)) and str(row[col]).strip() != ""
        ]
        activity = "; ".join(activity_values) if activity_values else None

        if activity is None:
            continue  # skip if no activity was found

        row_dict = {'activity': activity}
        for i, feature in enumerate(flattened_csi):
            row_dict[f'csi_feature_{i+1}'] = feature

        combined_data.append(row_dict)

# 4. Create DataFrame and Save
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"Filtered and combined activity data saved to {OUTPUT_CSV_FILE}")
