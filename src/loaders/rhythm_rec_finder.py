import os

# ----------------------------------------------- Function to get all records from AFDB -----------------------------------------------

def get_all_records(base_path="data/raw/rhythm"):
    records = []
    for file in os.listdir(base_path):
        if file.endswith(".hea"):
            records.append(os.path.join(base_path, file.replace(".hea", "")))
    return records
