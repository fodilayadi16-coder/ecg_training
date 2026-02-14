import os

# ----------------------------------------------- Function to get all records -----------------------------------------------

def get_all_records(base_path="data/raw/beat"):
    records = []                                                        # it stores a list of strings, where each string is the full file path to an ECG record (e.g., "data/raw/mitdb/100")
    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)                 # Constructs the full path to the current eg."data/raw/mitdb"
        if not os.path.isdir(dataset_path):
            continue

        for file in os.listdir(dataset_path):
            if file.endswith(".hea"):
                record_name = file.replace(".hea","")                   # removes the .hea extension to get base record name "100" instead "100.hea"
                records.append(os.path.join(dataset_path, record_name)) # .eg "data/raw/mitdb/100"

    return records