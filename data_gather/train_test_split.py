import fire
from pathlib import Path
import hashlib
import shutil

def file_normalized_hash(file_name):
    hash_object = hashlib.sha256(file_name.encode())
    hash_hex = hash_object.hexdigest()
    hash_int = int(hash_hex[:8], 16)
    max_hash_value = 2**32 - 1
    normalized_value = hash_int / max_hash_value
    return normalized_value

def main(data_path = "code", test_ratio=0.05):
    test_path = Path(data_path + "_test")
    train_path = Path(data_path + "_train")
    test_path.mkdir(exist_ok=True, parents=True)
    train_path.mkdir(exist_ok=True, parents=True)
    hash_modulo = 1 / test_ratio
    data_path = Path(data_path)
    selected_for_test = 0
    selected_for_train = 0
    for file_path in data_path.iterdir():
        hash = file_normalized_hash(str(file_path))
        if hash > test_ratio:
            shutil.copy(file_path, train_path)
            selected_for_train += 1
        else:
            shutil.copy(file_path, test_path)
            selected_for_test += 1
    print(f"Total selected for test: {selected_for_train}")
    print(f"Total selected for test: {selected_for_test}")
            

if __name__ == "__main__":
    fire.Fire(main)