import fire
from pathlib import Path
import hashlib
import shutil

from data_gather.filtering import filter_registry

def filter_file(text):
    for name, filter in filter_registry.items():
        if not filter(text):
            return False
    return True

def file_normalized_hash(file_name):
    hash_object = hashlib.sha256(file_name.encode())
    hash_hex = hash_object.hexdigest()
    hash_int = int(hash_hex[:8], 16)
    max_hash_value = 2**32 - 1
    normalized_value = hash_int / max_hash_value
    return normalized_value

def main(data_path = "code", test_ratio=0.1):
    test_path = Path(data_path + "_test")
    train_path = Path(data_path + "_train")
    test_path.mkdir(exist_ok=True, parents=True)
    train_path.mkdir(exist_ok=True, parents=True)
    data_path = Path(data_path)
    selected_for_test = 0
    selected_for_train = 0
    filtered_out = 0
    for file_path in data_path.iterdir():
        passed_filtering = filter_file(open(file_path).read())
        if not passed_filtering:
            filtered_out += 1
            continue
        hash = file_normalized_hash(str(file_path))
        if hash > test_ratio:
            shutil.copy(file_path, train_path)
            selected_for_train += 1
        else:
            shutil.copy(file_path, test_path)
            selected_for_test += 1
    print(f"Total selected for test: {selected_for_train}")
    print(f"Total selected for test: {selected_for_test}")
    print(f"Total filtered out: {filtered_out}")
            

if __name__ == "__main__":
    fire.Fire(main)