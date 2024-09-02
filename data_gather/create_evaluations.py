from pathlib import Path
import json
import random

import fire

def main(test_folder: Path, evaluations_per_file: int = 1, output_file_path: Path = Path("evaluation_data.json")):
    if isinstance(test_folder, str):
        test_folder = Path(test_folder)
    evaluations = []
    for code_file in test_folder.iterdir():
        contents = open(code_file).read()
        for _ in range(evaluations_per_file):
            cursor_position = random.randrange(len(contents))
            code_before = contents[:cursor_position]
            newline = contents.find("\n", cursor_position)
            code_to_predict = contents[cursor_position:newline]
            code_after = contents[newline:]
            evaluations.append({
                "file_path": str(code_file),
                "full_code": contents,
                "cursor_position": cursor_position,
                "code_before": code_before,
                "code_after": code_after,
                "expected_code": code_to_predict
            })
        
    with open(output_file_path, 'w') as output_file:
        json.dump(evaluations, output_file, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
    