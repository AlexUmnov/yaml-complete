
import base64
from pathlib import Path
import datetime

import fire
import github3
from tqdm.auto import tqdm


def get_result_code(result, output_folder):
    b64_content = result.repository.blob(result.sha).content
    decoded = base64.b64decode(b64_content).decode()
    new_name = f"{result.repository.full_name.replace('/', '_')}_{result.name}"
    with open(output_folder / new_name, 'w') as output_file:
        output_file.write(decoded)

def main(
    search_query="language:yaml",
    limit_results=-1,
    output_folder="code",
    size_increment=100,
    size_increments_count=20
):
    g = github3.login(token=open(".github-token").read())
    for size in range(0, size_increment*size_increments_count, size_increment):
        results = g.search_code(
            query=search_query + f" size:{size}..{size + size_increment}",
            number=limit_results
        )
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)
        for result in tqdm(results):
            if results.ratelimit_remaining <= 0:
                print("Sleeping for 60 seconds to regain rate limit")
                sleep(60)
            get_result_code(result, output_folder)


if __name__ == "__main__":
    fire.Fire(main)