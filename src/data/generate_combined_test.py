import json

names = ['python_cobol', 'java_fortran', 'js_pascal']

folders = ['codeNet', 'rosetta']

for p in names:
    combined = []
    for fd in folders:
        with open(f'src/data/{fd}/{p}_test.json') as f:
            combined.extend(json.load(f))

    with open(f"src/data/combined/{p}_test.json", "w") as f:
        json.dump(combined, f, indent=2)
