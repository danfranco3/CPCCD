import json

paths = [
    "src/data/codeNet/python_cobol_test.json",
    "src/data/codeNet/java_fortran_test.json",
    "src/data/codeNet/js_pascal_test.json",
    "src/data/rosetta/python_cobol_test.json",
    "src/data/rosetta/java_fortran_test.json",
    "src/data/rosetta/js_pascal_test.json"
]

combined = []
for path in paths:
    with open(path) as f:
        combined.extend(json.load(f))

with open("src/data/combined_test.json", "w") as f:
    json.dump(combined, f, indent=2)
