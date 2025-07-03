import json

paths = [
    "src/data/codeNet/python_cobol_train.json",
    "src/data/codeNet/java_fortran_train.json",
    "src/data/codeNet/js_pascal_train.json",
    "src/data/rosetta/python_cobol_train.json",
    "src/data/rosetta/java_fortran_train.json",
    "src/data/rosetta/js_pascal_train.json",
    "src/data/codeNet/ruby_go_train.json"
]

combined = []
for path in paths:
    with open(path) as f:
        combined.extend(json.load(f))

with open("src/data/combined_train.json", "w") as f:
    json.dump(combined, f, indent=2)
