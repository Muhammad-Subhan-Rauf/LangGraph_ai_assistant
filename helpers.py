import os

def find_files(file_name, search_path="."):
    matches = []
    for root, _, files in os.walk(search_path):
        if file_name in files:
            matches.append(os.path.join(root, file_name))
    return matches

def resolve_ambiguity(matches):
    print("Multiple files found:")
    for i, path in enumerate(matches):
        print(f"{i + 1}: {path}")
    while True:
        choice = input("Enter the number corresponding to the correct file: ")
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]
        else:
            print("Invalid input. Please try again.")

def resolve_file_path(file_name, search_path="."):
    matches = find_files(file_name, search_path)
    if not matches:
        return "File not found."
    elif len(matches) == 1:
        return matches[0]
    else:
        return resolve_ambiguity(matches)
