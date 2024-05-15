import json


def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of JSON objects."""
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    """Writes a list of JSON objects to a JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def merge_jsonl_files(file1_path, file2_path, output_file_path):
    """Merges two JSONL files ensuring unique idx values."""
    data1 = read_jsonl(file1_path)
    data2 = read_jsonl(file2_path)

    # Create a dictionary to keep track of unique idx values
    unique_data = {item['idx']: item for item in data1}

    # Merge data from the second file, ensuring idx values are unique
    for item in data2:
        if item['idx'] not in unique_data:
            unique_data[item['idx']] = item

    # Convert the dictionary back to a list
    merged_data = list(unique_data.values())

    # Write the merged data to the output file
    write_jsonl(merged_data, output_file_path)


# Example usage
file1_path = '/home/filippo/PycharmProjects/explaining-contextualized-hate/datasets/pavlopoulos20/data/gc.jsonl'
file2_path = '/home/filippo/PycharmProjects/explaining-contextualized-hate/datasets/pavlopoulos20/data/gn.jsonl'
output_file_path = '/home/filippo/PycharmProjects/explaining-contextualized-hate/datasets/pavlopoulos20/dataset.jsonl'

merge_jsonl_files(file1_path, file2_path, output_file_path)
print(f"Merged JSONL files saved to {output_file_path}")