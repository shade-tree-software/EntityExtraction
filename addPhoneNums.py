import sys
import json
from os import path
import re

# Given a Google Cluoud Vertex AI Text Entity Extraction jsonl import file
# and a directory where the associated dataset text files are stored, go through
# the referenced text files using a regex to look for phone numbers.  For each
# phone number found, add it to a copy of the jsonl file.

GS_ROOT = "gs://cloud-ai-platform-e2e62516-544b-4394-af74-9eba164b1045/"

jsonl_filepath = sys.argv[1]
datasets_dir = sys.argv[2]

with open(jsonl_filepath, "r") as f:
    jsonl_lines = f.readlines()

output = ""
for line in jsonl_lines:
    file_info = json.loads(line)
    if "languageCode" in file_info:
        del file_info["languageCode"]
    if "dataItemResourceLabels" in file_info:
        del file_info["dataItemResourceLabels"]
    file_path = path.join(datasets_dir, file_info["textGcsUri"].replace(GS_ROOT, ""))
    print(file_path)
    with open(file_path, "rb") as f:
        text = f.read()
        print(f"length: {len(text)}")
        phone_nums = re.findall("\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}", text.decode())
        offset = 0
        for num in phone_nums:
            offset = text.find(num.encode(), offset)
            file_info["textSegmentAnnotations"].append({
                "displayName": "phone_num",
                "startOffset": offset,
                "endOffset": offset + 12
            })
            offset += 1
        for annotation in file_info["textSegmentAnnotations"]:
            display_name = annotation['displayName']
            start_offset = int(annotation['startOffset'])
            end_offset = int(annotation['endOffset'])
            extracted_text = re.sub(re.compile("[\n\r]"), "", text[start_offset:end_offset].decode())
            print(f"  {display_name} [{start_offset}:{end_offset}]: {extracted_text}")
            if "annotationResourceLabels" in annotation:
                del annotation["annotationResourceLabels"]
        output += f"{json.dumps(file_info)}\n"
[root, ext] = path.splitext(jsonl_filepath)
output_path = f"{root}_modified{ext}"
with open(output_path, "w") as f:
    f.write(output)