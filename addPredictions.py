import sys
import json
from os import path
import re

# Given a Google Cluoud Vertex AI Text Entity Extraction jsonl export file
# and a directory where the exported dataset text files are stored, run predictions
# on each text file to find additional anotations.  Create a copy of the jsonl and
# add these annotations to the new jsonl as long as they don't overlap existing
# predictions.

GS_ROOT = "gs://cloud-ai-platform-e2e62516-544b-4394-af74-9eba164b1045/"

jsonl_filepath = sys.argv[1]
datasets_dir = sys.argv[2]

with open(jsonl_filepath, "r") as f:
    jsonl_lines = f.readlines()

def print_annotation(annotation, text):
    display_name = annotation['displayName']
    start_offset = int(annotation['startOffset'])
    end_offset = int(annotation['endOffset'])
    extracted_text = re.sub(re.compile("[\n\r]"), "", text[start_offset:end_offset].decode())
    print(f"  {display_name} [{start_offset}:{end_offset}]: {extracted_text}")

output = ""
for line in jsonl_lines:
    # get file info from jsonl
    file_info = json.loads(line)

    # clean up file info
    if "languageCode" in file_info:
        del file_info["languageCode"]
    if "dataItemResourceLabels" in file_info:
        del file_info["dataItemResourceLabels"]

    # open and read text file
    file_path = path.join(datasets_dir, file_info["textGcsUri"].replace(GS_ROOT, ""))
    print(file_path)
    with open(file_path, "rb") as f:
        text = f.read()
    print(f"length: {len(text)}")

    # clean up and print existing annotations
    for annotation in file_info["textSegmentAnnotations"]:
        if "annotationResourceLabels" in annotation:
            del annotation["annotationResourceLabels"]
        print(annotation, text)

    # TODO: Get predicted annotations for this file
    pred_annotations = []       

    # Go through predicted annotations looking for overlaps with existing annotations
    for pred_annotation in pred_annotations:
        overlaps = False
        for annotation in file_info["textSegmentAnnotations"]:
            if (pred_annotation["startOffset"] >= annotation["startOffset"] and
                pred_annotation["startOffset"] <= annotation["endOffset"]) or
               (pred_annotation["endOffset"] >= annotation["startOffset"] and
                pred_annotation["endOffset"] <= annotation["endOffset"]):
                overlaps = True
                break
        # add non-overlapping predictions to file info
        if not overlaps:
            print(annotation, text)
            file_info["textSegmentAnnotations"].append(annotation)

    # save updated file info 
    output += f"{json.dumps(file_info)}\n"

# write updated jsonl file
[root, ext] = path.splitext(jsonl_filepath)
output_path = f"{root}_modified{ext}"
with open(output_path, "w") as f:
    f.write(output)
