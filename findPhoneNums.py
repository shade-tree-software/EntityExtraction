import sys
import json
from os import listdir
from os.path import isfile, join, splitext, exists, basename
import re

# Takes an input path containing text files and scans each file with a regex looking for
# phone numbers.  If found, the filename and phone number offsets are added to a Google
# Cloud vertex AI Text Entity extraction jsonl import file.  Also, if the text file contains
# newlines, a modified version of the text file will be created without the newlines.

GCS_URI_PATH = "gs://irad_gcp_entity_extraction/"

input_path = sys.argv[1]

input_files = [join(input_path, f) for f in listdir(input_path) if isfile(join(input_path, f))]

jsonl_lines = []
for input_file in input_files:
    [root, ext] = splitext(input_file)
    if not root.endswith("_modified"):
        #with open(input_file, "rb") as i:
        with open(input_file, "r") as i:
            orig_text = i.read()
            #text = orig_text.replace("\n".encode(),"".encode())
            text = orig_text.replace("\n","")
            print(f"length: {len(orig_text)} -> {len(text)}")
            annotations = []
            #phone_nums = re.findall(r'\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}', text.decode())
            phone_nums = re.findall(r'\d{3}-\d{3}-\d{4}|\d{3}\.\d{3}\.\d{4}', text)
            offset = 0
            for num in phone_nums:
                offset = text.find(num, offset)
                annotation = {
                    "displayName": "phone_num",
                    "startOffset": offset,
                    "endOffset": offset + 12
                }
                annotations.append(annotation)
                print(f"{text[offset:offset + 12]}")
                offset += 1
            if len(phone_nums) > 0:
                jsonl_line = {
                    "textSegmentAnnotations": annotations,
                    "textGcsUri": join(GCS_URI_PATH, basename(input_file))
                }
                if len(orig_text) != len(text):
                    output_file = f"{root}_modified{ext}"
                    with open(output_file, "w") as o:
                        o.write(text)
                    jsonl_line["textGcsUri"] =  join(GCS_URI_PATH, basename(output_file))
                jsonl_lines.append(jsonl_line)
with open(join(input_path, "output.jsonl"), "w") as f:
    for line in jsonl_lines:
        f.write(f"{json.dumps(line)}\n")