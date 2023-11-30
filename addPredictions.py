import sys
import json
from os import path
import re
from time import sleep

from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google import api_core

# Given a Google Cluoud Vertex AI Text Entity Extraction jsonl export file
# and a directory where the exported dataset text files are stored, run predictions
# on each text file to find additional anotations.  Create a copy of the jsonl and
# add these annotations to the new jsonl as long as they don't overlap existing
# predictions.

PROJECT = "1071272000574"
ENDPOINT_ID = "3328355837696540672"
LOCATION = "us-central1"
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"

GS_ROOT = "gs://cloud-ai-platform-e2e62516-544b-4394-af74-9eba164b1045/"
MAX_LEN = 9900

jsonl_filepath = sys.argv[1]
datasets_dir = sys.argv[2]

with open(jsonl_filepath, "r") as f:
    jsonl_lines = f.readlines()

def get_predictions_for_chunk(text: str, chars_read: int, stop_index: int):
    # get predictions from Vertex AI for specified chunk of file
    client_options = {"api_endpoint": API_ENDPOINT}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    content = text[chars_read:stop_index] 
    instances = [predict.instance.TextExtractionPredictionInstance(content=content).to_value()]
    parameters = json_format.ParseDict({}, Value())
    endpoint = client.endpoint_path(project=PROJECT, location=LOCATION, endpoint=ENDPOINT_ID)
    while True:
        try:
            response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
            break
        except api_core.exceptions.ServiceUnavailable:
            print("api_core.exceptions.ServiceUnavailable: retrying...")
            sleep(1)

    predictions = response.predictions[0] #  assume only one file

    # convert Vertex AI predictions object into list of annotations
    annotations = []
    for index in range(len(predictions['ids'])):
        annotations.append({
            "displayName": predictions['displayNames'][index],
            "startOffset": int(predictions['textSegmentStartOffsets'][index]) + chars_read,
            "endOffset": int(predictions['textSegmentEndOffsets'][index]) + chars_read
        })
    return annotations

def get_predictions(text):
    all_annotations = []
    chars_read = 0
    while chars_read < len(text):
        stop_index = min(chars_read + MAX_LEN, len(text))
        annotations = get_predictions_for_chunk(text, chars_read, stop_index)
        all_annotations += annotations
        chars_read += MAX_LEN
    return all_annotations

def print_annotation(annotation, text):
    display_name = annotation['displayName']
    start_offset = int(annotation['startOffset'])
    end_offset = int(annotation['endOffset'])
    extracted_text = re.sub(re.compile("[\n\r]"), "", text[start_offset:end_offset])
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
    with open(file_path, "rt") as f:
        text = f.read().replace('\n', '\n\n') # fix for Vertex AI double-newline bug
    print(file_path)
    print(f"length: {len(text)}")
    first_100 = text[:100].replace('\n',' ')
    print(f"first 100 chars: {first_100}")

    # clean up and print existing annotations
    print("Existing annotations:")
    for annotation in file_info["textSegmentAnnotations"]:
        annotation["startOffset"] = int(annotation["startOffset"])
        annotation["endOffset"] = int(annotation["endOffset"])
        if "annotationResourceLabels" in annotation:
            del annotation["annotationResourceLabels"]
        print_annotation(annotation, text)

    # Get predicted annotations for this file
    pred_annotations = get_predictions(text)      

    # Go through predicted annotations looking for overlaps with existing annotations
    print("Predicted annotations:")
    for pred_annotation in pred_annotations:
        overlaps = False
        for annotation in file_info["textSegmentAnnotations"]:
            if ((pred_annotation["startOffset"] >= annotation["startOffset"] and
                 pred_annotation["startOffset"] <= annotation["endOffset"]) or
                (pred_annotation["endOffset"] >= annotation["startOffset"] and
                 pred_annotation["endOffset"] <= annotation["endOffset"])):
                overlaps = True
                break
        # add non-overlapping predictions to file info
        if not overlaps:
            print_annotation(pred_annotation, text)
            file_info["textSegmentAnnotations"].append(pred_annotation)

    # save updated file info 
    output += f"{json.dumps(file_info)}\n"

# write updated jsonl file
[root, ext] = path.splitext(jsonl_filepath)
output_path = f"{root}_modified{ext}"
with open(output_path, "w") as f:
    f.write(output)
