# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_text_entity_extraction_sample]
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google import api_core

import sys
import json
from os.path import splitext, join

num_args = len(sys.argv)
if num_args < 2 or num_args > 5 or ("-a" in sys.argv and "-n" in sys.argv):
  print("\nUsage:")
  print(f"  {sys.argv[0]} input_filename [options...]\n")
  print("Options:")
  print("  -j output as jsonl file")
  print("  -h create a copy of the input file with the filename added as the")
  print("     first line of the content")
  print("  -a adjust offsets to account for Google Cloud 'double newlines' bug")
  print("  -n create a copy of the input file with newlines replaced by spaces")
  print("   Note: -a and -n cannot be used together\n")
  exit(0)

input_filename = sys.argv[1]
output_as_jsonl = "-j" in sys.argv
adjust_offsets = "-a" in sys.argv
remove_newlines = "-n" in sys.argv
add_header = "-h" in sys.argv

MAX_LEN = 9900
GCS_URI_PATH = "gs://irad_gcp_entity_extraction/"

def get_prediction_dict(predictions, chars_read, index, content):
  prediction_dict = {
    "displayName": predictions['displayNames'][index],
  }

  # Google "Size Limit" Bug
  #
  # Google has a limit of 9998 bytes for predictions.  If the file is larger than this
  # limit we will break it up into chunks, but then we have to modify the offsets that
  # Google returns because we want them to be measured from the beginning of the file
  # and not just from the beginning of the chunk.

  # Google "Double Newlines" Bug
  # 
  # When importing files for cloud-based training, Google often seems to convert each
  # newline character into two newlines before adding the files to the training dataset.
  # This increases the length of the file and it means that any offsets that you specify
  # in your jsonl files will be wrong.  To counteract this when generating jsonl files
  # we can count the number of newlines in the training file prior to each offset and
  # add this count to the offset to get the correct asjusted offset.  We only do this
  # if the user has selected option -a.
  # 
  # The other way to get around this problem is to specify option -n in which case we
  # will create a new input file with all of the newlines replaced with spaces.
  # This seems to be the better of the two options.
  #
  start_offset = int(predictions['textSegmentStartOffsets'][index]) + chars_read
  offset_adjustment = content[0:start_offset].count('\n') if adjust_offsets else 0
  start_offset += offset_adjustment
  end_offset = int(predictions['textSegmentEndOffsets'][index]) + chars_read + offset_adjustment
  if output_as_jsonl == False:
    prediction_dict["confidence"] = 100 * predictions['confidences'][index]
    prediction_dict["value"] = content[start_offset:end_offset]
  prediction_dict["startOffset"] = start_offset
  prediction_dict["endOffset"] = end_offset 
  return prediction_dict

def predict_text_entity_extraction_sample(
  project: str,
  endpoint_id: str,
  location: str,
  content: str,
  chars_read: int,
  stop_index: int,
  api_endpoint: str = "us-central1-aiplatform.googleapis.com",):

  # The AI Platform services require regional API endpoints.
  client_options = {"api_endpoint": api_endpoint}
  # Initialize client that will be used to create and send requests.
  # This client only needs to be created once, and can be reused for multiple requests.
  client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
  # The format of each instance should conform to the deployed model's prediction input schema
  instance = predict.instance.TextExtractionPredictionInstance( content=content[chars_read:stop_index]).to_value()
  instances = [instance]
  parameters_dict = {}
  parameters = json_format.ParseDict(parameters_dict, Value())
  endpoint = client.endpoint_path( project=project, location=location, endpoint=endpoint_id)
  response = client.predict( endpoint=endpoint, instances=instances, parameters=parameters)
  files_predictions = response.predictions
  predictions = files_predictions[0] #  assume only one file
  l = lambda index: get_prediction_dict(predictions, chars_read, index, content) 
  return list(map(l, range(len(predictions['ids']))))

  # See gs://google-cloud-aiplatform/schema/predict/prediction/text_extraction_1.0.0.yaml for the format of the predictions.
  #   jsonl_lines = []
  #   for predictions in files_predictions:
  #     jsonl_lines.append({
  #       "textSegmentAnnotations": get_predictions_array(predictions, chars_read, content),
  #       "textGcsUri": join(GCS_URI_PATH, input_filename)
  #     })
  #   print(jsonl_lines)
  #   if output_as_jsonl:
  #     with open(f"{splitext(input_filename)[0]}.jsonl", "w") as output_file:
  #       for line in jsonl_lines:
  #         output_file.write(str(line))
  #         output_file.write("\n")
  #   else:
  #     for line in jsonl_lines:
  #       for annotation in line["textSegmentAnnotations"]:
  #         print(f'{annotation["displayName"]} ({annotation["confidence"]}) [{annotation["startOffset"]}:{annotation["endOffset"]}]: {annotation["value"]}')

with open(input_filename, "r") as f:
  content = f.read()

if add_header:
  content = f"{input_filename}\n{content}"

if remove_newlines:
  content = content.replace("\n", " ")

if remove_newlines or add_header:
  filename_parts = splitext(input_filename)
  input_filename = f"{filename_parts[0]}_modified{filename_parts[1]}"
  with open(input_filename, "w") as f:
    f.write(content)

file_length = len(content)
print(f"File is {file_length} characters")

all_annotations = []
chars_read = 0
try:
  while chars_read < file_length:
    stop_index = min(chars_read + MAX_LEN, file_length)
    print(f"checking charcaters {chars_read} to {stop_index - 1}")
    annotations = predict_text_entity_extraction_sample(
      "1071272000574",
      "3328355837696540672",
      "us-central1",
      content,
      chars_read,
      stop_index)
    if output_as_jsonl:
      all_annotations += annotations
      print(f"found {len(annotations)} annotations")
    else:
      for annotation in annotations:
        print(f'{annotation["displayName"]} ({annotation["confidence"]}) [{annotation["startOffset"]}:{annotation["endOffset"]}]: {annotation["value"]}')
    chars_read += MAX_LEN
  if output_as_jsonl:
    with open(f"{splitext(input_filename)[0]}.jsonl", "w") as output_file:
      jsonl_line = {
        "textSegmentAnnotations": all_annotations,
        "textGcsUri": join(GCS_URI_PATH, input_filename)
      }
      output_file.write(str(jsonl_line))
except api_core.exceptions.InvalidArgument as e:
  print(e)


