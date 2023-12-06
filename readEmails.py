import sys
import os
from email.parser import BytesParser
from email import policy
import html2text
import re

input_dir = sys.argv[1]
output_dir = sys.argv[2]

parser = BytesParser(policy=policy.default)

index = 0
for current_dir, subdirs, files in os.walk(input_dir):
    for filename in files:
        full_filename = os.path.join(current_dir, filename)
        with open(full_filename, "rb") as f:
            msg = parser.parse(f)
        text = ""
        if len(msg.keys()):
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    text += part.as_string()
                elif content_type == 'text/html':
                    try:
                        text += html2text.html2text(part.as_string())
                    except AssertionError:
                        continue
        text = ' '.join(list(set(re.findall(r'\b[a-zA-Z\']+\b', text))))
        if len(text) > 0:
            output_file = os.path.join(output_dir, f"{index}.txt")
            with open(output_file, "w") as f:
                f.write(text)
            index += 1