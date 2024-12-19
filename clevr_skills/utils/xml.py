# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import re
from xml.dom import minidom


def write_xml_clean_and_pretty(xml: minidom.Document, output_path: str) -> None:
    """
    Writes XML to output_path, with proper indentation and newlines.
    :param xml: The XML to be written.
    :param output_path: The destination path.
    :return None:
    """
    text = xml.toprettyxml(indent="  ")
    text = re.sub(r">\s+<!", r"><!", text)
    text = re.sub(r"]>\s+<", r"]><", text)
    text = "".join([s for s in text.strip().splitlines(True) if s.strip()])
    with open(output_path, "w+", encoding="utf8") as file:
        file.write(text)
