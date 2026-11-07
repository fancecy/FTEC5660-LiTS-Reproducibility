"""Conservative answer-format extension, without reference-answer access."""
import re
from lits.components.utils import retrieve_answer_from_last_step

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_extended(text):
    strict = retrieve_answer_from_last_step(text)
    if strict:
        return strict, "original_marker"
    boxed = re.findall(r"\\boxed\{\s*(" + NUMBER + r")\s*\}", text)
    if boxed:
        return boxed[-1], "boxed_numeric"
    paragraph = re.split(r"\n\s*\n", text.strip())[-1].strip()
    # A final conclusion must explicitly introduce the answer and contain only
    # one numeric candidate. Do not take the last arbitrary number in reasoning.
    if re.match(r"(?i)^(?:so\b|thus\b|therefore\b|hence\b|final answer\s*:|answer\s*:)", paragraph):
        values = re.findall(NUMBER, paragraph)
        if len(values) == 1:
            return values[0].rstrip("."), "single_number_conclusion"
    return "", "unparsed"
