import re


def clean_ocr_text(text):

    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)

    text = re.sub(r'([a-z]),([a-zA-Z])', r'\1, \2', text)

    text = re.sub(r'\)\.([A-Z])', r'). \1', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()