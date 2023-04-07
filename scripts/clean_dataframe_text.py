# library imports


import json
import unidecode
import pandas as pd
import numpy as np
import json
import re
import time
from tqdm import tqdm


def convert_si_to_number(x):
    total_stars = 0
    if 'K' in x:
        if len(x) > 1:
            total_stars = float(x.replace('K', '')) * 1000  # convert K to a thousand
    elif 'M' in x:
        if len(x) > 1:
            total_stars = float(x.replace('M', '')) * 1000000  # convert M to a million
    elif 'B' in x:
        total_stars = float(x.replace('B', '')) * 1000000000  # convert B to a Billion
    else:
        total_stars = int(x)  # Less than 1000
    return int(total_stars)


def clean_title(t):
    t = unidecode.unidecode(t)
    t = t.replace('\n', ' ')
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\d+', '', t)
    t = t.lower()
    t = t.strip()
    return t

def clean_description(d):
    d = unidecode.unidecode(d)
    d = d.replace('\n', ' ')
    d = re.sub(r'[^\w\s]', '', d)
    d = re.sub(r'\d+', '', d)
    if re.findall(r'(.*) brought to you by.*', d):
      d = re.sub(r'brought to you by.*', '', d)
    if re.search(r'(.*) sponsored by.*', d):
      d = re.sub(r'sponsored by.*', '', d)
    d = d.lower()
    d = d.strip()
    
    return d

def clean_description_list(dlist):
    
    new_string = ""
    for d in dlist:
      d = unidecode.unidecode(d)
      d = d.replace('\n', ' ')
      d = re.sub(r'[^\w\s]', '', d)
      d = re.sub(r'\d+', '', d)
      if re.findall(r'(.*) brought to you by.*', d):
        d = re.sub(r'brought to you by.*', '', d)
      if re.search(r'(.*) sponsored by.*', d):
        d = re.sub(r'sponsored by.*', '', d)
      d = d.lower()
      d = d.strip()
      new_string = new_string + " " + d
    return new_string


# Preferred method for cleaning the text currently
def clean_text(text):
    text = unidecode.unidecode(text)
    text = text.replace('\n', ' ')
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove "Sponsored by" phrases
    text = re.sub(r'(?i)sponsored\sby\s\w+', '', text)
    
    # Remove special characters and symbols (excluding spaces)
    text = re.sub(r'[^a-zA-Z0-9.,!?;:\s]', '', text)
    text = text.lower()
    text = text.strip()
    
    return text

# Define a function to join a list of strings
def join_strings(string_list, separator=" "):
    return separator.join(string_list)

# Combine the join and clean functions to apply to a column of a dataframe
def join_and_clean_text(string_list, separator=" "):
    # Join the list of strings
    single_text = join_strings(string_list, separator)

    # Clean the combined text
    cleaned_text = clean_text(single_text)

    return cleaned_text
