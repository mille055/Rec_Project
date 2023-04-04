# library imports

import unidecode
import pandas as pd
import numpy as np
import json
import re
import time
from tqdm import tqdm
import pickle

from clean_dataframe_text import *

example_df = pd.read_pickle('../data/podcast_df_040423.pkl')
altered_df = example_df.copy()
altered_df.episode_descriptions = altered_df.episode_descriptions.apply(join_and_clean_text)

print(altered_df)
print('example of cleaned episode description:')
print(altered_df.iloc[0].episode_descriptions)
