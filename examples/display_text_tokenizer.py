import json

from chameleon.inference.constants import tokenizer_text_path
from rich import print

with open(tokenizer_text_path, "r") as fp:
    data = json.load(fp)


token_dict = data["model"]["vocab"]

import ipdb; ipdb.set_trace()
token_list = list(token_dict.keys())
print(len(token_list))
