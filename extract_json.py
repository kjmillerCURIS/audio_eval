import os
import sys
import json
from json import JSONDecodeError


def extract_json(response, return_json_as_str=False):
    start, end = response.find('{'), response.rfind('}')
    if not (start >= 0 and end >= 0):
        print('response is missing start or end bracket, cannot extract json')
        return None

    json_str = response[start:end+1]
    if ' "label\': ' in json_str[-20:]:
        json_str = json_str[:-20] + json_str[-20:].replace(' "label\': ', ' "label": ')

    try:
        d = json.loads(json_str)
    except JSONDecodeError as e:
        print(f'JSONDecodeError {e}')
#        print('Here was the string...')
#        print(json_str)
        return None

    if return_json_as_str:
        return json_str

    return d
