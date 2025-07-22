import os
import sys
import json
import openai
from openai_utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


def extract_json(response):
    start, end = response.find('{'), response.rfind('}')
    assert(start >= 0 and end >= 0)
    json_str = response[start:end+1]
    d = json.loads(json_str)
    return d


def run_llm_helper(query):
    messages = [{"role": "system", "content": query}]

    response = openai.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def run_llm(query, is_json=False):
    response = run_llm_helper(query)
    if is_json:
        return extract_json(response)
    else:
        return response
