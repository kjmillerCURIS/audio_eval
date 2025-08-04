import os
import sys
import json
import openai
from openai_utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


def extract_json(response, return_json_as_str=False):
    start, end = response.find('{'), response.rfind('}')
    assert(start >= 0 and end >= 0)
    json_str = response[start:end+1]
    d = json.loads(json_str)
    if return_json_as_str:
        return json_str

    return d


def run_llm_helper(query):
    if isinstance(query, str):
        messages = [{"role": "system", "content": query}]
    else:
        assert(isinstance(query, list))
        assert(len(query) % 2 == 1)
        messages = []
        for t in range(len(query)):
            if t == 0:
                role = 'system'
            elif t % 2 == 1:
                role = 'assistant'
            else:
                role = 'user'

            messages.append({'role' : role, 'content' : query[t]})

    response = openai.chat.completions.create(model="gpt-4o", messages=messages)
    return response.choices[0].message.content


def run_llm(query, is_json=False, return_json_as_str=False):
    response = run_llm_helper(query)
    if is_json:
        return extract_json(response, return_json_as_str=return_json_as_str), response
    else:
        return response


def fill_out_prompt(prompt_template, **kwargs):
    prompt = prompt_template
    for k in sorted(kwargs.keys()):
        prompt = prompt.replace('FORMAT_TARGET' + k + 'FORMAT_TARGET', kwargs[k])

    return prompt
