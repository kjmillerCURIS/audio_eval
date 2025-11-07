import os
import sys
import json
from json import JSONDecodeError
import openai
from openai_utils import OPENAI_API_KEY, GOOGLE_API_KEY
openai.api_key = OPENAI_API_KEY
from google import genai
from pydantic import BaseModel
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


class OverallJudgement(BaseModel):
    reasoning: str
    label: str


class DimensionwiseJudgement(BaseModel):
    reasoning: str
    content: str
    voice_quality: str
    instruction_following_audio: str


class FusionJudgement(BaseModel):
    reasoning: str
    overall_label: str


def extract_json(response, return_json_as_str=False):
    start, end = response.find('{'), response.rfind('}')
    assert(start >= 0 and end >= 0)
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


def run_llm_helper(query, llm_name='gpt4o', dimensionwise=0, skip_config=False):
    assert(llm_name in ['gpt4o', 'gemini-2.5-flash'])
    if llm_name == 'gpt4o':
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
    elif llm_name == 'gemini-2.5-flash':
        assert(isinstance(query, str))
        if skip_config:
            response = gemini_client.models.generate_content(model=llm_name, contents=query)
        else:
            response = gemini_client.models.generate_content(
                            model=llm_name,
                            contents=query,
                            config={
                                "response_mime_type": "application/json",
                                "response_schema": [OverallJudgement, DimensionwiseJudgement, FusionJudgement][dimensionwise],
                            },
                        )
        
        return response.text
    else:
        assert(False)


#use dimensionwise=2 for fusion
def run_llm(query, is_json=False, return_json_as_str=False, llm_name='gpt4o', dimensionwise=0, skip_config=False):
    response = run_llm_helper(query, llm_name=llm_name, dimensionwise=dimensionwise, skip_config=skip_config)
    if is_json:
        return extract_json(response, return_json_as_str=return_json_as_str), response
    else:
        return response


def fill_out_prompt(prompt_template, **kwargs):
    prompt = prompt_template
    for k in sorted(kwargs.keys()):
        prompt = prompt.replace('FORMAT_TARGET' + k + 'FORMAT_TARGET', kwargs[k])

    assert('FORMAT_TARGET' not in prompt) #make sure everything is filled out!
    return prompt
