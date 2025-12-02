import os
import sys
import openai
from openai import OpenAI
from openai_utils import OPENAI_API_KEY, GOOGLE_API_KEY
openai.api_key = OPENAI_API_KEY
from google import genai
from pydantic import BaseModel
from huggingface_hub import login
from huggingface_utils import HUGGINGFACE_API_KEY
from extract_json import extract_json
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
deepinfra_client = OpenAI(api_key=HUGGINGFACE_API_KEY, base_url='https://api.deepinfra.com/v1/openai')


DEEPINFRA_MODELS = {'gpt-oss-20b-fp4' : 'openai/gpt-oss-20b',
                    'llama3.3-70b-fp8' : 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
                    'llama3.3-nemotron-49b-fp8' : 'nvidia/Llama-3.3-Nemotron-Super-49B-v1.5'}
SUPPORTED_LLM_NAMES = ['gpt4o', 'gemini-2.5-flash'] + sorted(DEEPINFRA_MODELS.keys())


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


#CAUTION: gpt4o and gemini-2.5-flash handle list query differently
def run_llm_helper(query, llm_name='gpt4o', dimensionwise=0, skip_config=False):
    assert(llm_name in  SUPPORTED_LLM_NAMES)
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
        if isinstance(query, list):
            assert(skip_config)
            chat = gemini_client.chats.create(model=llm_name)
            responses = []
            for query_one in query:
                response = chat.send_message(query_one)
                responses.append(response.text)

            return responses

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
    elif llm_name in DEEPINFRA_MODELS:
        assert(isinstance(query, str))
        model = DEEPINFRA_MODELS[llm_name]
        messages = [{'role' : 'system', 'content' : query}]
        response = deepinfra_client.chat.completions.create(model=model, messages=messages, stream=False)
        return response.choices[0].message.content
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
