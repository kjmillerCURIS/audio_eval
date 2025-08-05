import os
import sys
import json
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from llm_utils import fill_out_prompt
os.environ["TRANSFORMERS_CACHE"] = "models"


DEFAULT_SYSTEM_PROMPT = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'
VOICE_ASSISTANT_SYSTEM_PROMPT_FILENAME = 'voice_assistant_system_prompt.txt'
USE_AUDIO_IN_VIDEO = True


def setup_qwen_model():
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    return {'model' : model, 'processor' : processor}


#knowledge_base should be a string
#will read user query from audio_input_filename and write Qwen's response to audio_output_filename
def run_qwen(audio_input_filename, setting_name, user_name, knowledge_base, qwen_model, audio_output_filename):
    with open(VOICE_ASSISTANT_SYSTEM_PROMPT_FILENAME, 'r') as f:
        voice_assistant_system_prompt = f.read()

    voice_assistant_system_prompt = fill_out_prompt(voice_assistant_system_prompt, knowledge_base=knowledge_base)
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": DEFAULT_SYSTEM_PROMPT}
            ],
        },
        {
            "role": "system",
            "content": [
                {"type": "text", "text": voice_assistant_system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_input_filename}
            ],
        },
    ]

    text = qwen_model['processor'].apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = qwen_model['processor'](text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(qwen_model['model'].device).to(qwen_model['model'].dtype)
    text_ids, audio = qwen_model['model'].generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, eos_token_id=qwen_model['processor'].tokenizer.eos_token_id, max_new_tokens=64)
    qwen_response_text = qwen_model['processor'].batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    sf.write(
        audio_output_filename,
        audio.reshape(-1).detach().cpu().numpy(),
        samplerate=24000,
    )

    return {'qwen_response_audio_filename' : audio_output_filename, 'qwen_response_text' : qwen_response_text}


if __name__ == '__main__':
    knowledge_base = {'patient' : {'name' : 'Jane Doe', 'patientID' : '12345'}, 'workshops' : [{'name' : 'Arthritis Management', 'description' : 'How to deal with symptoms of arthritis, one day at a time. Learn about home remedies and hear from long-term arthritis patients as well as leading experts in the field.', 'location' : 'Onsite (Ballroom 1A)', 'time' : '10 AM'}]}
    knowledge_base = json.dumps(knowledge_base)
    setting_name = 'hospital'
    user_name = 'patient'
    audio_input_filename = 'outputs/hospital_0_A_user.wav'
    audio_output_filename = 'outputs/hospital_0_A_qwen_response.wav'
    qwen_model = setup_qwen_model()
    run_qwen(audio_input_filename, setting_name, user_name, knowledge_base, qwen_model, audio_output_filename)
