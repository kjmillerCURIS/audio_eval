import os
import sys
sys.path.append('.')
from WinoSound.run_alm_on_conversations import openai_client, encode_audio_file, make_audio_message, make_text_message


AUDIO_PATH = 'WinoSound/meow.wav'


if __name__ == '__main__':
    messages = []
    messages.append(make_text_message('Meow! Please translate this audio into cat so I can understand it!', 'gpt4o'))
    messages.append(make_audio_message(AUDIO_PATH, 'gpt4o'))
    response = openai_client.chat.completions.create(model='gpt-4o-audio-preview', messages=[{'role' : 'user', 'content' : messages}])
    print(response.choices[0].message.content.strip())
