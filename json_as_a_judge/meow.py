import os
import sys
import copy
import json
import pprint


FULL_JSON_DESCRIPTION = \
{
  "user/agent_text_transcription": "a text transcription of the speech, obtained using Whisper",
  "user/agent_emotion": "a vector of emotion scores for the speech from the emotion2vec model",
  "user/agent_audio_quality": {
    "UTMOSv2_Mean_Opinion_Score": "Mean opinion score from UTMOSv2 model (1-5, higher is better)",
    "DNSMOS_Personalized_Signal_Quality": "Signal quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Background_Quality": "Background noise quality score from DNSMOS model (1-5, higher is better)",
    "DNSMOS_Personalized_Overall_Quality": "Overall naturalness and audio quality score from DNSMOS model (1-5, higher is better)",
    "P808_Overall_Quality": "Overall naturalness and audio quality score from P.808 recommendation standard (1-5, higher is better)",
  },
  "user/agent_audio_properties": {
    "Mean_Pitch_Hz": "Mean pitch (fundamental frequency) of speech",
    "Std_Dev_Pitch_Hz": "Standard deviation in pitch",
    "Mean_RMS_dBFS": "Mean root mean squared decibels full scale",
    "Speech_Rate_WPM": "speech rate in words per minute",
    "Articulation_Rate_WPM": "speech rate in words per minute excluding pauses and gaps in speech",
  },
  "user/agent_speaker_consistency": "consistency in agent speaker identity, measured using cosine similarity between speaker embeddings in adjacent audio chunks"
}


OUTPUTA_JSON = {'agent_emotion' : 'mowierjwe', 'agent_text_transcription' : 'mweoir', 'agent_audio_quality' : 'weoijweroijwer'}


def produce_json_description(outputA_json, params):
    p = params
    json_description = copy.deepcopy(FULL_JSON_DESCRIPTION)
    top_level_keys = sorted(json_description.keys())
    for k in top_level_keys:
        if k.replace('user/agent_', 'agent_') not in outputA_json:
            json_description.pop(k, None)

    return json.dumps(json_description, indent=4)


meow = produce_json_description(OUTPUTA_JSON, None)
import pdb
pdb.set_trace()
