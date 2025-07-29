from evaluation.config import HF_CACHE_PATH

import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


class Model:
    def __init__(self):
        """Initialize model resources, weights, and tokenizer here."""
        raise NotImplementedError

    def __call__(self, audio_file_path: str) -> tuple[str, str]:
        """
        Run inference on an audio file.

        Args:
            audio_file_path (str): path to input audio file

        Returns:
            tuple: (output_audio_path, generated_text)
        """
        raise NotImplementedError


class Qwen2_5Omni(Model):
    def __init__(self):
        """
        Initialize Qwen2.5 Omni model and processor.

        Args:
            cache_dir (str, optional): path to cache pretrained models
            device (str or torch.device, optional): device to run model on
        """

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto", device_map="auto")
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        self.USE_AUDIO_IN_VIDEO = True

    def __call__(self, input_file_path: str) -> tuple[str, str]:
        """
        Run inference on audio input.

        Args:
            audio_file_path (str): Path to input audio file

        Returns:
            tuple:
                output_audio_path (str): Path to saved output audio (wav)
                generated_text (str): Generated text response from model
        """
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                            "capable of perceiving auditory and visual inputs, as well as generating text and speech."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": input_file_path},
                ],
            },
        ]

        # Prepare inputs
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.USE_AUDIO_IN_VIDEO)

        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
        )

        # Move tensors to device and correct dtype
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Generate output (text tokens + audio)
        text_ids, audio = self.model.generate(
            **inputs,
            use_audio_in_video=self.USE_AUDIO_IN_VIDEO,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )

        # Decode generated text
        generated_text = self.processor.batch_decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Save generated audio
        output_audio_path = input_file_path.replace(".wav", "_qwen2.5-omni_output.wav")
        sf.write(
            output_audio_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )

        return output_audio_path, generated_text




MODELS = {
    "qwen-2.5-omni": Qwen2_5Omni,
}

def get_model(name):
    assert name in MODELS
    return MODELS[name]()