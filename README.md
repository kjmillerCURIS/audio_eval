What to install (use python=3.10):
* openai api (pip install openai)
* TTS-specific things:
  * whisperx (pip install git+https://github.com/m-bain/whisperx.git) (don't worry if it complains about numpy version)
  * pydub (pip install pydub)
  * ffmpeg (if you don't have sudo, just wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz, extract, and add it to PATH in your ~/.bashrc)
* Voice-assistant-specific things:
  * transformers (pip install transformers)
  * torchvision (pip install torchvision)
  * soundfile (pip install soundfile)
  * qwen_omni_utils (pip install qwen-omni-utils[decord] -U)
  * accelerate (pip install accelerate)
* Evaluation-specific things:
  * funasr (pip install funasr)
  * utmosv2 (pip install git+https://github.com/sarulab-speech/UTMOSv2.git)
  * ...
