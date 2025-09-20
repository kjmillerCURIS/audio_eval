#WIP TOOL 

import emphases

def emphasis_scores(transcript, audio_path, top_k=-1):
    """
    - Python package from ICASSP: https://pypi.org/project/emphases/
    - Needs newer GCC version: module load gcc/12.2.0
    - Likely not needed for SpeakBench considering the examples they have with emphasis
    """

    alignment, prominence = emphases.from_text_and_audio(
                                transcript, 
                                audio_path,
                                sample_rate)

    # Check which words were emphasized
    for word, score in zip(alignment, prominence[0]):
        print(f'{word} has a prominence of {score}')