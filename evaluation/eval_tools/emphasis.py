import emphases

def emphasis_scores(text_path, audio_path, top_k=-1):
    # Detect emphases
    alignment, prominence = emphases.from_file(text_path, audio_path)

    # Check which words were emphasized
    for word, score in zip(alignment, prominence[0]):
        print(f'{word} has a prominence of {score}')