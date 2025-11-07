from json_as_a_judge.config import HF_CACHE_PATH

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch.nn.functional as F

speechbrain_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=HF_CACHE_PATH + "/models--sb-speaker" 
    )

def speaker_match_score(audio_path_1, audio_path_2):

    def get_embedding(audio_path):
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        embedding = speechbrain_classifier.encode_batch(signal)  # Shape: (1, embedding_dim, 1)
        embedding = embedding.squeeze(-1).squeeze(0)  # Shape: (embedding_dim,)
        return embedding

    emb1 = get_embedding(audio_path_1).view(1, -1)
    emb2 = get_embedding(audio_path_2).view(1, -1)

    cos_sim = F.cosine_similarity(emb1, emb2, dim=1).item()
    return cos_sim


def get_consistency_score(audio_path, chunk=4):
    """
    - Quantifies speaker self consistency by chunking audio and comparing embeddings
    - Chunk is chunk size in seconds
    """
    # Load the audio file
    signal, fs = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    chunk_samples = int(chunk * fs)
    total_samples = signal.shape[1]

    embeddings_list = []

    # Extract embeddings for each chunk
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk_signal = signal[:, start:end]

        if chunk_signal.shape[1] < 10:
            continue
        
        embedding = speechbrain_classifier.encode_batch(chunk_signal)  # Shape: (1, embedding_dim, 1)
        embedding = embedding.squeeze(-1).squeeze(0)  # Shape: (embedding_dim,)
        embeddings_list.append(embedding)
    
    if len(embeddings_list) < 5:
        # Not enough chunks to compute metrics
        return None, None

    # Compute metric 1: sum of L2 distances between consecutive embeddings
    total_l2_distance = 0.0
    for i in range(len(embeddings_list) - 1):
        diff = embeddings_list[i+1] - embeddings_list[i]
        dist = torch.norm(diff, p=2).item()
        total_l2_distance += dist

    # Compute metric 2: average cosine similarity between consecutive embeddings
    cos_sims = []
    for i in range(len(embeddings_list) - 1):
        cos_sim = F.cosine_similarity(
            embeddings_list[i],
            embeddings_list[i+1],
            dim=1
        ).item()
        cos_sims.append(cos_sim)

    avg_cos_sim = sum(cos_sims) / len(cos_sims)

    return total_l2_distance, avg_cos_sim
