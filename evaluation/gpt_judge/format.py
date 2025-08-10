import json
import pandas as pd

# Sample paths (to be replaced by actual user input)
full_json_path = "/projectnb/ivc-ml/ac25/Audio_Eval/AudioJudge/experiments/main_experiments/datasets/speakbench508_dataset.json"
indices_json_path = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/gpt4o_test/sampled_indices_speakbench.json"
csv_judge_path = "/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/gpt4o_test/output_speakbench.csv"

# --- Step 1: Convert full JSON to SpeakBench-style JSONL format (filtered by selected indices) ---

# Load annotation JSON (list of dicts) and selected indices
with open(full_json_path, "r") as f:
    full_data = json.load(f)

with open(indices_json_path, "r") as f:
    selected_indices = set(json.load(f))

# Convert format
filtered_output = []
for row in full_data:
    if row["index"] in selected_indices:
        label = row["label"]
        if label == "tie":
            preference = "tie"
        elif label == "1":
            preference = "model1"
        elif label == "2":
            preference = "model2"
        else:
            print("Skipping invalid sample")
            continue  # Skip invalid
        

        filtered_output.append({
            "model1": row["model_a"],
            "model2": row["model_b"],
            "preference": preference
        })

with open("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/gpt4o_test/human_pref_speakbench.jsonl", "w") as f:
    for entry in filtered_output:
        json.dump(entry, f)
        f.write("\n")


# --- Step 2: Convert prediction CSV to effective win rate format ---

# Load predictions
df = pd.read_csv(csv_judge_path)

# Normalize prediction column: 1 = model_a wins, 2 = model_b wins, 'tie' = tie
score_tracker = {}

def update_score(model, value):
    if model not in score_tracker:
        score_tracker[model] = {"score": 0.0, "count": 0}
    score_tracker[model]["score"] += value
    score_tracker[model]["count"] += 1

for _, row in df.iterrows():
    a = row["model_a"]
    b = row["model_b"]
    pred = str(row["prediction"]).strip().lower()

    if pred == "1":
        update_score(a, 1.0)
        update_score(b, 0.0)
    elif pred == "2":
        update_score(a, 0.0)
        update_score(b, 1.0)
    elif pred == "tie":
        update_score(a, 0.5)
        update_score(b, 0.5)

# Compute effective win rate
winrate_data = []
for model, vals in score_tracker.items():
    if vals["count"] > 0:
        effective_win_rate = vals["score"] / vals["count"]
        winrate_data.append({"model_name": model, "effective_win_rate": effective_win_rate})

winrate_df = pd.DataFrame(winrate_data)


winrate_df.to_csv("/projectnb/ivc-ml/ac25/Audio_Eval/audio_eval/evaluation/gpt4o_test/pred_win_rates_speakbench.csv", index=False)
