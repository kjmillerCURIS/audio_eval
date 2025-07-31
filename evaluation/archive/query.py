import argparse
from evaluation.models.model_zoo import get_model


def main():
    parser = argparse.ArgumentParser(description="Run model inference on an audio file.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the input audio file.")
    args = parser.parse_args()

    model = get_model(args.model)
    output_audio_path, generated_text = model(args.audio_path)

    print(f"Generated text:\n{generated_text}")
    print(f"\nOutput audio saved at: {output_audio_path}")


if __name__ == "__main__":
    main()
