#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHA_TTS_PATH = REPO_ROOT / "third_party" / "Matcha-TTS"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
if str(MATCHA_TTS_PATH) not in sys.path:
    sys.path.append(str(MATCHA_TTS_PATH))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CosyVoice voice changer e2e flow.")
    parser.add_argument(
        "--model-dir",
        default=str(REPO_ROOT / "checkpoints" / "Fun-CosyVoice3-0.5B"),
        help="Checkpoint directory to load with AutoModel.",
    )
    parser.add_argument(
        "--source-audio",
        default=str(REPO_ROOT / "data" / "input_audio.wav"),
        help="Source audio whose content and cadence are preserved.",
    )
    parser.add_argument(
        "--ref-audio",
        default=str(REPO_ROOT / "data" / "ref_voice.mp3"),
        help="Reference audio whose speaker identity is transferred.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data"),
        help="Directory where converted wav files are written.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming inference.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Output speed passed to inference_vc.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    source_audio = Path(args.source_audio)
    ref_audio = Path(args.ref_audio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"model dir does not exist: {model_dir}")
    if not source_audio.exists():
        raise FileNotFoundError(f"source audio does not exist: {source_audio}")
    if not ref_audio.exists():
        raise FileNotFoundError(f"reference audio does not exist: {ref_audio}")

    cosyvoice = AutoModel(model_dir=str(model_dir))
    streamed_chunks = []
    for index, result in enumerate(
        cosyvoice.inference_vc(str(source_audio), str(ref_audio), stream=args.stream, speed=args.speed)
    ):
        streamed_chunks.append(result["tts_speech"])
        output_path = output_dir / f"voice_changed_{index}.wav"
        torchaudio.save(str(output_path), result["tts_speech"], cosyvoice.sample_rate)
        print(output_path)

    if args.stream and streamed_chunks:
        stitched_path = output_dir / "voice_changed_stream_full.wav"
        torchaudio.save(str(stitched_path), torch.cat(streamed_chunks, dim=1), cosyvoice.sample_rate)
        print(stitched_path)


if __name__ == "__main__":
    main()
