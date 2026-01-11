import os
import sys

import glob
import json
from os import PathLike
from pathlib import Path

import torch
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, TXXX, COMM

import numpy as np

# --------------------------------------------------------------------------------
# START OF WORKAROUND FOR BASE ROOT RESTRICTION OF Music2Emotion
# Music2Emotion only works it it thinks it sits directly at the base root dir.
# Since I used it as sub directory submodule i need to fake this behavior
# Path to Music2Emotion repo root
VENDOR_ROOT = Path(__file__).parent / "Music2Emotion"

# Patch current working directory temporarily
# Dangerous if your script relies on relative paths elsewhere
os.chdir(VENDOR_ROOT)

# Insert at front of sys.path so it's preferred over site-packages
sys.path.insert(0, str(VENDOR_ROOT))
#
# END OF WORKAROUND
# --------------------------------------------------------------------------------
from Music2Emotion.music2emo import Music2emo

modul = Music2emo()


def list_mp3s(path: str | PathLike[str]) -> list[str]:
    files = glob.glob("**/*.mp3", root_dir=path, recursive=True)
    return [os.path.join(path, filename) for filename in files]

def update_mp3_categories(path: str | PathLike[str], categories: dict[str, int]):
    audio = MP3(path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()

    audio.tags.add(TXXX(encoding=3, desc='ai_categories', text=[json.dumps(categories)]))

    audio.tags.add(COMM(encoding=3, text=["processed by Voxalyzer 1.0"]))

    audio.save()


def add_mp3_tags(path: str | PathLike[str], tags: list[str]):
    audio = MP3(path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()

    orig_tags = audio.tags.get("TXXX:ai_tags")
    if orig_tags is None:
        orig_tags = []

    if tags:
        audio.tags.add(
            TXXX(
                encoding=3,
                desc='ai_tags',
                text=tags + orig_tags
            )
        )
    audio.save()


def update_mp3_tags(path: str | PathLike[str], tags: list[str]):
    audio = MP3(path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()

    if tags:
        audio.tags.add(
            TXXX(
                encoding=3,
                desc='ai_tags',
                text=tags
            )
        )
    audio.save()


def update_mp3_category(path: str | Path, category: str, new_value: int | None):
    audio = MP3(path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()

    cats = {}
    txxx_cats = audio.tags.get("TXXX:ai_categories")
    if txxx_cats:
        try:
            cats = json.loads(txxx_cats.text[0])
        except:
            pass

    if new_value is None:
        if category in cats:
            del cats[category]
    else:
        cats[category] = new_value

    audio.tags.add(TXXX(encoding=3, desc='ai_categories', text=[json.dumps(cats)]))
    audio.save()


def analyze(file_path: str | PathLike[str]):
    output_dic = modul.predict(file_path)

    valence = output_dic["valence"]
    arousal = output_dic["arousal"]
    predicted_moods = output_dic["predicted_moods"]

    valence = np.interp(valence, [1, 9], [1, 10])
    arousal = np.interp(arousal, [1, 9], [1, 10])

    print(("-" * 50) + " " + file_path + " " + ("-" * 50))
    print(f"🎭  Tags: {', '.join(predicted_moods) if predicted_moods else 'None'}")
    print(f"💖  Valence: {valence:.2f} (Scale: 1-10)")
    print(f"⚡  Arousal: {arousal:.2f} (Scale: 1-10)")

    update_mp3_category(file_path, "Valence", round(valence))
    update_mp3_category(file_path, "Arousal", round(arousal))
    update_mp3_tags(file_path, predicted_moods)

def main():
    if len(sys.argv) > 0:
        root_dir = sys.argv[1]
    else:
        root_dir = __file__.parent

    if torch.cuda.is_available():
        print("Using GPU Mode quite fast")
    else:
        print("Using CPU MODE a bit slower")

    files = list_mp3s(root_dir)
    for file in files:
        analyze(file)

if __name__ == "__main__":
    main()