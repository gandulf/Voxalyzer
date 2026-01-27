import os

import glob
import json
from os import PathLike
from pathlib import Path

from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError, COMM, ID3
from mutagen.mp3 import MP3

def get_comment(id3, key):
    return [c.text[0] for c in id3.getall('COMM')]

def set_comment(id3, key, value):
    id3.add(COMM(encoding=3, lang='XXX', desc='', text=value))

# 2. Register the key
EasyID3.RegisterKey('comment', getter=get_comment, setter=set_comment)
EasyID3.RegisterTXXXKey('ai_categories', 'ai_categories')
EasyID3.RegisterTXXXKey('ai_tags', 'ai_tags')
EasyID3.RegisterTXXXKey('ai_summary', 'ai_summary')

def list_mp3s(path: str) -> list:
    base_path = Path(path)
    files = [str(f) for f in base_path.rglob("*.[Mm][Pp]3")]
    return [os.path.join(path, filename) for filename in files]

def _easy_audio(path: str) -> EasyID3:
    try:
        audio = EasyID3(path)
    except ID3NoHeaderError:
        # If the file has no ID3 tag at all, create one
        from mutagen.id3 import ID3
        meta = ID3()
        meta.save(path)
        audio = EasyID3(path)

    return audio

def clean_mp3(path: str):
    try:
        audio = ID3(path)
        audio.delall('TXXX')
        audio.save()
    except:
        pass


def update_mp3(path: str, genres: list, bpm: int, categories: dict, tags:list):
    audio = _easy_audio(path)

    audio["ai_categories"] = [json.dumps(categories)]
    audio['ai_tags'] = tags
    audio['genre'] = genres
    audio['bpm'] = [bpm]

    audio.save()

def update_mp3_categories(path: str, categories: dict):
    audio = _easy_audio(path)

    audio["ai_categories"] = [json.dumps(categories)]

    audio.save()

def add_mp3_tags(path: str, tags: list):
    if not tags:
        return

    audio = _easy_audio(path)

    orig_tags = audio.get("ai_tags",[])
    audio["ai_tags"] = tags + orig_tags

    audio.save()

def normalize(tag: str):
    if isinstance(tag, str):
        tag = tag.capitalize()
        if "Soundtrack" in tag or "Ost" in tag or "Soundtack" in tag:
            tag = "Soundtrack"

        if "Unknown" == tag:
            return None

        return tag
    elif isinstance(tag, list):
        tags = sorted(set([normalize(t) for t in tag]))
        tags = [tag for tag in tags if tag is not None]
        return tags
    else:
        return tag

def update_mp3_tags(path: str, tags: list = []):
    audio = _easy_audio(path)

    audio['ai_tags'] = tags
    audio.save()


def update_mp3_summary(path: str, new_summary: str):
    audio = _easy_audio(path)

    audio["comment"]= [new_summary]
    audio.save()

def update_mp3_genre(path: str, new_genres: list , force: bool = False):
    audio = _easy_audio(path)

    current_genre = audio.get('genre', ['Unknown'])[0]

    if (force or current_genre == "Unknown") and new_genres:
        audio['genre'] = new_genres
        audio.save()

def update_mp3_bpm(path: str, bpm=None, force: bool = False):
    audio = _easy_audio(path)

    current_bpm = audio.get('bpm', [None])[0]

    if (force or current_bpm is None) and bpm:
        audio['bpm'] = [bpm]
        audio.save()

def update_mp3_category(path: str, category: str, new_value: int):
    audio = _easy_audio(path)

    cats = {}
    txxx_cats = audio.get("ai_categories",None)

    if txxx_cats:
        try:
            cats = json.loads(txxx_cats[0])
        except:
            pass

    if new_value is None:
        if category in cats:
            del cats[category]
    else:
        cats[category] = new_value

    audio["ai_categories"]=[json.dumps(cats)]
    audio.save()

def has_genre(file_path: str) -> bool:
    try:
        audio = EasyID3(file_path)
    except ID3NoHeaderError:
        return False

    # 2. READ the genre
    # EasyID3 returns a list, so we take the first element if it exists
    current_genre = audio.get('genre', [None])[0]

    return current_genre is not None


def has_bpm(file_path: str) -> bool:
    try:
        audio = EasyID3(file_path)
    except ID3NoHeaderError:
        return False

    # 2. READ the bpm
    current_bpm = audio.get('bpm', [None])[0]

    return current_bpm is not None


def is_analyzed(file_path: str) -> bool:
    try:
        audio = EasyID3(file_path)
    except ID3NoHeaderError:
        return False

    return "ai_categories" in audio and audio["ai_categories"] is not None

def print_mp3_tags(file_path: str):
    """Prints all ID3 tags from an MP3 file."""
    try:
        audio = MP3(file_path, ID3=ID3)
        if audio.tags:
            print("\n--- Tags for {0} ---", file_path)
            for key, value in audio.tags.items():
                if not key.startswith("APIC"):
                    print("{0}: {1}", key,value)
            print("---------------------------------\n")
        else:
            print("No tags found in {}", file_path)
    except Exception as e:
        print("An error occurred while reading tags: {0}",e)


