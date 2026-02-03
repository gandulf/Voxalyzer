# Compilation mode, standalone everywhere, except on macOS there app bundle
# nuitka-project-if: {OS} in ("Windows", "Linux", "FreeBSD"):
#    nuitka-project: --mode=standalone
#    nuitka-project: --windows-console-mode=force
# nuitka-project-else:
#    nuitka-project: --mode=standalone
#    nuitka-project: --macos-create-app-bundle
#
# nuitka-project: --include-data-dir={MAIN_DIRECTORY}/models=models
# nuitka-project: --mingw64
# nuitka-project: --output-dir=dist

import os
import sys
import json
import traceback
from os import PathLike
from typing import TypedDict

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import essentia
essentia.EssentiaLogger().warningActive = False
essentia.EssentiaLogger().infoActive = False

import numpy as np

from mp3 import *
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D, TempoCNN, TensorflowPredictMusiCNN


def get_path(path:str):
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        if sys._MEIPASS is not None:
            file_path = os.path.join(sys._MEIPASS, path)
        elif "__compiled__" in globals():
            file_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), path)
        else:
            file_path = os.path.join(os.path.dirname(sys.executable), path)
            if not os.path.exists(file_path):
                file_path = os.path.join(os.getcwd(), path)
    else:
        # Running as Python script
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    return file_path

# Models for final predictions
genre_model = TensorflowPredict2D(graphFilename=get_path("models/mtg_jamendo_genre-discogs-effnet-1.pb"))
mirex_model = TensorflowPredict2D(graphFilename=get_path("models/moods_mirex-msd-musicnn-1.pb"), input="serving_default_model_Placeholder", output="PartitionedCall")
russel_muse_model = TensorflowPredict2D(graphFilename=get_path("models/muse-msd-musicnn-2.pb"), output="model/Identity")
russel_deam_model = TensorflowPredict2D(graphFilename=get_path("models/deam-msd-musicnn-2.pb"), output="model/Identity")
russel_emomusic_model = TensorflowPredict2D(graphFilename=get_path("models/emomusic-msd-musicnn-2.pb"), output="model/Identity")

bpm_model = TempoCNN(graphFilename=get_path("models/deeptemp-k16-3.pb"))

engagement_model = TensorflowPredict2D(graphFilename=get_path("models/engagement_regression-discogs-effnet-1.pb"), output="model/Identity")
darkness_model = TensorflowPredict2D(graphFilename=get_path("models/nsynth_bright_dark-discogs-effnet-1.pb"), output="model/Softmax")

aggressive_model = TensorflowPredict2D(graphFilename=get_path("models/mood_aggressive-discogs-effnet-1.pb"), output="model/Softmax")
happy_model = TensorflowPredict2D(graphFilename=get_path("models/mood_happy-discogs-effnet-1.pb"), output="model/Softmax")
party_model = TensorflowPredict2D(graphFilename=get_path("models/mood_party-discogs-effnet-1.pb"), output="model/Softmax")
relaxed_model = TensorflowPredict2D(graphFilename=get_path("models/mood_relaxed-discogs-effnet-1.pb"), output="model/Softmax")
sad_model = TensorflowPredict2D(graphFilename=get_path("models/mood_sad-discogs-effnet-1.pb"), output="model/Softmax")
tonal_model = TensorflowPredict2D(graphFilename=get_path("models/tonal_atonal-discogs-effnet-1.pb"), output="model/Softmax")

moods_model = TensorflowPredict2D(graphFilename=get_path("models/mtg_jamendo_moodtheme-discogs-effnet-1.pb"))

mood_labels = [ "Exuberant", "Cheerful", "Melancholic", "Humorous", "Aggressive" ]

MODEL_VERSION= "0.2.0"

with open(get_path("models/mtg_jamendo_genre-discogs-effnet-1.json"), "r") as f:
    jamendo_genre_metadata = json.load(f)

with open(get_path("models/mtg_jamendo_moodtheme-discogs-effnet-1.json"), 'r') as f:
    jamendo_moods_metadata = json.load(f)

genre_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=get_path("models/discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
def analyze_genres(audio, threshold:float = 0.33):
    #genre_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = genre_embedding_model(audio)
    predictions = genre_model(embeddings)
    genre_embedding_model.reset()

    genre_labels = jamendo_genre_metadata["classes"]

    average_probs = np.mean(predictions, axis=0)

    # Create a list of (Genre, Probability) pairs
    results = list(zip(genre_labels, average_probs))

    # Sort by highest probability
    results.sort(key=lambda x: x[1], reverse=True)
    high_prob_moods = [mood for mood, prob in results if prob > threshold]
    return high_prob_moods if high_prob_moods else [results[0][0]]

mirex_embedding_model = TensorflowPredictMusiCNN(graphFilename=get_path("models/msd-musicnn-1.pb"), output="model/dense/BiasAdd")
def analyze_mirex(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = mirex_embedding_model(audio)
    predictions = mirex_model(embeddings)
    mirex_embedding_model.reset()

    average_probs = np.mean(predictions, axis=0)
    dominant_idx = np.argmax(average_probs)
    return mood_labels[dominant_idx]

moods_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=get_path("models/discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
def analyze_moods(audio, threshold:float=0.33):
    #embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = moods_embedding_model(audio)
    predictions = moods_model(embeddings)
    moods_embedding_model.reset()

    labels = jamendo_moods_metadata['classes']

    avg_probs = np.mean(predictions, axis=0)

    results = sorted(zip(labels, avg_probs), key=lambda x: x[1], reverse=True)

    # Sort by highest probability
    results.sort(key=lambda x: x[1], reverse=True)
    high_prob_moods = [mood for mood, prob in results if prob > threshold]
    return high_prob_moods if high_prob_moods else []

russel_embedding_model = TensorflowPredictMusiCNN(graphFilename=get_path("models/msd-musicnn-1.pb"), output="model/dense/BiasAdd")


def analyze_russel_emomusic(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = russel_embedding_model(audio)
    predictions = russel_emomusic_model(embeddings)
    russel_embedding_model.reset()

    final_scores = np.mean(predictions, axis=0)

    scores_0_10 = (final_scores - 1) * 1.25
    scores_0_10 = np.clip(scores_0_10, 0, 10)

    valence = scores_0_10[0]
    arousal = scores_0_10[1]

    return round(float(valence),1), round(float(arousal),1)


def analyze_russel_deam(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = russel_embedding_model(audio)
    predictions = russel_deam_model(embeddings)
    russel_embedding_model.reset()

    final_scores = np.mean(predictions, axis=0)

    scores_0_10 = (final_scores - 1) * 1.25
    scores_0_10 = np.clip(scores_0_10, 0, 10)

    valence = scores_0_10[0]
    arousal = scores_0_10[1]

    return round(float(valence),1), round(float(arousal),1)

def analyze_russel_muse(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = russel_embedding_model(audio)
    predictions = russel_muse_model(embeddings)
    russel_embedding_model.reset()

    final_scores = np.mean(predictions, axis=0)

    scores_0_10 = (final_scores - 1) * 1.25
    scores_0_10 = np.clip(scores_0_10, 0, 10)

    valence = scores_0_10[0]
    arousal = scores_0_10[1]

    return round(float(valence),1), round(float(arousal),1)

def analyze_bpm(audio):
    global_tempo, local_tempo, local_tempo_probabilities = bpm_model(audio)
    return int(global_tempo)

def analyze_engagement(audio):
    return _analyze_discogs_mood(audio,engagement_model,0)
def analyze_darkness(audio):
    return _analyze_discogs_mood(audio, darkness_model, 1)

def analyze_aggressive(audio):
    return _analyze_discogs_mood(audio, aggressive_model, 0)
def analyze_happy(audio):
    return _analyze_discogs_mood(audio,happy_model,0)
def analyze_party(audio):
    return _analyze_discogs_mood(audio,party_model,1)
def analyze_relaxed(audio):
    return _analyze_discogs_mood(audio,relaxed_model,1)
def analyze_sad(audio):
    return _analyze_discogs_mood(audio,sad_model,1)
def analyze_tonal(audio):
    return _analyze_discogs_mood(audio,tonal_model,1)

discogs_mood_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=get_path("models/discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
def _analyze_discogs_mood(audio, model:TensorflowPredict2D, index:int = 0):
    #embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = discogs_mood_embedding_model(audio)
    predictions = model(embeddings)
    discogs_mood_embedding_model.reset()

    average_probs = np.mean(predictions, axis=0)
    value = average_probs[index]

    return round(value * 10)

class AnalyzeResult(TypedDict):
    genres: list
    bpm: int
    valence: float
    arousal: float
    categories: dict
    tags: list

def analyze(file_path: str, force: bool = False, update :bool = True):

    dashes =int(120 - len(file_path))
    print("> " + file_path + " " + ("-" * dashes))

    if force:
        clean_mp3(file_path)

    if not force and is_analyzed(file_path, MODEL_VERSION):
        print("Skipping already analyzed file")
    else:
        id3 = easy_id3(file_path)
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()

        single_update = not force and update

        genres = get_genres(id3)
        if force or not genres:
            genres = normalize(analyze_genres(audio))
            print(f"🎭  Genre: {genres}")
            if single_update:
                update_mp3_genres(file_path, genres, True)

        bpm = get_bpm(id3)
        if force or not bpm:
            bpm = analyze_bpm(audio)
            print(f"🕑  BPM: {bpm}")
            if single_update:
                update_mp3_bpm(file_path, bpm)

        valence = get_category(id3, "Valence")
        arousal = get_category(id3, "Arousal")
        if force or not valence or not arousal:
            valence, arousal = analyze_russel_deam(audio)
            print(f"💖  Valence: {valence:.1f}")
            print(f"⚡   Arousal: {arousal:.1f}")
            if single_update:
                update_mp3_category(file_path, "Valence", valence)
                update_mp3_category(file_path, "Arousal", arousal)

        darkness = get_category(id3, "Darkness")
        if force or not darkness:
            darkness = analyze_darkness(audio)
            print(f"🔠  Darkness: {darkness:.0f}")
            if single_update:
                update_mp3_category(file_path, "Darkness", darkness)

        engagement = get_category(id3, "Engagement")
        if force or not engagement:
            engagement = analyze_engagement(audio)
            print(f"🔠  Engagement: {engagement:.0f}")
            if single_update:
                update_mp3_category(file_path, "Engagement", engagement)

        aggressive = get_category(id3, "Aggressive")
        if force or not aggressive:
            aggressive = analyze_aggressive(audio)
            print(f"🔠  Aggressive: {aggressive:.0f}")
            if single_update:
                update_mp3_category(file_path, "Aggressive", aggressive)

        happy = get_category(id3, "Happy")
        if force or not happy:
            happy = analyze_happy(audio)
            print(f"🔠  Happy: {happy:.0f}")
            if single_update:
                update_mp3_category(file_path, "Happy", happy)

        party = get_category(id3, "Party")
        if force or not party:
            party = analyze_party(audio)
            print(f"🔠  Party: {party:.0f}")
            if single_update:
                update_mp3_category(file_path, "Party", party)

        relaxed = get_category(id3, "Relaxed")
        if force or not relaxed:
            relaxed = analyze_relaxed(audio)
            print(f"🔠  Relaxed: {relaxed:.0f}")
            if single_update:
                update_mp3_category(file_path, "Relaxed", relaxed)

        sad = get_category(id3, "Sad")
        if force or not sad:
            sad = analyze_sad(audio)
            print(f"🔠  Sad: {sad:.0f}")
            if single_update:
                update_mp3_category(file_path, "Sad", sad)

        tonal = get_category(id3, "Tonal")
        if force or not tonal:
            tonal = analyze_tonal(audio)
            print(f"🔠  Tonal: {tonal:.0f}")
            if single_update:
                update_mp3_category(file_path, "Tonal", tonal)

        tags = get_tags(id3)
        if force or not tags:
            tags = [analyze_mirex(audio)]
            tags = tags + analyze_moods(audio)
            tags = normalize(tags)
            print(f"🎭  Tags: {tags}")
            if single_update:
                update_mp3_tags(file_path, tags)


        categories = {
                "Valence": valence,
                "Arousal": arousal,
                "Engagement": engagement,
                "Darkness": darkness,
                "Aggressive": aggressive,
                "Happy": happy,
                "Party":party,
                "Relaxed": relaxed,
                "Sad":sad,
                "Tonal":tonal
            }

        if update and not single_update:
            update_mp3(file_path, genres, bpm, categories, tags)

        update_mp3_version(file_path, MODEL_VERSION)

        return {
            "genres": genres,
            "bpm": bpm,
            "tags": tags,
            "categories": categories
        }

def main():
    if __file__ in sys.argv:
        sys.argv.remove(__file__)
    if "voxalyzer.py" in sys.argv:
        sys.argv.remove("voxalyzer.py")

    port = 8000
    try:
        index = sys.argv.index("--port")
        sys.argv.pop(index)
        port = int(sys.argv.pop(index))
    except ValueError:
        pass

    if len(sys.argv) == 0:
        import server
        server.serve(port)
    else:
        force = False
        clean = False
        if "--force" in sys.argv:
            sys.argv.remove("--force")
            force = True

        if "--clean" in sys.argv:
            sys.argv.remove("--clean")
            clean = True

        failed_files=[]
        for arg in sys.argv:
            if os.path.isfile(arg) and arg.lower().endswith(".mp3"):
                if clean:
                    clean_mp3(arg)
                else:
                    analyze(arg, force)
            elif os.path.isdir(arg):
                files = list_mp3s(arg)
                for file in files:
                    try:
                        if clean:
                            clean_mp3(file)
                        else:
                            analyze(file, force)
                    except KeyboardInterrupt:
                        sys.exit(0)
                    except Exception:
                        traceback.print_exc()
                        failed_files.append(file)
            elif arg.lower().endswith(".exe"):
                pass
            else:
                print("Unrecognized argument: %s" % arg)

        if len(failed_files)>0:
            print("Could not analyze:")
            for file in failed_files:
                print(file)

if __name__ == "__main__":
    main()
