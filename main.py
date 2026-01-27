import os
import sys
import json
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

# Models for final predictions
genre_model = TensorflowPredict2D(graphFilename="models/mtg_jamendo_genre-discogs-effnet-1.pb")
mirex_model = TensorflowPredict2D(graphFilename="models/moods_mirex-msd-musicnn-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
russel_model = TensorflowPredict2D(graphFilename="models/muse-msd-musicnn-2.pb", output="model/Identity")
bpm_model = TempoCNN(graphFilename="models/deeptemp-k16-3.pb")

engagement_model = TensorflowPredict2D(graphFilename="models/engagement_regression-discogs-effnet-1.pb", output="model/Identity")
darkness_model = TensorflowPredict2D(graphFilename="models/nsynth_bright_dark-discogs-effnet-1.pb", output="model/Softmax")

aggressive_model = TensorflowPredict2D(graphFilename="models/mood_aggressive-discogs-effnet-1.pb", output="model/Softmax")
happy_model = TensorflowPredict2D(graphFilename="models/mood_happy-discogs-effnet-1.pb", output="model/Softmax")
party_model = TensorflowPredict2D(graphFilename="models/mood_party-discogs-effnet-1.pb", output="model/Softmax")
relaxed_model = TensorflowPredict2D(graphFilename="models/mood_relaxed-discogs-effnet-1.pb", output="model/Softmax")
sad_model = TensorflowPredict2D(graphFilename="models/mood_sad-discogs-effnet-1.pb", output="model/Softmax")
tonal_model = TensorflowPredict2D(graphFilename="models/tonal_atonal-discogs-effnet-1.pb", output="model/Softmax")

moods_model = TensorflowPredict2D(graphFilename="models/mtg_jamendo_moodtheme-discogs-effnet-1.pb")

mood_labels = [ "Exuberant", "Cheerful", "Melancholic", "Humorous", "Aggressive" ]

with open("models/mtg_jamendo_genre-discogs-effnet-1.json", "r") as f:
    jamendo_genre_metadata = json.load(f)

with open("models/mtg_jamendo_moodtheme-discogs-effnet-1.json", 'r') as f:
    jamendo_moods_metadata = json.load(f)

genre_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
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

mirex_embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
def analyze_mirex(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = mirex_embedding_model(audio)
    predictions = mirex_model(embeddings)
    mirex_embedding_model.reset()

    average_probs = np.mean(predictions, axis=0)
    dominant_idx = np.argmax(average_probs)
    return mood_labels[dominant_idx]

moods_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
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

russel_embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
def analyze_russel(audio):
    #embedding_model = TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    embeddings = russel_embedding_model(audio)
    predictions = russel_model(embeddings)
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

discogs_mood_embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
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
    print(("-" * 50) + " " + file_path + " " + ("-" * 50))

    if force:
        clean_mp3(file_path)

    if not force and is_analyzed(file_path):
        print("Skipping already analyzed file %s" % file_path)
    else:
        audio = MonoLoader(filename=file_path, sampleRate=16000, resampleQuality=4)()

        genres = normalize(analyze_genres(audio))
        print(f"🎭  Genre: {genres}")

        bpm = analyze_bpm(audio)
        print(f"🕑  BPM: {bpm}")

        valence, arousal = analyze_russel(audio)
        print(f"💖  Valence: {valence:.1f}")
        print(f"⚡   Arousal: {arousal:.1f}")

        darkness = analyze_darkness(audio)
        print(f"🔠  Darkness: {darkness:.0f}")

        engagement = analyze_engagement(audio)
        print(f"🔠  Engagement: {engagement:.0f}")

        aggressive = analyze_aggressive(audio)
        print(f"🔠  Aggressive: {aggressive:.0f}")

        happy = analyze_happy(audio)
        print(f"🔠  Happy: {happy:.0f}")

        party = analyze_party(audio)
        print(f"🔠  Party: {party:.0f}")

        relaxed = analyze_relaxed(audio)
        print(f"🔠  Relaxed: {relaxed:.0f}")

        sad = analyze_sad(audio)
        print(f"🔠  Sad: {sad:.0f}")

        tonal = analyze_tonal(audio)
        print(f"🔠  Tonal: {tonal:.0f}")

        tags = []
        tags.append(analyze_mirex(audio))
        tags = tags + analyze_moods(audio)
        tags = normalize(tags)

        print(f"🎭  Tags: {tags}")

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
        if update:
            update_mp3(file_path, genres, bpm, categories, tags)

        return {
            "genres": genres,
            "bpm":bpm,
            "tags": tags,
            "categories": categories
        }

def main():
    if __file__ in sys.argv:
        sys.argv.remove(__file__)

    if len(sys.argv) == 0:
        import server
        server.serve()
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
            if os.path.isfile(arg):
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
                    except Exception as e:
                        print(e)
                        failed_files.append(file)
            else:
                print("Unrecognized argument: %s" % arg)

        if len(failed_files)>0:
            print("Could not analyze:")
            for file in failed_files:
                print(file)

if __name__ == "__main__":
    main()