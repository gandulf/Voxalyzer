import sys
import os
from typing import TypedDict


class AnalyzeResult(TypedDict):
    genres: list
    bpm: int
    categories: dict
    tags: list

def to_ten(prob):
    return int(round(float(prob)*10))

def rescale(value, old_min=1, old_max=9, new_min=0, new_max=10):
    # Standard rescaling formula
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

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


def get_best(labels, probabilities, threshold = 0.33):
    items = list(zip(labels, probabilities))
    # Sort by highest probability
    items.sort(key=lambda x: x[1], reverse=True)
    high_prob_item = [item for item, prob in items if prob > threshold]
    return high_prob_item if high_prob_item else [items[0][0]]