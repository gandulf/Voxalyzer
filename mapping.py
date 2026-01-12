SUMMARY_MOODS = [
    "happy",
    "energetic",
    "calm",
    "sad",
    "romantic",
    "dark",
    "tense",
    "dreamy",
    "hopeful",
    "neutral",
]

MOODS_TO_SUMMARY_MOOD = {
    # 1. Happy / Joyful
    "happy": "happy",
    "cheerful": "happy",
    "joyful": "happy",
    "fun": "happy",
    "playful": "happy",
    "uplifting": "happy",

    # 2. Energetic / Exciting
    "energetic": "energetic",
    "powerful": "energetic",
    "driving": "energetic",
    "intense": "energetic",
    "epic": "energetic",

    # 3. Calm / Relaxed
    "calm": "calm",
    "peaceful": "calm",
    "chill": "calm",
    "soothing": "calm",
    "mellow": "calm",
    "laid-back": "calm",
    "relaxing": "calm",

    # 4. Sad / Melancholic
    "sad": "sad",
    "melancholic": "sad",
    "sorrowful": "sad",
    "depressive": "sad",
    "gloomy": "sad",
    "blue": "sad",

    # 5. Romantic / Tender
    "romantic": "romantic",
    "tender": "romantic",
    "warm": "romantic",
    "intimate": "romantic",
    "sentimental": "romantic",

    # 6. Dark / Heavy
    "dark": "dark",
    "ominous": "dark",
    "heavy": "dark",
    "gritty": "dark",
    "menacing": "dark",

    # 7. Angry / Tense
    "angry": "tense",
    "tense": "tense",
    "aggressive": "tense",
    "anxious": "tense",
    "stressful": "tense",

    # 8. Dreamy / Atmospheric
    "dreamy": "dreamy",
    "ethereal": "dreamy",
    "ambient": "dreamy",
    "spacey": "dreamy",
    "atmospheric": "dreamy",

    # 9. Hopeful / Inspiring
    "hopeful": "hopeful",
    "inspirational": "hopeful",
    "optimistic": "hopeful",
    "triumphant": "hopeful",

    # 10. Neutral / Background
    "neutral": "neutral",
    "background": "neutral",
    "functional": "neutral",
    "unobtrusive": "neutral",
}

def collapse_mood_tags(tags):
    if tags:
        return sorted(
            set(MOODS_TO_SUMMARY_MOOD[tag] for tag in tags if tag in MOODS_TO_SUMMARY_MOOD)
        )
    else:
        return []
