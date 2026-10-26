"""
Moduł odpowiedzialny za kanoniczną normalizację etykiet receptorów 5-HT.
"""

# Słownik mapujący warianty nazw na oficjalne etykiety IUPHAR
LABEL_MAPPING = {
    "5ht1a": "5-HT1A",
    "5-ht1a": "5-HT1A",
    "htr1a": "5-HT1A",

    "5ht1b": "5-HT1B",
    "5-ht1b": "5-HT1B",
    "htr1b": "5-HT1B",

    "5ht1d": "5-HT1D",
    "5-ht1d": "5-HT1D",

    "5ht1e": "5-HT1E",
    "5-ht1e": "5-HT1E",

    "5ht1f": "5-HT1F",
    "5-ht1f": "5-HT1F",

    # Przypadek historyczny: 5-HT1C został oficjalnie przeklasyfikowany jako 5-HT2C
    "5ht1c": "5-HT2C",
    "5-ht1c": "5-HT2C",

    "5ht2a": "5-HT2A",
    "5-ht2a": "5-HT2A",
    "htr2a": "5-HT2A",

    "5ht2b": "5-HT2B",
    "5-ht2b": "5-HT2B",

    "5ht2c": "5-HT2C",
    "5-ht2c": "5-HT2C",
    "htr2c": "5-HT2C",

    "5ht4": "5-HT4",
    "5-ht4": "5-HT4",

    "5ht5a": "5-HT5A",
    "5-ht5a": "5-HT5A",

    "5ht6": "5-HT6",
    "5-ht6": "5-HT6",

    "5ht7": "5-HT7",
    "5-ht7": "5-HT7",
}

VALID_CANONICAL_LABELS = set(LABEL_MAPPING.values())


def normalize_label(raw_label: str) -> str:
    """
    Zamienia dowolny wariant nazwy receptora na postać kanoniczną.
    Zwraca 'Other' w przypadku braku dopasowania.
    """
    clean_key = str(raw_label).strip().lower().replace("_", "-")
    return LABEL_MAPPING.get(clean_key, "Other")