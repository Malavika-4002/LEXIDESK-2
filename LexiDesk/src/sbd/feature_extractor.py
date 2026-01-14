# src/sbd/feature_extractor.py

import re

def get_token_signature(token):
    return "".join([
        'C' if char.isupper()
        else 'c' if char.islower()
        else 'D' if char.isdigit()
        else char
        for char in token
    ])

def get_token_length_feature(token):
    length = len(token)
    if length < 4:
        return 'short'
    elif length <= 6:
        return 'normal'
    else:
        return 'long'

def token_to_features(token, text, start, end):
    """
    Core feature extraction for CRF-based SBD.
    """
    char_before = text[start - 1] if start > 0 else '<BOS>'
    char_after = text[end] if end < len(text) else '<EOS>'

    # --- GENERALIZED ABBREVIATION PATTERNS ---
    is_initial = re.fullmatch(r'[A-Z]\.', token) is not None
    is_acronym = re.fullmatch(r'([A-Z]\.)+', token) is not None
    is_title_like = re.fullmatch(r'(Mr|Mrs|Ms|Dr)\.', token, re.IGNORECASE) is not None
    is_likely_abbreviation = is_initial or is_acronym or is_title_like

    features = {
        'bias': 1.0,
        'token': token,
        'lower': token.lower(),
        'sig': get_token_signature(token),
        'len_cat': get_token_length_feature(token),

        'is_lower': token.islower(),
        'is_upper': token.isupper(),
        'is_title': token.istitle(),
        'is_digit': token.isdigit(),

        'char_before': char_before,
        'is_space_before': char_before.isspace(),

        'char_after': char_after,
        'is_space_after': char_after.isspace(),

        # --- ABBREVIATION AWARENESS ---
        'is_likely_abbreviation': is_likely_abbreviation,

        'ends_with_period': token.endswith('.'),
        'is_numeric_with_period': re.fullmatch(r'\d+\.', token) is not None,
    }

    # Existing strong signal
    if char_after.isupper():
        features['next_char_is_upper'] = True

    # ✅ NEW FEATURE (CRITICAL FIX)
    if isinstance(char_after, str) and char_after.islower():
        features['next_starts_lower'] = True

    return features


def add_neighboring_token_features(sentence_features):
    expanded_sentence_features = []

    sentence_features = [None] + sentence_features + [None]

    for i in range(1, len(sentence_features) - 1):
        new_features = sentence_features[i].copy()

        prev_feats = sentence_features[i - 1]
        if prev_feats is not None:
            for key, value in prev_feats.items():
                new_features['-1:' + key] = value
        else:
            new_features['BOS'] = True

        next_feats = sentence_features[i + 1]
        if next_feats is not None:
            for key, value in next_feats.items():
                new_features['+1:' + key] = value
        else:
            new_features['EOS'] = True

        expanded_sentence_features.append(new_features)

    return expanded_sentence_features
