import spacy
from ioi_constants import original_ioi_names


def get_spacy_names(tokenizer):
    assert tokenizer is not None
    # Load SpaCy for NER
    nlp = spacy.load("en_core_web_sm")
    count = 0

    names = []
    # Iterate through vocabulary
    for token, token_id in tqdm(tokenizer.get_vocab().items()):
        # Check if it's a single token
        if len(tokenizer.tokenize(token)) == 1:
            # Use SpaCy for NER
            if token[0] == token[0].upper() and token[1:] == token[1:].lower() and token.isalpha() and len(token) > 3:
                doc = nlp(token)
                if doc.ents and doc.ents[0].label_ == 'PERSON':
                    names.append(token)
                    count += 1
    return names


def get_names(name_list: str, tokenizer=None):
    if name_list == 'original_ioi':
        names = original_ioi_names
    elif name_list == 'spacy_single_token_names':
        names = get_spacy_names(tokenizer)
    else:
        raise NotImplementedError

    print(f"Number of single tokens that are proper names: {len(names)}")
    return names
