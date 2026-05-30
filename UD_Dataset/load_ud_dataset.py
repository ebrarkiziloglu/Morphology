from collections import defaultdict
from typing import List, Dict, Any, Optional

UD_GERMAN_GSD_SPLITS = {
    'train': 'de_gsd-ud-train.conllu',
    'dev': 'de_gsd-ud-dev.conllu',
    'test': 'de_gsd-ud-test.conllu',
}

UD_GERMAN_HDT_SPLITS = {
    'train-a-1': 'de_hdt-ud-train-a-1.conllu',
    'train-a-2': 'de_hdt-ud-train-a-2.conllu',
    'train-b-1': 'de_hdt-ud-train-b-1.conllu',
    'train-b-2': 'de_hdt-ud-train-b-2.conllu',
    'dev': 'de_hdt-ud-dev.conllu',
    'test': 'de_hdt-ud-test.conllu',
}

UD_GERMAN_PUD_SPLITS = {
    'test': 'de_pud-ud-test.conllu',
}

UD_GERMAN_LIT_SPLITS = {
    'test': 'de_lit-ud-test.conllu',
}

UD_GERMAN_SPLITS = [UD_GERMAN_GSD_SPLITS, UD_GERMAN_HDT_SPLITS, UD_GERMAN_PUD_SPLITS, UD_GERMAN_LIT_SPLITS]

def parse_conllu_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Parse a CoNLL-U file and return a list of sentences.
    
    Each sentence is a dict with:
        - 'sent_id': sentence ID
        - 'text': original sentence text
        - 'tokens': list of token dicts with CoNLL-U fields
    
    Token fields: id, form, lemma, upos, xpos, feats, head, deprel, deps, misc
    """
    sentences = []
    current_sentence = {'tokens': []}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                # Empty line marks end of sentence
                if current_sentence['tokens']:
                    sentences.append(current_sentence)
                current_sentence = {'tokens': []}
                continue

            if line.startswith('#'):
                # Metadata line
                if line.startswith('# sent_id'):
                    current_sentence['sent_id'] = line.split('=', 1)[1].strip()
                elif line.startswith('# text'):
                    current_sentence['text'] = line.split('=', 1)[1].strip()
                continue

            # Token line - 10 tab-separated fields
            fields = line.split('\t')
            if len(fields) != 10:
                continue

            token_id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = fields

            # Skip multi-word token lines (e.g., "6-7")
            if '-' in token_id:
                continue

            # Parse features into dict
            feats_dict = {}
            if feats != '_':
                for feat in feats.split('|'):
                    if '=' in feat:
                        key, value = feat.split('=', 1)
                        feats_dict[key] = value

            token = {
                'id': token_id,
                'form': form,
                'lemma': lemma,
                'upos': upos,  # Universal POS tag
                'xpos': xpos,  # Language-specific POS tag
                'feats': feats_dict,  # Morphological features as dict
                'feats_str': feats,   # Original feature string
                'head': head,
                'deprel': deprel,
                'deps': deps,
                'misc': misc
            }
            current_sentence['tokens'].append(token)

        # Don't forget the last sentence
        if current_sentence['tokens']:
            sentences.append(current_sentence)

    return sentences


def load_ud_dataset(
    base_path: str,
    splits_filemap: Optional[Dict[str, str]],
) -> Dict[str, List[Dict]]:
    """
    Load all splits of the UD dataset.
    
    Returns:
        dict with 'train', 'dev', 'test' keys, each containing list of sentences
    """
    import os

    from datasets import Dataset, DatasetDict
    splits = splits_filemap if splits_filemap is not None else {}
    datasets_dict = {}
    for split_name, filename in splits.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            parsed_sentences = parse_conllu_file(filepath)
            datasets_dict[split_name] = Dataset.from_list(parsed_sentences)
            print(f"Loaded {split_name}: {len(parsed_sentences)} sentences")

    dataset = DatasetDict(datasets_dict)

    return dataset
