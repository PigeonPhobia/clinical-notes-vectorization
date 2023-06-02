import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def print_note(s):
    print(s.replace(',', '\n'))


def match_titles(s):
    lines = s.split(',')
    return [line for line in lines if line.isupper() and line.strip()[-1] == ':']


def extract_content(s):
    titles = match_titles(s)
    if not titles: return s
    output = ''
    for i, title in enumerate(titles):
        if 'DATE' in title or 'TIME' in title: continue
        
        start_idx = s.find(title)
        assert start_idx != -1, "Title must be contained in the notes!"
        end_idx = start_idx + len(title)
        next_idx = s.find(titles[i+1]) if i < len(titles)-1 else len(s)
        assert next_idx != -1, "Title must be contained in the notes!"
        
        if i == 0: output += (s[:start_idx] + s[end_idx:next_idx])
        else: output += s[end_idx:next_idx]
    return output


def remove_number_index(s):
    return re.sub(r'\d+\.(?=\s)', ' ', s)


def remove_commas(s):
    return re.sub(r',', ' ', s)


def remove_redundant_whitespace(s):
    s = re.sub(r'\s+', ' ', s)
    return re.sub(r'^\s+', '', s)


def fill_empty_note(s):
    return 'not applicable' if not s else s


def spacy_normalize(notes_doc, lower=True, stop_removal=False, lemmatized=False, punct_removal=True, digit_removal=True):
    res = []
    key = "lemma_" if lemmatized else "text"
    to_remove_stop = lambda t: t.is_stop if stop_removal else False
    to_remove_punct = lambda t: t.pos_ in ['PUNCT'] if punct_removal else False
    to_remove_digit = lambda t: t.is_digit and not t.is_alpha if digit_removal else False
    for note in notes_doc:
        res.append([getattr(t, key).lower() if lower else getattr(t, key) for t in note if \
            not to_remove_stop(t) and \
            not to_remove_punct(t) and \
            not to_remove_digit(t) \
        ])
    return res


def join_as_sentence(note_list):
    return [" ".join(note) for note in note_list]


def count_vectorize(documents):
    count_vectorizer = CountVectorizer()
    X_tf = count_vectorizer.fit_transform(documents)
    return X_tf, count_vectorizer.get_feature_names_out()


def tfidf_vectorize(documents):
    tfidf_vectorizer = TfidfVectorizer()
    X_tf = tfidf_vectorizer.fit_transform(documents)
    return X_tf, tfidf_vectorizer.get_feature_names_out()


def extend_related_terms(G, notes_list):
    res = []
    for i, note_list in enumerate(notes_list):
        new_list = []
        for token in note_list:
            if G.has_node(token):
                token += ' ' + ' '.join(G.neighbors(token))
            new_list.append(token)
        res.append(new_list)
    return res


def get_aggregated_doc_vector(model, note_list):
    num_word = num_oov = 0
    res = []
    for note in note_list:
        vec_list = []
        note_tokens = note.split()
        for token in note_tokens:
            if model.has_index_for(token): vec_list.append(model.get_vector(token))
            else: num_oov += 1
            num_word += 1
        vec_array = np.stack(vec_list, axis=0)
        vec_array = vec_array.mean(axis=0)
        res.append(vec_array)
    res = np.stack(res, axis=0)
    return res, num_word, num_oov