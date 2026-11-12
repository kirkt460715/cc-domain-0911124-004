import math
from collections import Counter
from corpus_gate.errors import integer, shape, fail, digest
from corpus_gate.text.normalize import tokens


DEFAULTS = {
    'minimum_document_tokens': 5,
    'minimum_document_frequency': 1,
    'maximum_document_frequency': 90,
    'maximum_vocabulary': 2000,
    'high_frequency_minimum_documents': 20,
}


def settings(value):
    value = {} if value is None else value
    shape(value, set(), set(DEFAULTS))
    result = {**DEFAULTS, **value}
    integer(result['minimum_document_tokens'], 'minimum_document_tokens', 0, 100000)
    integer(result['minimum_document_frequency'], 'minimum_document_frequency', 1, 10000)
    integer(result['maximum_document_frequency'], 'maximum_document_frequency', 1, 100)
    integer(result['maximum_vocabulary'], 'maximum_vocabulary', 1, 100000)
    integer(result['high_frequency_minimum_documents'], 'high_frequency_minimum_documents', 1, 100000)
    return result


def counts(text):
    return Counter(token['text'].casefold() for token in tokens(text))


def build_counts(items, options=None):
    values = settings(options)
    tally = {}
    frequency = Counter()
    for identity, term_counts in items:
        tally[identity] = term_counts
        frequency.update(set(term_counts))
    documents = len(tally)
    if not documents:
        return {'terms': [], 'document_frequency': [], 'idf': [], 'documents': 0, 'options': values, 'candidates': 0, 'discarded': {'below_minimum': [], 'above_maximum': [], 'over_limit': []}, 'discarded_counts': {'below_minimum': 0, 'above_maximum': 0, 'over_limit': 0}, 'digest': digest({'terms': [], 'documents': 0, 'options': values})}
    candidates = sorted(frequency, key=lambda term: (-frequency[term], term))
    below = [term for term in candidates if frequency[term] < values['minimum_document_frequency']]
    # A high-document-frequency ceiling removes stopwords, but on a tiny corpus a
    # term present in most documents may carry the only distinguishing signal.
    high_frequency_filter_applied = documents >= values['high_frequency_minimum_documents']
    above = [term for term in candidates if high_frequency_filter_applied and term not in below and frequency[term] * 100 > values['maximum_document_frequency'] * documents]
    eligible = [term for term in candidates if term not in below and term not in above]
    kept = sorted(eligible[:values['maximum_vocabulary']])
    over = sorted(eligible[values['maximum_vocabulary']:])
    idf = [round(math.log((1 + documents) / (1 + frequency[term])) + 1.0, 12) for term in kept]
    model = {'terms': kept, 'document_frequency': [frequency[term] for term in kept], 'idf': idf, 'documents': documents, 'options': values, 'high_frequency_filter_applied': high_frequency_filter_applied, 'candidates': len(frequency), 'discarded': {'below_minimum': below[:50], 'above_maximum': above[:50], 'over_limit': over[:50]}, 'discarded_counts': {'below_minimum': len(below), 'above_maximum': len(above), 'over_limit': len(over)}}
    model['digest'] = digest(model)
    return model


def transform(text, model):
    """Apply the frozen vocabulary and TF-IDF weights, then L2 normalize."""
    return transform_counts(counts(text), model)


def transform_counts(tally, model):
    index = {term: position for position, term in enumerate(model['terms'])}
    known = []
    unseen = []
    for term in sorted(tally):
        position = index.get(term)
        if position is None:
            unseen.append((term, tally[term]))
        else:
            known.append((position, (1.0 + math.log(tally[term])) * model['idf'][position]))
    norm = math.sqrt(sum(weight * weight for _, weight in known))
    if norm:
        weights = [(position, round(weight / norm, 12)) for position, weight in known]
    else:
        weights = []
    ranked = sorted(unseen, key=lambda item: (-item[1], item[0]))
    known_tokens = sum(tally[term] for term in tally if term in index)
    return {'indices': [position for position, _ in weights], 'weights': [weight for _, weight in weights], 'tokens': sum(tally.values()), 'known_tokens': known_tokens, 'unseen': len(unseen), 'unseen_terms': [term for term, _ in ranked[:50]], 'empty': not weights}


def as_map(vector):
    return {position: weight for position, weight in zip(vector['indices'], vector['weights'])}
