import hashlib
import math
from corpus_gate.errors import fail, integer, shape


def _map(vector):
    return {position: weight for position, weight in zip(vector['indices'], vector['weights'])}


def _normalize(mapping):
    total = math.sqrt(sum(value * value for value in mapping.values()))
    if not total:
        return {}
    return {position: value / total for position, value in mapping.items()}


def _materialize(mapping):
    positions = sorted(mapping)
    return {'indices': positions, 'weights': [round(mapping[position], 12) for position in positions]}


def combine(vectors):
    totals = {}
    for vector in vectors:
        for position, value in _map(vector).items():
            totals[position] = totals.get(position, 0.0) + value
    return _normalize(totals)


def cosine(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    score = sum(value * right.get(position, 0.0) for position, value in left.items())
    return min(1.0, max(-1.0, score))


def _hash(seed, iteration, identity):
    return int(hashlib.sha256(f'{seed}\0{iteration}\0{identity}'.encode('utf-8')).hexdigest(), 16)


def _centers(points, k, seed, iteration):
    identities = sorted(points)
    first = min(identities, key=lambda identity: (_hash(seed, iteration, identity), identity))
    chosen = [first]
    while len(chosen) < k:
        def separation(identity):
            nearest = min(1.0 - cosine(points[identity], points[center]) for center in chosen)
            return nearest
        candidate = max((identity for identity in identities if identity not in chosen), key=lambda identity: (separation(identity), -_hash(seed, iteration, identity)))
        chosen.append(candidate)
    return [dict(points[identity]) for identity in chosen]


def _adjacency(forbidden):
    neighbours = {}
    for left, right in forbidden:
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    return neighbours


_SEARCH = {'exhausted': False}


def _assign(points, centers, forbidden, limit=50000):
    """Assign points to centres without linking any forbidden pair.

    Greedy nearest-centre assignment is corrected by a bounded backtracking
    search over constrained points.  ``_SEARCH['exhausted']`` distinguishes a
    proven conflict (search space exhausted) from a search-budget cutoff; the
    latter is reported as ``unknown_constraint_feasibility`` rather than
    silently claiming that constraints are impossible.
    """
    neighbours = _adjacency(forbidden)
    assignments = {}
    constrained = sorted(neighbours, key=lambda identity: (-len(neighbours[identity]), _hash('order', 0, identity), identity))
    def rank(identity, topic):
        return (-cosine(points[identity], centers[topic]), topic)
    expanded = [0]
    _SEARCH['exhausted'] = False
    def search(index):
        expanded[0] += 1
        if expanded[0] > limit:
            return False
        if index == len(constrained):
            return True
        identity = constrained[index]
        blocked = {assignments[other] for other in neighbours[identity] if other in assignments}
        allowed = [topic for topic in range(len(centers)) if topic not in blocked]
        for topic in sorted(allowed, key=lambda topic: rank(identity, topic)):
            assignments[identity] = topic
            if search(index + 1):
                return True
            del assignments[identity]
        return False
    if constrained and not search(0):
        _SEARCH['exhausted'] = expanded[0] <= limit
        return None
    for identity in sorted(points):
        if identity in assignments:
            continue
        blocked = {assignments[other] for other in neighbours.get(identity, ()) if other in assignments}
        allowed = [topic for topic in range(len(centers)) if topic not in blocked] or list(range(len(centers)))
        assignments[identity] = min(allowed, key=lambda topic: rank(identity, topic))
    return assignments


def _violations(assignments, forbidden):
    return [list(pair) for pair in forbidden if pair[0] in assignments and pair[1] in assignments and assignments[pair[0]] == assignments[pair[1]]]


def _recenter(points, assignments, centers):
    result = []
    for topic in range(len(centers)):
        members = [identity for identity, value in assignments.items() if value == topic]
        if not members:
            result.append(dict(centers[topic]))
            continue
        keys = set()
        for member in members:
            keys.update(points[member])
        result.append(_normalize({key: sum(points[member].get(key, 0.0) for member in members) for key in keys}))
    return result


def solve(points, k, forbidden, seed='topic', iteration=0, maximum_iterations=50):
    """Constrained spherical k-means under cosine distance.

    ``points`` maps atomic units to normalized sparse vectors.  ``forbidden``
    are must-not-link pairs the final partition must honour; feasibility is not
    assumed merely because the unconstrained nearest centres look plausible.
    """
    forbidden = [(left, right) for left, right in forbidden if left in points and right in points and left != right]
    if k < 1:
        return {'feasible': False, 'reason': 'invalid_k', 'assignments': {}, 'centers': [], 'iterations': 0, 'distortion': None}
    if k > len(points):
        return {'feasible': False, 'reason': 'not_enough_units', 'assignments': {}, 'centers': [], 'iterations': 0, 'distortion': None, 'available_units': len(points)}
    centers = _centers(points, k, seed, iteration)
    assignments = {}
    steps = 0
    for steps in range(maximum_iterations):
        updated = _assign(points, centers, forbidden)
        if updated is None:
            reason = 'constraint_unsatisfied' if _SEARCH['exhausted'] else 'unknown_constraint_feasibility'
            return {'feasible': False, 'reason': reason, 'assignments': {}, 'centers': centers, 'iterations': steps + 1, 'distortion': None, 'violations': [list(pair) for pair in forbidden]}
        if updated == assignments:
            break
        assignments = updated
        centers = _recenter(points, assignments, centers)
    violations = _violations(assignments, forbidden)
    empty = [topic for topic in range(k) if topic not in assignments.values()]
    if violations:
        return {'feasible': False, 'reason': 'constraint_unsatisfied', 'assignments': assignments, 'centers': centers, 'iterations': steps + 1, 'distortion': None, 'violations': violations}
    if empty:
        return {'feasible': False, 'reason': 'empty_topic', 'assignments': assignments, 'centers': centers, 'iterations': steps + 1, 'distortion': None, 'empty_topics': empty}
    loss = 0.0
    for identity, topic in assignments.items():
        loss += 1.0 - cosine(points[identity], centers[topic])
    return {'feasible': True, 'reason': None, 'assignments': assignments, 'centers': centers, 'iterations': steps + 1, 'distortion': round(loss, 12)}


def solve_best(points, k, forbidden, seed, restarts=3, maximum_iterations=50):
    feasible = None
    last = None
    for attempt in range(max(1, restarts)):
        candidate = solve(points, k, forbidden, seed, attempt, maximum_iterations)
        if candidate['feasible']:
            if feasible is None or (candidate['distortion'], attempt) < (feasible['distortion'], feasible.get('attempt', 0)):
                candidate['attempt'] = attempt
                feasible = candidate
        elif last is None or candidate['reason'] in {'constraint_unsatisfied', 'empty_topic'}:
            last = candidate
    if feasible is not None:
        return feasible
    result = dict(last or {'feasible': False, 'reason': 'no_solution', 'assignments': {}, 'centers': [], 'iterations': 0, 'distortion': None})
    result['feasible'] = False
    return result


def silhouette(points, assignments):
    identities = sorted(points)
    labels = sorted({assignments[identity] for identity in identities})
    if len(labels) < 2 or len(identities) < 2:
        return None
    scores = []
    for identity in identities:
        own = [other for other in identities if other != identity and assignments[other] == assignments[identity]]
        if not own:
            scores.append(0.0)
            continue
        inside = sum(1.0 - cosine(points[identity], points[other]) for other in own) / len(own)
        outside = []
        for label in labels:
            if label == assignments[identity]:
                continue
            users = [other for other in identities if assignments[other] == label]
            outside.append(sum(1.0 - cosine(points[identity], points[other]) for other in users) / len(users))
        nearest = min(outside)
        denominator = max(inside, nearest)
        scores.append(0.0 if denominator == 0 else (nearest - inside) / denominator)
    return round(sum(scores) / len(scores), 12)


def _combinations(value):
    return value * (value - 1) // 2


def adjusted_rand(left, right):
    common = sorted(set(left) & set(right))
    if len(common) < 2:
        return None
    counts = {}
    rows = {}
    columns = {}
    for identity in common:
        key = (left[identity], right[identity])
        counts[key] = counts.get(key, 0) + 1
        rows[left[identity]] = rows.get(left[identity], 0) + 1
        columns[right[identity]] = columns.get(right[identity], 0) + 1
    index = sum(_combinations(value) for value in counts.values())
    row_index = sum(_combinations(value) for value in rows.values())
    column_index = sum(_combinations(value) for value in columns.values())
    pairs = _combinations(len(common))
    expected = row_index * column_index / pairs if pairs else 0.0
    maximum = (row_index + column_index) / 2
    if maximum == expected:
        return 1.0
    return round((index - expected) / (maximum - expected), 12)


def stability(points, forbidden, assignments, k, seed, resamples, fraction, maximum_iterations=20):
    details = []
    scores = []
    for iteration in range(resamples):
        sampled = [identity for identity in sorted(points) if _hash(seed, iteration, identity) % 10000 < fraction]
        if len(sampled) < k:
            details.append({'iteration': iteration, 'sampled': len(sampled), 'skipped': 'too_few_units'})
            continue
        subset = {identity: points[identity] for identity in sampled}
        local = [(left, right) for left, right in forbidden if left in subset and right in subset]
        solution = solve_best(subset, k, local, seed + ':resample', restarts=2, maximum_iterations=maximum_iterations)
        if not solution['feasible']:
            details.append({'iteration': iteration, 'sampled': len(sampled), 'skipped': solution['reason']})
            continue
        score = adjusted_rand(assignments, solution['assignments'])
        if score is None:
            details.append({'iteration': iteration, 'sampled': len(sampled), 'skipped': 'degenerate_partition'})
            continue
        scores.append(score)
        details.append({'iteration': iteration, 'sampled': len(sampled), 'ari': score})
    return {'resamples': resamples, 'completed': len(scores), 'mean_ari': round(sum(scores) / len(scores), 12) if scores else None, 'iterations': details}


def settings(value):
    if not isinstance(value, dict):
        fail('topic_definition', 'Topic definition must be an object')
    allowed = {'resamples', 'sample_fraction', 'minimum_similarity', 'vector', 'maximum_terms', 'representatives', 'boundary_documents', 'document_counterfactual', 'restarts', 'greedy_limit', 'maximum_units', 'maximum_iterations'}
    shape(value, {'candidates', 'seed'}, allowed)
    candidates = value['candidates']
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 20:
        fail('topic_candidates', 'Topic candidates must be a bounded list')
    for candidate in candidates:
        integer(candidate, 'topic_candidate', 1, 50)
    if candidates != sorted(set(candidates)):
        fail('topic_candidates', 'Topic candidates must be unique and increasing')
    seed = value['seed']
    from corpus_gate.errors import identifier
    identifier(seed, 'seed')
    fraction = value.get('sample_fraction', 8000)
    integer(fraction, 'sample_fraction', 1000, 10000)
    similarity = value.get('minimum_similarity', 0.35)
    if type(similarity) not in (int, float) or isinstance(similarity, bool) or not 0 <= similarity <= 1:
        fail('topic_similarity', 'Minimum similarity must be between zero and one')
    counterfactual = value.get('document_counterfactual', True)
    if type(counterfactual) is not bool:
        fail('topic_counterfactual', 'Document counterfactual flag must be boolean')
    from corpus_gate.topics import vectors as vector_settings
    return {
        'candidates': candidates,
        'seed': seed,
        'resamples': integer(value.get('resamples', 8), 'resamples', 1, 50),
        'sample_fraction': fraction,
        'minimum_similarity': round(float(similarity), 12),
        'vector': vector_settings.settings(value.get('vector', {})),
        'maximum_terms': integer(value.get('maximum_terms', 20), 'maximum_terms', 1, 100),
        'representatives': integer(value.get('representatives', 5), 'representatives', 1, 50),
        'boundary_documents': integer(value.get('boundary_documents', 10), 'boundary_documents', 0, 100),
        'document_counterfactual': counterfactual,
        'restarts': integer(value.get('restarts', 3), 'restarts', 1, 20),
        'greedy_limit': integer(value.get('greedy_limit', 50000), 'greedy_limit', 100, 1000000),
        'maximum_units': integer(value.get('maximum_units', 2000), 'maximum_units', 2, 10000),
        'maximum_iterations': integer(value.get('maximum_iterations', 50), 'maximum_iterations', 1, 500),
    }
