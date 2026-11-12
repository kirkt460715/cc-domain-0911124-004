import json
from collections import Counter
from corpus_gate.errors import identifier, canonical, digest, fail
from corpus_gate import documents
from corpus_gate.ingest import sources
from corpus_gate.dedup import clusters
from corpus_gate.text.normalize import tokens
from corpus_gate.topics import vectors, clustering
from corpus_gate.topics.constraints import UnionFind


def counts_of(text):
    return Counter(token['text'].casefold() for token in tokens(text))


def participants(db, experiment_id):
    """Documents whose assignment belongs to an experiment.

    The frozen partition and documents added later through incremental
    classification are both reported, but only the frozen partition is used
    when comparing two experiments.
    """
    experiment = db.one('SELECT experiment_id FROM topic_experiments WHERE experiment_id=?', (experiment_id,))
    if experiment is None:
        fail('topic_experiment_missing', 'Topic experiment does not exist')
    rows = db.all('SELECT document_id,state,topic_id,origin,payload FROM topic_assignments WHERE experiment_id=? ORDER BY document_id', (experiment_id,))
    frozen = [row for row in rows if row['origin'] in {'representative', 'duplicate_inherited'}]
    incremental = [row for row in rows if row['origin'] in {'incremental', 'constraint_forced'}]
    return {'frozen': frozen, 'incremental': incremental}


def all_active_rules(db):
    return db.all("SELECT left_id,right_id,kind,revision,state,reason,actor FROM topic_constraints WHERE state='active' ORDER BY left_id,right_id")


def active_rules(db, present):
    return [row for row in all_active_rules(db) if row['left_id'] in present and row['right_id'] in present]


def ignored_rules(db, present):
    return [{**row, 'reason': 'document_outside_cluster_run'} for row in all_active_rules(db) if row['left_id'] not in present or row['right_id'] not in present]


class Plan:
    """One frozen topic exploration over an existing duplicate-cluster run.

    A duplicate cluster is one atomic, weighted voting unit.  Editor ``same``
    rules join atomic units; ``different`` rules forbid two resulting units
    from sharing a topic.  Conflicts are checked after all same-links are
    resolved, so rule arrival order never changes the verdict.
    """

    def __init__(self, db, cluster, context, settings):
        self.db = db
        self.cluster = cluster
        self.context = context
        self.settings = settings
        self.cluster_of = {}
        self.representative_of = {}
        self.groups = clusters.groups(db, cluster['run_id'])
        for group in self.groups:
            for identity in group['members']:
                self.cluster_of[identity] = group['cluster_id']
            if group['representative']:
                self.representative_of[group['cluster_id']] = group['representative']
        self.rows = active_rules(db, set(self.cluster_of))
        self.ignored = ignored_rules(db, set(self.cluster_of))
        self.contradictions = []
        self.excluded = []
        self.counts = {}
        self.units = []
        self.unit_of = {}
        self.points = {}
        self.forbidden = []
        self.document_maps = {}
        self.model = vectors.build_counts([])
        self.selection = None
        self.topics = []
        self.centers_by_public = {}
        self.build()

    # ---- construction ---------------------------------------------------
    def build(self):
        units = UnionFind(set(self.cluster_of))
        for group in self.groups:
            for member in group['members'][1:]:
                units.union(group['members'][0], member)
        for rule in self.rows:
            if rule['kind'] == 'same':
                units.union(rule['left_id'], rule['right_id'])
        roots = {}
        for identity in sorted(self.cluster_of):
            roots.setdefault(units.root(identity), []).append(identity)
        for rule in self.rows:
            if rule['kind'] != 'different':
                continue
            if units.root(rule['left_id']) == units.root(rule['right_id']):
                same_cluster = self.cluster_of[rule['left_id']] == self.cluster_of[rule['right_id']]
                self.contradictions.append({'left_id': rule['left_id'], 'right_id': rule['right_id'], 'reason': 'duplicate_cluster_indivisible' if same_cluster else 'same_and_different_cycle'})
        if self.contradictions:
            return
        self._unit_rows(roots)
        if not self.units:
            return
        self.model = vectors.build_counts([(unit['unit_id'], unit['term_counts']) for unit in self.units if unit['usable']])
        for identity in sorted(self.counts):
            self.document_maps[identity] = vectors.as_map(vectors.transform_counts(self.counts[identity], self.model))
        for unit in self.units:
            mapping = self._unit_centroid(unit)
            if mapping:
                self.points[unit['unit_id']] = mapping
        forbidden = set()
        for rule in self.rows:
            if rule['kind'] != 'different':
                continue
            left = self.unit_of.get(rule['left_id'])
            right = self.unit_of.get(rule['right_id'])
            if left is None or right is None or left == right:
                continue
            forbidden.add(tuple(sorted([left, right])))
        self.forbidden = sorted(forbidden)

    def _unit_rows(self, roots):
        for root in sorted(roots, key=lambda key: sorted(roots[key])[0]):
            members = sorted(roots[root])
            unit_id = 'unit-' + digest(members)
            included, usable, short = [], [], []
            tally = Counter()
            for identity in members:
                document = documents.get(self.db, identity)
                decision = sources.authorize(self.db, document['source_id'], self.context, exporting=True)
                self.unit_of[identity] = unit_id
                if not decision['allowed']:
                    self.excluded.append({'document_id': identity, 'cluster_id': self.cluster_of[identity], 'unit_id': unit_id, 'reason': 'source_authorization', 'details': decision['reasons']})
                    continue
                included.append(identity)
                term_counts = counts_of(document['normalized'])
                self.counts[identity] = term_counts
                if document['metrics']['tokens'] < self.settings['vector']['minimum_document_tokens']:
                    short.append(identity)
                    continue
                usable.append(identity)
                tally.update(term_counts)
            representative = next((self.representative_of[self.cluster_of[identity]] for identity in included if self.representative_of.get(self.cluster_of[identity]) in members), None)
            self.units.append({'unit_id': unit_id, 'document_ids': members, 'cluster_ids': sorted({self.cluster_of[identity] for identity in members}), 'included': included, 'usable': usable, 'short': short, 'term_counts': tally, 'representative': representative})

    def _unit_centroid(self, unit):
        centroids = []
        for cluster_id in unit['cluster_ids']:
            mappings = [self.document_maps[identity] for identity in unit['usable'] if self.counts.get(identity) is not None and self.cluster_of[identity] == cluster_id]
            mappings = [mapping for mapping in mappings if mapping]
            if not mappings:
                continue
            keys = set().union(*mappings)
            centroids.append({key: sum(mapping.get(key, 0.0) for mapping in mappings) / len(mappings) for key in keys})
        if not centroids:
            return None
        keys = set().union(*centroids)
        centre = {key: sum(centroid.get(key, 0.0) for centroid in centroids) / len(centroids) for key in keys}
        norm = sum(value * value for value in centre.values()) ** 0.5
        if not norm:
            return None
        return {key: value / norm for key, value in centre.items()}

    # ---- candidate evaluation -------------------------------------------
    def evaluate(self):
        candidates = []
        if self.contradictions:
            return [{'k': k, 'feasible': False, 'reason': 'constraint_contradiction'} for k in self.settings['candidates']], {'selected_k': None, 'policy': 'Editor constraints are hard requirements; contradictory constraints are reported before any topic count is attempted'}
        if len(self.points) <= 1:
            reason = 'single_unit' if len(self.points) == 1 else 'empty_vocabulary'
            return [{'k': k, 'feasible': False, 'reason': reason, 'available_units': len(self.points)} for k in self.settings['candidates']], {'selected_k': None, 'policy': 'A single effective duplicate cluster cannot define a topic comparison; the cluster is reported as one indivisible unit'}
        for k in self.settings['candidates']:
            entry = {'k': k, 'feasible': False, 'reason': None}
            if not self.points:
                entry['reason'] = 'empty_vocabulary'
                candidates.append(entry)
                continue
            if k > len(self.points):
                entry['reason'] = 'not_enough_units'
                entry['available_units'] = len(self.points)
                entry['available_duplicate_clusters'] = len(self.points)
                candidates.append(entry)
                continue
            solution = clustering.solve_best(self.points, k, self.forbidden, self.settings['seed'], self.settings['restarts'], self.settings['maximum_iterations'])
            if not solution['feasible']:
                entry['reason'] = solution['reason']
                entry['violations'] = solution.get('violations')
                entry['empty_topics'] = solution.get('empty_topics')
                candidates.append(entry)
                continue
            entry.update({'feasible': True, 'iterations': solution['iterations'], 'distortion': solution['distortion'], 'silhouette': clustering.silhouette(self.points, solution['assignments']), 'stability': clustering.stability(self.points, self.forbidden, solution['assignments'], k, self.settings['seed'], self.settings['resamples'], self.settings['sample_fraction'], self.settings['maximum_iterations']), 'topic_sizes': {str(key): value for key, value in sorted(Counter(solution['assignments'].values()).items())}, 'minimum_topic_units': min(Counter(solution['assignments'].values()).values())})
            entry['_assignments'] = solution['assignments']
            entry['_centers'] = solution['centers']
            candidates.append(entry)
        policy = 'Highest mean adjusted Rand index across fixed-seed resamples, then highest silhouette, then lowest distortion. Distortion alone never selects a topic count.'
        feasible = [row for row in candidates if row['feasible']]
        rationale = {'selected_k': None, 'policy': policy}
        if not feasible:
            return candidates, rationale
        ready = [row for row in feasible if row['silhouette'] is not None and row['stability']['mean_ari'] is not None]
        pool = ready or feasible
        def rank(row):
            mean = row['stability']['mean_ari']
            silhouette = row['silhouette']
            return (mean is not None, mean if mean is not None else -1.0, silhouette is not None, silhouette if silhouette is not None else -1.0, -row['distortion'], -row['k'])
        chosen = max(pool, key=rank)
        minimum = min(feasible, key=lambda row: (row['distortion'], row['k']))
        rationale.update({'selected_k': chosen['k'], 'stability_ari': chosen['stability']['mean_ari'], 'silhouette': chosen['silhouette'], 'distortion': chosen['distortion'], 'minimum_distortion_k': minimum['k'], 'distortion_optimum_selected': minimum['k'] == chosen['k'], 'selection_ready': chosen in ready, 'degenerate_single_topic': chosen['k'] == 1})
        self.selection = chosen
        return candidates, rationale

    # ---- topic views ----------------------------------------------------
    def topics_for(self):
        chosen = self.selection
        centers = {index: center for index, center in enumerate(chosen['_centers'])}
        terms = {}
        for topic in sorted(centers):
            rows = [{'term': self.model['terms'][position], 'weight': round(weight, 12), 'document_frequency': self.model['document_frequency'][position], 'idf': self.model['idf'][position]} for position, weight in centers[topic].items()]
            rows.sort(key=lambda row: (-row['weight'], row['term']))
            terms[topic] = rows[:self.settings['maximum_terms']]
        def signature(topic):
            return (tuple((-row['weight'], row['term']) for row in terms[topic]), tuple(sorted(self._units_for(topic))))
        ordered = sorted(centers, key=signature)
        public = {topic: 'topic-' + str(position + 1).rjust(2, '0') for position, topic in enumerate(ordered)}
        topics = []
        for topic in ordered:
            units = self._units_for(topic)
            members = sorted(identity for unit_id in units for identity in self._unit_by_id(unit_id)['included'])
            tally = Counter()
            entries = []
            boundaries = []
            for identity in members:
                document = documents.get(self.db, identity)
                tally[document['source_id']] += 1
                mapping = self.document_maps.get(identity, {})
                own = clustering.cosine(mapping, centers[topic])
                nearest, nearest_topic = -1.0, None
                for other in centers:
                    if other == topic:
                        continue
                    score = clustering.cosine(mapping, centers[other])
                    if score > nearest:
                        nearest, nearest_topic = score, other
                entry = {'document_id': identity, 'source_id': document['source_id'], 'cluster_id': self.cluster_of[identity], 'unit_id': self.unit_of[identity], 'similarity': round(own, 12), 'nearest_other_similarity': round(nearest, 12) if nearest_topic is not None else None, 'nearest_topic': public.get(nearest_topic), 'repost': not bool(self.representative_of.get(self.cluster_of[identity]) == identity)}
                if not mapping:
                    entry['boundary_reason'] = 'no_known_terms'
                elif nearest_topic is not None and nearest >= own:
                    entry['boundary_reason'] = 'closer_to_other_topic'
                elif own < self.settings['minimum_similarity']:
                    entry['boundary_reason'] = 'below_minimum_similarity'
                entries.append(entry)
                if entry.get('boundary_reason'):
                    boundaries.append({'topic_id': public[topic], **entry})
            entries.sort(key=lambda row: (-row['similarity'], row['document_id']))
            representatives = []
            seen = set()
            for entry in entries:
                if entry['unit_id'] in seen:
                    continue
                seen.add(entry['unit_id'])
                representatives.append(entry['document_id'])
                if len(representatives) >= self.settings['representatives']:
                    break
            boundaries.sort(key=lambda row: (-row['similarity'], row['document_id']))
            term_rows = terms[topic]
            index = {term: position for position, term in enumerate(self.model['terms'])}
            for row in term_rows:
                position = index[row['term']]
                row['topic_documents'] = sum(1 for identity in members if self.document_maps.get(identity, {}).get(position, 0.0) > 0.0)
                row['unit_document_frequency'] = self.model['document_frequency'][position]
            cluster_ids = sorted({self.cluster_of[identity] for identity in members})
            reposts = len(members) - len({self.cluster_of[identity] for identity in members})
            topics.append({
                'topic_id': public[topic],
                'internal_label': topic,
                'units': units,
                'cluster_ids': cluster_ids,
                'documents': members,
                'representatives': representatives,
                'terms': term_rows,
                'sources': dict(sorted(tally.items())),
                'sizes': {'documents': len(members), 'units': len(units), 'clusters': len(cluster_ids), 'sources': len(tally)},
                'repost_documents': reposts,
                'boundary_documents': boundaries[:self.settings['boundary_documents']] if self.settings['boundary_documents'] else [],
                'members': entries,
            })
        self.topics = topics
        self.centers_by_public = {public[topic]: centers[topic] for topic in ordered}
        self.centers_materialized = {public[topic]: clustering._materialize(centers[topic]) for topic in ordered}
        self.internal_public = public
        return topics, public

    def _units_for(self, topic):
        if not hasattr(self, '_assignments'):
            return []
        return sorted(unit_id for unit_id, value in self._assignments.items() if value == topic and unit_id in self.points)

    def _unit_by_id(self, unit_id):
        return next(unit for unit in self.units if unit['unit_id'] == unit_id)

    def topic_for_document(self, identity):
        unit_id = self.unit_of.get(identity)
        if unit_id is None or not hasattr(self, '_assignments'):
            return None
        topic = self._assignments.get(unit_id)
        if topic is None:
            return None
        return next((row['topic_id'] for row in self.topics if row['internal_label'] == topic), None)

    def bind(self):
        self._assignments = dict(self.selection['_assignments'])
        topics, public = self.topics_for()
        self.public = public
        return topics

    # ---- assignments ----------------------------------------------------
    def assignment_rows(self):
        rows = []
        for member in self.cluster['members']:
            identity = member['document_id']
            unit = next((unit for unit in self.units if unit['unit_id'] == self.unit_of.get(identity)), None)
            topic_id = self.topic_for_document(identity)
            if self.contradictions:
                state, reason, origin = 'unassigned', 'constraint_contradiction', 'blocked'
            elif unit is None:
                state, reason, origin = 'unassigned', 'not_in_cluster', 'absent'
            elif identity not in unit['included']:
                state, reason, origin = 'unassigned', 'source_authorization', 'excluded'
            elif identity in unit['short']:
                state, reason, origin = 'unassigned', 'short_text', 'short_text'
            elif not self.model['terms']:
                state, reason, origin = 'unassigned', 'empty_vocabulary', 'empty_vocabulary'
            elif topic_id is None:
                state, reason, origin = 'unassigned', 'no_known_terms', 'no_known_terms'
            else:
                state, reason, origin = 'assigned', None, 'representative' if member['representative'] else 'duplicate_inherited'
            vector = vectors.transform_counts(self.counts[identity], self.model) if identity in self.counts else {'tokens': 0, 'known_tokens': 0, 'unseen_terms': [], 'unseen': 0}
            rows.append({'document_id': identity, 'cluster_id': member['cluster_id'], 'unit_id': unit['unit_id'] if unit else None, 'topic_id': topic_id, 'state': state, 'reason': reason, 'origin': origin, 'representative': bool(member['representative']), 'unseen_terms': vector['unseen_terms'], 'known_fraction': round(vector['known_tokens'] / vector['tokens'], 12) if vector['tokens'] else 0})
        return rows

    # ---- deduplication comparison ---------------------------------------
    def dedup_effect(self):
        included = [identity for unit in self.units for identity in unit['included']]
        units_total = len([unit for unit in self.units if unit['included']])
        documents_total = len(included)
        duplicates = documents_total - units_total
        forbidden_documents = set()
        for left, right in self.forbidden:
            left_unit = self._unit_by_id(left)
            right_unit = self._unit_by_id(right)
            for a in left_unit['included']:
                for b in right_unit['included']:
                    if a != b:
                        forbidden_documents.add(tuple(sorted([a, b])))
        document_points = {identity: mapping for identity, mapping in self.document_maps.items() if mapping}
        counterfactual = None
        k = self.selection['k']
        if k <= len(document_points):
            solution = clustering.solve_best(document_points, k, sorted(forbidden_documents), self.settings['seed'] + ':equal-weight-documents', self.settings['restarts'], self.settings['maximum_iterations'])
            if solution['feasible']:
                weighted = {identity: self.topic_for_document(identity) for identity in document_points}
                counterfactual = {'feasible': True, 'k': k, 'adjusted_rand': clustering.adjusted_rand(weighted, solution['assignments']), 'topic_sizes': {str(key): value for key, value in sorted(Counter(solution['assignments'].values()).items())}, 'silhouette': clustering.silhouette(document_points, solution['assignments']), 'interpretation': 'Equal-weight reposts are a counterfactual only; the frozen weighted experiment remains authoritative'}
            else:
                counterfactual = {'feasible': False, 'reason': solution['reason'], 'interpretation': 'Counting each repost as an equal document cannot reproduce the selected topics, which is the repost flooding risk'}
        if counterfactual is None:
            counterfactual = {'feasible': False, 'reason': 'not_enough_documents', 'interpretation': 'Equal-weight document comparison needs at least as many documents as topics'}
        largest = max(self.units, key=lambda unit: (len(unit['included']), unit['unit_id']), default=None)
        amplification = None
        if largest is not None and units_total:
            amplification = {'unit_id': largest['unit_id'], 'documents': len(largest['included']), 'cluster_ids': largest['cluster_ids'], 'document_share': round(len(largest['included']) / documents_total, 12) if documents_total else None, 'weighted_vote_share': round(1 / units_total, 12), 'inflation_factor': round(len(largest['included']) * units_total / documents_total, 12) if documents_total else None}
        votes = {row['topic_id']: row['sizes']['units'] for row in self.topics}
        document_sizes = {row['topic_id']: row['sizes']['documents'] for row in self.topics}
        before_after = {'weighted_votes': votes, 'document_counts': document_sizes, 'difference': {topic: document_sizes[topic] - votes[topic] for topic in document_sizes}}
        return {
            'weighting': 'one_vote_per_duplicate_cluster',
            'documents': documents_total,
            'units': units_total,
            'duplicate_documents': duplicates,
            'repost_share': round(duplicates / documents_total, 12) if documents_total else None,
            'largest_unit_documents': max((len(unit['included']) for unit in self.units), default=0),
            'largest_unit': amplification,
            'topic_sizes_by_documents': document_sizes,
            'topic_sizes_by_units': votes,
            'repost_effect': before_after,
            'equal_weight_counterfactual': counterfactual,
        }

    def diagnostics(self):
        return {
            'cluster_members': len(self.cluster['members']),
            'documents': sum(len(unit['included']) for unit in self.units),
            'units': len(self.points),
            'atomic_units': len([unit for unit in self.units if unit['included']]),
            'duplicate_documents': sum(len(unit['included']) for unit in self.units) - len([unit for unit in self.units if unit['included']]),
            'short_documents': sum(len(unit['short']) for unit in self.units),
            'excluded_documents': len(self.excluded),
            'empty_units': len([unit for unit in self.units if unit['included'] and unit['unit_id'] not in self.points]),
            'unseen_terms': sum(vectors.transform_counts(self.counts[identity], self.model)['unseen'] for identity in self.counts),
            'vocabulary_terms': len(self.model['terms']),
            'constraint_contradictions': len(self.contradictions),
            'selected_k': self.selection['k'] if self.selection else None,
            'repost_weighting': 'one_vote_per_duplicate_cluster',
        }


def _payload_state(plan, candidates, rationale):
    if plan.contradictions:
        return 'infeasible', 'constraint_contradiction'
    included = sum(len(unit['included']) for unit in plan.units)
    if not included:
        return 'insufficient', 'no_authorized_documents'
    if not plan.model['terms'] or not plan.points:
        return 'insufficient', 'empty_vocabulary'
    if plan.selection is not None:
        return 'complete', None
    reasons = {row.get('reason') for row in candidates}
    if reasons and reasons <= {'not_enough_units', 'empty_vocabulary', 'single_unit'}:
        if 'single_unit' in reasons or len(plan.points) <= 1:
            return 'insufficient', 'single_unit'
        return 'insufficient', 'not_enough_units' if 'not_enough_units' in reasons else 'empty_vocabulary'
    for reason in ('constraint_unsatisfied', 'empty_topic', 'not_enough_units', 'unknown_constraint_feasibility'):
        if reason in reasons:
            return 'infeasible', reason
    return 'infeasible', sorted(reasons)[0] if reasons else 'no_candidate'


def run(db, experiment_id, cluster_run_id, context, definition):
    identifier(experiment_id)
    settings = clustering.settings(definition)
    context = sources.request(context)
    with db.transaction():
        if db.one('SELECT experiment_id FROM topic_experiments WHERE experiment_id=?', (experiment_id,)):
            fail('topic_experiment_exists', 'Topic experiment identifiers are immutable')
        cluster = clusters.get(db, cluster_run_id)
        if not cluster['fresh']:
            fail('topic_cluster_stale', 'Recompute duplicate clusters before topic exploration')
        plan = Plan(db, cluster, context, settings)
        if len(plan.points) > settings['maximum_units']:
            fail('topic_capacity', 'Effective topic units exceed the configured in-memory limit', units=len(plan.points), maximum_units=settings['maximum_units'])
        candidates, rationale = plan.evaluate()
        state, reason = _payload_state(plan, candidates, rationale)
        if state == 'complete':
            plan.bind()
        visible = [{key: value for key, value in row.items() if not key.startswith('_')} for row in candidates]
        payload = {
            'schema': 'corpus-gate-topic-experiment',
            'version': 1,
            'experiment_id': experiment_id,
            'state': state,
            'reason': reason,
            'cluster_run_id': cluster_run_id,
            'cluster_digest': cluster['digest'],
            'corpus_revision': db.revision(),
            'context': context,
            'definition': settings,
            'vocabulary': {'digest': plan.model['digest'], 'documents': plan.model['documents'], 'options': plan.model['options'], 'candidates': plan.model['candidates'], 'terms': plan.model['terms'], 'document_frequency': plan.model['document_frequency'], 'idf': plan.model['idf'], 'discarded': plan.model['discarded'], 'discarded_counts': plan.model['discarded_counts']},
            'units': [{'unit_id': unit['unit_id'], 'documents': unit['document_ids'], 'cluster_ids': unit['cluster_ids'], 'included': unit['included'], 'short': unit['short'], 'topic_id': plan.topic_for_document(next((identity for identity in unit['included'] if identity not in unit['short']), None)) if state == 'complete' else None} for unit in plan.units],
            'constraints': {'rules': plan.rows, 'forbidden_unit_pairs': plan.forbidden, 'contradictions': plan.contradictions, 'ignored_rules': plan.ignored, 'forced_same_units': sorted({plan.unit_of.get(rule['left_id']) for rule in plan.rows if rule['kind'] == 'same'} & {plan.unit_of.get(rule['right_id']) for rule in plan.rows if rule['kind'] == 'same'})},
            'candidates': visible,
            'selection': rationale,
            'feasible_topic_counts': [row['k'] for row in visible if row['feasible']],
            'infeasible_topic_counts': [{'k': row['k'], 'reason': row['reason']} for row in visible if not row['feasible']],
            'topics': plan.topics,
            'assignments': {row['document_id']: row['topic_id'] for row in plan.assignment_rows() if row['state'] == 'assigned'},
            'excluded_documents': plan.excluded,
            'unassigned_documents': [{'document_id': row['document_id'], 'cluster_id': row['cluster_id'], 'unit_id': row['unit_id'], 'reason': row['reason']} for row in plan.assignment_rows() if row['state'] != 'assigned'],
            'deduplication': plan.dedup_effect() if state == 'complete' else {},
            'diagnostics': plan.diagnostics(),
            'topic_centers': plan.centers_materialized if state == 'complete' else {},
        }
        sha = digest(payload)
        db.execute('INSERT INTO topic_experiments VALUES(?,?,?,?,?,?,?,?,?,?,?)', (experiment_id, db.revision(), cluster_run_id, canonical(context), canonical(settings), state, reason, plan.model['digest'], sha, canonical(payload), db.clock()))
        for row in plan.assignment_rows():
            record = dict(row)
            row_sha = digest(record)
            db.execute('INSERT INTO topic_assignments VALUES(?,?,?,?,?,?,?,?,?)', (experiment_id, row['document_id'], row['state'], row['topic_id'], row['reason'], row['origin'], canonical(record), row_sha, db.clock()))

        if plan.ignored:
            db.log('topic.constraints_ignored', {'experiment_id': experiment_id, 'count': len(plan.ignored), 'reason': 'document_outside_cluster_run'})
        db.log('topic.experiment', {'experiment_id': experiment_id, 'state': state, 'topics': len(plan.topics), 'selected_k': rationale.get('selected_k')})
        return get(db, experiment_id)


def get(db, experiment_id):
    row = db.one('SELECT * FROM topic_experiments WHERE experiment_id=?', (experiment_id,))
    if row is None:
        fail('topic_experiment_missing', 'Topic experiment does not exist')
    payload = json.loads(row['payload'])
    if digest(payload) != row['digest']:
        fail('topic_experiment_corrupt', 'Topic experiment payload digest differs')
    assignments = db.all('SELECT * FROM topic_assignments WHERE experiment_id=? ORDER BY document_id', (experiment_id,))
    for assignment in assignments:
        stored = json.loads(assignment['payload'])
        if digest(stored) != assignment['digest'] or stored['document_id'] != assignment['document_id']:
            fail('topic_assignment_corrupt', 'Stored topic assignment digest differs')
        assignment['payload'] = stored
    payload['state'] = row['state']
    payload['reason'] = row['reason']
    payload['created_at'] = row['created_at']
    payload['assignment_records'] = assignments
    payload['assignments_detail'] = [assignment for assignment in assignments if assignment['origin'] not in {'incremental', 'constraint_forced'}]
    payload['incremental_assignments'] = [assignment['payload'] for assignment in assignments if assignment['origin'] in {'incremental', 'constraint_forced'}]
    payload['assignments_table'] = [{'document_id': assignment['document_id'], 'state': assignment['state'], 'topic_id': assignment['topic_id'], 'reason': assignment['reason'], 'origin': assignment['origin'], 'cluster_id': assignment['payload'].get('cluster_id'), 'unit_id': assignment['payload'].get('unit_id')} for assignment in assignments]
    merged = dict(payload['assignments'])
    for row in payload['incremental_assignments']:
        if row['state'] == 'assigned' and row.get('topic_id'):
            merged[row['document_id']] = row['topic_id']
    payload['assignments_with_incremental'] = merged
    payload['fresh'] = payload['corpus_revision'] == db.revision()
    payload['invalid_reasons'] = validity(db, payload, assignments)
    payload['valid'] = not payload['invalid_reasons']
    payload['participants'] = {'frozen': len([row for row in assignments if row['origin'] in {'representative', 'duplicate_inherited'}]), 'incremental': len([row for row in assignments if row['origin'] in {'incremental', 'constraint_forced'}])}
    return payload


def validity(db, payload, assignments):
    reasons = []
    cluster = db.one('SELECT corpus_revision,digest FROM cluster_runs WHERE run_id=?', (payload['cluster_run_id'],))
    if cluster is None:
        reasons.append({'code': 'cluster_missing', 'cluster_run_id': payload['cluster_run_id']})
    elif cluster['digest'] != payload['cluster_digest']:
        reasons.append({'code': 'cluster_changed', 'cluster_run_id': payload['cluster_run_id']})
    for assignment in assignments:
        if assignment['origin'] in {'incremental', 'constraint_forced'}:
            # Incremental documents are a later, separate layer; their own
            # authorization is re-checked when they are classified.
            continue
        row = assignment['payload']
        if row['state'] != 'assigned':
            continue
        document = db.one('SELECT state,source_id FROM documents WHERE document_id=?', (row['document_id'],))
        if document is None:
            reasons.append({'code': 'document_missing', 'document_id': row['document_id']})
            continue
        if document['state'] != 'accepted':
            reasons.append({'code': 'document_state', 'document_id': row['document_id'], 'state': document['state']})
        decision = sources.authorize(db, document['source_id'], payload['context'], exporting=True)
        if not decision['allowed']:
            reasons.append({'code': 'source_authorization', 'document_id': row['document_id'], 'source_id': document['source_id'], 'reasons': decision['reasons']})
    return reasons


def listing(db):
    result = []
    for row in db.all('SELECT experiment_id FROM topic_experiments ORDER BY created_at,experiment_id'):
        try:
            payload = get(db, row['experiment_id'])
            result.append({'experiment_id': row['experiment_id'], 'state': payload['state'], 'reason': payload['reason'], 'selected_k': payload['selection'].get('selected_k'), 'topics': len(payload['topics']), 'fresh': payload['fresh'], 'valid': payload['valid'], 'invalid_reasons': payload['invalid_reasons']})
        except Exception as exc:
            result.append({'experiment_id': row['experiment_id'], 'state': 'corrupt', 'fresh': False, 'valid': False, 'invalid_reasons': [{'code': 'corrupt', 'message': str(exc)}]})
    return result


def classify(db, experiment_id, document_ids):
    if not isinstance(document_ids, list) or not 1 <= len(document_ids) <= 100:
        fail('topic_classify_documents', 'Topic classification requires one to one hundred document identifiers')
    for identity in document_ids:
        identifier(identity)
    with db.transaction():
        payload = get(db, experiment_id)
        if payload['state'] != 'complete':
            fail('topic_experiment_unusable', 'Only a completed topic experiment can classify documents', state=payload['state'], reason=payload['reason'])
        model = {'terms': payload['vocabulary']['terms'], 'document_frequency': payload['vocabulary']['document_frequency'], 'idf': payload['vocabulary']['idf']}
        centers = {topic_id: {position: weight for position, weight in zip(center['indices'], center['weights'])} for topic_id, center in payload['topic_centers'].items()}
        if not centers:
            fail('topic_experiment_unusable', 'Frozen topic centers are unavailable')
        frozen = {row['document_id'] for row in payload['assignments_detail']}
        incremental = {row['document_id'] for row in payload['incremental_assignments']}
        present = frozen | incremental
        rules = active_rules(db, present | set(document_ids))
        known_topic = {row['document_id']: row['topic_id'] for row in payload['assignment_records'] if row['state'] == 'assigned'}
        results = []
        for identity in document_ids:
            if identity in frozen:
                fail('topic_document_member', 'Document belongs to the frozen experiment partition', document_id=identity)
            document = documents.get(db, identity)
            reasons = []
            if document['state'] != 'accepted':
                reasons.append('document_state')
            if not document['is_head']:
                reasons.append('not_head')
            decision = sources.authorize(db, document['source_id'], payload['context'], exporting=True)
            if not decision['allowed']:
                reasons.append('source_authorization')
            term_counts = counts_of(document['normalized'])
            vector = vectors.transform_counts(term_counts, model)
            mapping = vectors.as_map(vector)
            peers = []
            for rule in rules:
                if rule['left_id'] == identity and rule['right_id'] in present:
                    peers.append((rule['kind'], rule['right_id']))
                elif rule['right_id'] == identity and rule['left_id'] in present:
                    peers.append((rule['kind'], rule['left_id']))
            same_topics = {known_topic[peer] for kind, peer in peers if kind == 'same' and peer in known_topic}
            different_topics = {known_topic[peer] for kind, peer in peers if kind == 'different' and peer in known_topic}
            result = {'document_id': identity, 'state': 'unassigned', 'reason': None, 'topic_id': None, 'similarity': None, 'unseen_terms': vector['unseen_terms'], 'known_fraction': round(vector['known_tokens'] / vector['tokens'], 12) if vector['tokens'] else 0, 'origin': 'incremental'}
            if reasons:
                result['reason'] = reasons[0]
            elif document['metrics']['tokens'] < payload['definition']['vector']['minimum_document_tokens']:
                result['reason'] = 'short_text'
            elif len(same_topics) > 1:
                result['reason'] = 'constraint_conflict'
            elif not mapping:
                if len(same_topics) == 1:
                    forced = next(iter(same_topics))
                    result['reason'] = 'forced_without_known_terms'
                    if forced not in different_topics:
                        result['state'] = 'assigned'
                        result['topic_id'] = forced
                        result['origin'] = 'constraint_forced'
                else:
                    result['reason'] = 'no_known_terms' if vector['tokens'] else 'empty_text'
            else:
                allowed = list(centers)
                if same_topics:
                    allowed = [topic for topic in allowed if topic in same_topics]
                allowed = [topic for topic in allowed if topic not in different_topics]
                if not allowed:
                    result['reason'] = 'constraint_conflict'
                else:
                    scores = {topic: clustering.cosine(mapping, centers[topic]) for topic in allowed}
                    best = max(allowed, key=lambda topic: (scores[topic], topic))
                    result['similarity'] = round(scores[best], 12)
                    if scores[best] < payload['definition']['minimum_similarity'] and not same_topics:
                        result['reason'] = 'below_threshold'
                    else:
                        result['state'] = 'assigned'
                        result['topic_id'] = best
                        result['origin'] = 'constraint_forced' if same_topics and scores[best] < payload['definition']['minimum_similarity'] else 'incremental'
            record = dict(result)
            sha = digest(record)
            db.execute("INSERT INTO topic_assignments VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id,document_id) DO UPDATE SET state=excluded.state,topic_id=excluded.topic_id,reason=excluded.reason,origin=excluded.origin,payload=excluded.payload,digest=excluded.digest,assigned_at=excluded.assigned_at", (experiment_id, identity, result['state'], result['topic_id'], result['reason'], result['origin'], canonical(record), sha, db.clock()))
            results.append(result)
            present.add(identity)
            if result['state'] == 'assigned':
                known_topic[identity] = result['topic_id']
        db.log('topic.classified', {'experiment_id': experiment_id, 'documents': len(results), 'assigned': sum(row['state'] == 'assigned' for row in results)})
        return {'experiment_id': experiment_id, 'transformation': 'frozen_vocabulary_and_tfidf', 'refitted': False, 'results': results, 'pending_area': [row['document_id'] for row in results if row['state'] != 'assigned']}


def compare(db, before, after):
    left = get(db, before)
    right = get(db, after)
    if left['state'] != 'complete' or right['state'] != 'complete':
        fail('topic_compare_state', 'Topic comparison requires two completed experiments')
    old_topics = {topic['topic_id']: set(topic['documents']) for topic in left['topics']}
    new_topics = {topic['topic_id']: set(topic['documents']) for topic in right['topics']}
    old_terms = {topic['topic_id']: {row['term'] for row in topic['terms']} for topic in left['topics']}
    new_terms = {topic['topic_id']: {row['term'] for row in topic['terms']} for topic in right['topics']}
    overlap = {}
    for new_id in sorted(new_topics):
        for old_id in sorted(old_topics):
            overlap[(old_id, new_id)] = len(old_topics[old_id] & new_topics[new_id])
    def dominant(new_id):
        """Align a new topic to the old topic sharing most members, then terms."""
        scored = sorted(old_topics, key=lambda old_id: (-overlap[(old_id, new_id)], old_id))
        if scored and overlap[(scored[0], new_id)] > 0:
            return scored[0]
        best, best_ratio = None, 0.0
        for old_id in sorted(old_topics):
            terms = new_terms[new_id] | old_terms[old_id]
            ratio = len(new_terms[new_id] & old_terms[old_id]) / len(terms) if terms else 0.0
            if ratio > best_ratio:
                best, best_ratio = old_id, ratio
        return best if best_ratio >= 0.5 else None
    mapping = {new_id: dominant(new_id) for new_id in new_topics}
    def half(members):
        return max(1, (len(members) + 1) // 2)
    split_of = {}
    for old_id, members in old_topics.items():
        children = [new_id for new_id in sorted(new_topics) if overlap[(old_id, new_id)] >= half(new_topics[new_id])]
        if len(children) > 1:
            split_of[old_id] = children
    merge_of = {}
    for new_id, members in new_topics.items():
        parents = [old_id for old_id in sorted(old_topics) if overlap[(old_id, new_id)] >= half(old_topics[old_id])]
        if len(parents) > 1:
            merge_of[new_id] = parents
    details = []
    for old_id, members in sorted(old_topics.items()):
        if old_id in split_of:
            details.append({'topic_id': old_id, 'status': 'split', 'before_documents': len(members), 'after_topics': [{'topic_id': child, 'documents': len(new_topics[child]), 'shared': overlap[(old_id, child)]} for child in split_of[old_id]]})
        elif old_id in merge_of.get(mapping.get(old_id, ''), []):
            parents = merge_of[mapping[old_id]]
            details.append({'topic_id': old_id, 'status': 'merged', 'before_documents': len(members), 'after_topic': mapping[old_id], 'merged_with': [parent for parent in parents if parent != old_id], 'shared': overlap[(old_id, mapping[old_id])]})
        elif mapping.get(old_id) is None or all(mapping.get(new_id) != old_id for new_id in new_topics):
            details.append({'topic_id': old_id, 'status': 'dissolved', 'before_documents': len(members), 'after_topics': []})
        else:
            linked = [new_id for new_id in sorted(new_topics) if mapping.get(new_id) == old_id]
            details.append({'topic_id': old_id, 'status': 'retained', 'before_documents': len(members), 'after_topic': linked[0], 'after_documents': len(new_topics[linked[0]]), 'document_changes': len(members ^ new_topics[linked[0]])})
    merges = [{'topic_id': new_id, 'before_topics': parents, 'documents': len(new_topics[new_id]), 'shared': {old_id: overlap[(old_id, new_id)] for old_id in parents}} for new_id, parents in sorted(merge_of.items())]
    assigned_before = {row['document_id']: row.get('topic_id') for row in left['assignments_detail'] if row['state'] == 'assigned'}
    assigned_after = {row['document_id']: row.get('topic_id') for row in right['assignments_detail'] if row['state'] == 'assigned'}
    migrations = []
    moved = set()
    for identity in sorted(set(assigned_before) & set(assigned_after)):
        before_topic = assigned_before[identity]
        after_topic = assigned_after[identity]
        aligned = mapping.get(after_topic)
        if aligned is not None and aligned != before_topic:
            migrations.append({'document_id': identity, 'before_topic': before_topic, 'after_topic': after_topic, 'aligned_before_topic': aligned, 'reason': 'realigned', 'source_id': documents.get(db, identity)['source_id']})
            moved.add(identity)
        elif aligned is None and before_topic != after_topic:
            migrations.append({'document_id': identity, 'before_topic': before_topic, 'after_topic': after_topic, 'aligned_before_topic': None, 'reason': 'new_branch', 'source_id': documents.get(db, identity)['source_id']})
            moved.add(identity)
    for old_id, children in sorted(split_of.items()):
        canonical = [child for child in children if mapping.get(child) == old_id]
        primary = canonical[0] if canonical else max(children, key=lambda child: (overlap[(old_id, child)], child))
        for child in children:
            if child == primary:
                continue
            for identity in sorted(old_topics[old_id] & new_topics[child]):
                if identity in moved:
                    continue
                migrations.append({'document_id': identity, 'before_topic': old_id, 'after_topic': child, 'aligned_before_topic': mapping.get(child), 'parent_topic': old_id, 'reason': 'split_branch', 'source_id': documents.get(db, identity)['source_id']})
                moved.add(identity)
    for identity in sorted(set(assigned_before) - set(assigned_after)):
        migrations.append({'document_id': identity, 'before_topic': assigned_before[identity], 'after_topic': None, 'reason': 'left_experiment'})
    for identity in sorted(set(assigned_after) - set(assigned_before)):
        migrations.append({'document_id': identity, 'before_topic': None, 'after_topic': assigned_after[identity], 'reason': 'new_to_experiment'})
    return {
        'before': before,
        'after': after,
        'alignment': 'Topics are matched by member overlap, falling back to distinguishing terms; numeric topic identifiers are local labels and never identify content',
        'topic_map_after_to_before': mapping,
        'topics': details,
        'merges': merges,
        'migrations': migrations,
        'summary': {'retained': sum(row['status'] == 'retained' for row in details), 'split': sum(row['status'] == 'split' for row in details), 'dissolved': sum(row['status'] == 'dissolved' for row in details), 'merged': sum(row['status'] == 'merged' for row in details), 'merges': len(merges), 'document_moves': len([row for row in migrations if row['before_topic'] and row['after_topic']])},
        'label_change_is_not_content_change': True,
    }
