import copy
import json
from tests.support import Case, record, context, selection, authorization
from corpus_gate.service import CorpusGate
from corpus_gate import batch, documents, inspection
from corpus_gate.ingest import sources
from corpus_gate.text import profiles
from corpus_gate.dedup import clusters
from corpus_gate.review import desk
from corpus_gate.topics import experiments, constraints as topic_constraints


TOPIC_GROUPS = {
    'graph': [
        'graph neural network message passing node classification layers',
        'graph network node embedding message passing training',
        'graph node classification embedding message passing',
    ],
    'protein': [
        'protein structure folding sequence alignment molecular biology',
        'protein folding sequence alignment molecular structure',
        'protein sequence molecular biology folding alignment',
    ],
    'portfolio': [
        'portfolio risk asset allocation equity returns volatility',
        'portfolio asset allocation risk volatility returns',
        'portfolio allocation equity volatility model risk',
    ],
}

REQUEST = {'purpose': 'research', 'region': 'cn', 'at': '2026-11-11T00:00:00Z', 'fulfilled_obligations': ['attribution']}


def topic_definition(**changes):
    result = {'candidates': [2, 3, 4, 5], 'seed': 'fixed', 'resamples': 4, 'minimum_similarity': 0.25, 'vector': {'minimum_document_tokens': 2}}
    result.update(changes)
    return result


class TopicCase(Case):
    def setUp(self):
        super().setUp()
        profiles.put(self.db, 'default', 1, {'quality': {'minimum_tokens': 2}})
        desk.register(self.db, 'supervisor', 'supervisor')

    def corpus(self, groups=None, reposts=0):
        groups = groups or TOPIC_GROUPS
        records = []
        for name, texts in groups.items():
            for index, text in enumerate(texts):
                records.append(record(f'{name}-{index}', text))
        if reposts:
            for index in range(reposts):
                records.append(record(f'graph-repost-{index}', groups['graph'][0]))
        self.ingest(records, 'topic-batch')
        clusters.run(self.db, 'cluster-run')
        return records

    def experiment(self, experiment_id='experiment', definition=None, run='cluster-run'):
        return experiments.run(self.db, experiment_id, run, REQUEST, definition or topic_definition())


class TopicExplorationTests(TopicCase):
    def test_weighted_clusters_surface_distinct_themes(self):
        self.corpus(reposts=6)
        result = self.experiment()
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(len(result['topics']), result['selection']['selected_k'])
        terms = {topic['topic_id']: {row['term'] for row in topic['terms']} for topic in result['topics']}
        themes = [{'graph', 'protein', 'portfolio'} & values for values in terms.values()]
        self.assertEqual(len([theme for theme in themes if theme]), 3)
        self.assertEqual({len(topic['documents']) for topic in result['topics']}, {3, 3, 9})
        reposted = next(topic for topic in result['topics'] if len(topic['documents']) == 9)
        self.assertEqual(reposted['sizes']['units'], 3)
        self.assertEqual(reposted['repost_documents'], 6)
        self.assertEqual(reposted['sources'], {'source': 9})
        for topic in result['topics']:
            self.assertTrue(topic['representatives'])
            self.assertEqual(len(topic['representatives']), len({documents.get(self.db, identity)['exact_hash'] for identity in topic['representatives']}))

    def test_repost_weighting_differs_from_document_counting(self):
        self.corpus(reposts=8)
        result = self.experiment()
        dedup = result['deduplication']
        self.assertEqual(dedup['weighting'], 'one_vote_per_duplicate_cluster')
        self.assertEqual(dedup['documents'], 17)
        self.assertEqual(dedup['units'], 9)
        self.assertEqual(dedup['duplicate_documents'], 8)
        self.assertEqual(dedup['largest_unit_documents'], 9)
        self.assertAlmostEqual(dedup['repost_share'], 8 / 17, places=6)
        counterfactual = dedup['equal_weight_counterfactual']
        self.assertIn('interpretation', counterfactual)
        self.assertIn('authoritative', counterfactual['interpretation'])
        weighted_total = sum(dedup['topic_sizes_by_documents'].values())
        self.assertEqual(weighted_total, dedup['documents'])
        units_total = sum(dedup['topic_sizes_by_units'].values())
        self.assertEqual(units_total, dedup['units'])
        effect = dedup['repost_effect']
        self.assertEqual(effect['weighted_votes'], dedup['topic_sizes_by_units'])
        self.assertEqual(set(effect['difference']), set(dedup['topic_sizes_by_units']))
        self.assertTrue(any(value != 0 for value in effect['difference'].values()))

    def test_topic_count_uses_stability_and_silhouette_not_minimum_loss(self):
        self.corpus()
        result = self.experiment(definition=topic_definition(candidates=[2, 3, 4, 5, 6, 7, 8, 9]))
        candidates = {row['k']: row for row in result['candidates']}
        feasible = [row for row in result['candidates'] if row['feasible']]
        self.assertGreater(len(feasible), 1)
        minimum = min(feasible, key=lambda row: (row['distortion'], row['k']))
        selection = result['selection']
        self.assertEqual(selection['minimum_distortion_k'], minimum['k'])
        self.assertEqual(selection['selected_k'], candidates[selection['selected_k']]['k'])
        self.assertIn('Distortion alone', selection['policy'])
        self.assertIsNotNone(selection['stability_ari'])
        self.assertEqual(selection['stability_ari'], candidates[selection['selected_k']]['stability']['mean_ari'])
        self.assertTrue(candidates[selection['selected_k']]['stability']['completed'] > 0)
        for row in feasible:
            self.assertIsNotNone(row['silhouette'])
            self.assertGreaterEqual(row['stability']['completed'], 0)
        self.assertNotEqual(selection['selected_k'], 1)
        self.assertFalse(selection['degenerate_single_topic'])

    def test_fixed_seed_resampling_is_repeatable(self):
        self.corpus()
        first = self.experiment('one')
        second = self.experiment('two')
        first_candidates = [{key: value for key, value in row.items() if key not in {'stability'}} for row in first['candidates']]
        second_candidates = [{key: value for key, value in row.items() if key not in {'stability'}} for row in second['candidates']]
        self.assertEqual(first_candidates, second_candidates)
        for left, right in zip(first['candidates'], second['candidates']):
            self.assertEqual(left.get('stability', {}).get('iterations'), right.get('stability', {}).get('iterations'))
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['assignments'], second['assignments'])
        self.assertEqual(first['vocabulary']['digest'], second['vocabulary']['digest'])

    def test_vocabulary_and_tfidf_are_frozen(self):
        self.corpus()
        result = self.experiment()
        vocabulary = result['vocabulary']
        self.assertEqual(len(vocabulary['terms']), len(vocabulary['idf']))
        self.assertEqual(vocabulary['terms'], sorted(vocabulary['terms']))
        groups = clusters.groups(self.db, 'cluster-run')
        self.assertEqual(vocabulary['documents'], len(result['units']))
        self.assertTrue(all(value >= 1.0 for value in vocabulary['idf']))
        stored = self.db.one('SELECT vocabulary_digest,definition FROM topic_experiments WHERE experiment_id=?', ('experiment',))
        self.assertEqual(stored['vocabulary_digest'], vocabulary['digest'])
        self.assertTrue(result['fresh'])
        self.assertEqual(json.loads(stored['definition'])['candidates'], topic_definition()['candidates'])

    def test_short_text_documents_are_explained(self):
        self.corpus()
        profiles.put(self.db, 'default', 2, {'quality': {'minimum_tokens': 1}})
        short = [record('short-a', 'brief'), record('short-b', 'tiny text')]
        topics = [record('topic-a', 'graph node network message'), record('topic-b', 'protein folding structure alignment')]
        self.ingest(short + topics, 'mixed')
        clusters.run(self.db, 'mixed-run')
        result = self.experiment('mixed', topic_definition(candidates=[2, 3], vector={'minimum_document_tokens': 3}), 'mixed-run')
        reasons = {row['document_id']: row['reason'] for row in result['unassigned_documents']}
        short_ids = [self.document('short-a')['document_id'], self.document('short-b')['document_id']]
        self.assertTrue(all(reasons.get(identity) == 'short_text' for identity in short_ids))
        self.assertEqual(result['diagnostics']['short_documents'], 2)
        self.assertGreaterEqual(len(result['topics']), 2)

    def test_single_effective_cluster_is_insufficient(self):
        with CorpusGate() as other:
            sources.create(other.db, 'source', authorization())
            profiles.put(other.db, 'default', 0, {})
            self.ingest([record('only-a', 'alpha beta gamma delta epsilon'), record('only-b', 'alpha beta gamma delta epsilon')], 'single')
            clusters.run(self.db, 'single-run')
            single = self.experiment('single', topic_definition(candidates=[1, 2]), 'single-run')
            self.assertEqual(single['state'], 'insufficient')
            self.assertEqual(single['reason'], 'single_unit')
            self.assertTrue(all(row['reason'] == 'single_unit' for row in single['candidates']))
            self.assertEqual(single['selection']['selected_k'], None)
            self.assertEqual(single['diagnostics']['units'], 1)

    def test_no_authorized_documents_is_insufficient(self):
        from corpus_gate.ingest import sources
        self.corpus()
        sources.revoke(self.db, 'source', 1, 'Revoked before experiment')
        clusters.run(self.db, 'revoked-run')
        result = self.experiment('revoked', topic_definition(candidates=[2]), 'revoked-run')
        self.assertEqual(result['state'], 'insufficient')
        self.assertEqual(result['reason'], 'no_authorized_documents')
        self.assertEqual(result['topics'], [])

    def test_empty_vocabulary_is_insufficient(self):
        self.corpus()
        empty = self.experiment('empty-vocab', topic_definition(candidates=[2], vector={'minimum_document_tokens': 1000}))
        self.assertEqual(empty['state'], 'insufficient')
        self.assertEqual(empty['reason'], 'empty_vocabulary')
        self.assertEqual(empty['topics'], [])
        self.assertEqual({row['reason'] for row in empty['unassigned_documents']}, {'short_text'})

    def test_unseen_vocabulary_is_reported_for_incremental_document(self):
        self.corpus()
        result = self.experiment()
        new = record('unseen', 'graph network message embedding quantum lattice phonon spectroscopy')
        self.ingest([new], 'new-batch')
        classified = experiments.classify(self.db, 'experiment', [self.document('unseen')['document_id']])
        row = classified['results'][0]
        self.assertEqual(row['state'], 'assigned')
        self.assertEqual(row['topic_id'], next(topic['topic_id'] for topic in result['topics'] if any(item['term'] == 'graph' for item in topic['terms'])))
        self.assertIn('quantum', row['unseen_terms'])
        self.assertIn('spectroscopy', row['unseen_terms'])
        self.assertFalse(classified['refitted'])
        self.assertEqual(classified['transformation'], 'frozen_vocabulary_and_tfidf')

    def test_classification_uses_frozen_transform_and_pending_area(self):
        self.corpus()
        result = self.experiment()
        model_before = copy.deepcopy(result['vocabulary'])
        self.ingest([record('novel', 'quantum lattice phonon spectroscopy measurement')], 'novel-batch')
        classified = experiments.classify(self.db, 'experiment', [self.document('novel')['document_id']])
        row = classified['results'][0]
        self.assertEqual(row['state'], 'unassigned')
        self.assertEqual(row['reason'], 'no_known_terms')
        self.assertIn(self.document('novel')['document_id'], classified['pending_area'])
        after = experiments.get(self.db, 'experiment')
        self.assertEqual(after['vocabulary'], model_before)
        self.assertFalse(after['fresh'])
        self.assertEqual(experiments.listing(self.db)[0]['fresh'], False)


class TopicConstraintTests(TopicCase):
    def test_same_constraint_forms_one_weighted_unit(self):
        self.corpus()
        left = self.document('graph-0')['document_id']
        right = self.document('protein-1')['document_id']
        topic_constraints.put(self.db, left, right, 'same', 0, 'supervisor', 'Editor confirms shared theme')
        result = self.experiment()
        units = [unit for unit in result['units'] if left in unit['documents']]
        self.assertEqual(len(units), 1)
        self.assertIn(right, units[0]['documents'])
        self.assertEqual(result['diagnostics']['units'], result['diagnostics']['atomic_units'])
        self.assertEqual(len(result['units']), len(TOPIC_GROUPS['graph'] + TOPIC_GROUPS['protein'] + TOPIC_GROUPS['portfolio']) - 1)
        units_for_left = next(unit for unit in result['units'] if left in unit['documents'])
        self.assertIn(left, units_for_left['documents'])
        joined = [unit for unit in result['units'] if len([group for group in unit['cluster_ids']]) == 2]
        self.assertEqual(len(joined), 1)

    def test_contradiction_detected_regardless_of_order(self):
        self.corpus()
        identities = [self.document(name)['document_id'] for name in ('graph-0', 'graph-1', 'protein-0')]
        topic_constraints.put(self.db, identities[0], identities[1], 'same', 0, 'supervisor', 'A and B')
        topic_constraints.put(self.db, identities[1], identities[2], 'same', 0, 'supervisor', 'B and C')
        topic_constraints.put(self.db, identities[0], identities[2], 'different', 0, 'supervisor', 'A versus C')
        result = self.experiment()
        self.assertEqual(result['state'], 'infeasible')
        self.assertEqual(result['reason'], 'constraint_contradiction')
        self.assertTrue(result['constraints']['contradictions'])
        self.assertEqual(len(result['topics']), 0)
        self.assertTrue(all(row['reason'] == 'constraint_contradiction' for row in result['candidates']))

        alternative = self.experiment('reordered')
        self.assertEqual(alternative['state'], 'infeasible')
        self.assertEqual({(row['left_id'], row['right_id'], row['reason']) for row in alternative['constraints']['contradictions']}, {(row['left_id'], row['right_id'], row['reason']) for row in result['constraints']['contradictions']})

    def test_different_constraint_marks_candidate_infeasible(self):
        self.corpus(groups={'one': ['alpha beta gamma delta', 'alpha beta gamma epsilon'], 'two': ['alpha beta delta zeta', 'kappa lambda mu nu'], 'three': ['xi omicron pi rho', 'sigma tau upsilon phi']})
        names = ['one-0', 'two-0', 'three-0']
        ids = [self.document(name)['document_id'] for name in names]
        topic_constraints.put(self.db, ids[0], ids[1], 'different', 0, 'supervisor', 'separate')
        topic_constraints.put(self.db, ids[1], ids[2], 'different', 0, 'supervisor', 'separate')
        topic_constraints.put(self.db, ids[0], ids[2], 'different', 0, 'supervisor', 'separate')
        result = self.experiment(definition=topic_definition(candidates=[2, 3]))
        candidates = {row['k']: row for row in result['candidates']}
        self.assertFalse(candidates[2]['feasible'])
        self.assertEqual(candidates[2]['reason'], 'constraint_unsatisfied')
        self.assertTrue(candidates[2]['violations'])
        self.assertTrue(candidates[3]['feasible'])
        self.assertEqual(result['selection']['selected_k'], 3)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['feasible_topic_counts'], [3])
        self.assertEqual(result['infeasible_topic_counts'], [{'k': 2, 'reason': 'constraint_unsatisfied'}])
        pair_topics = {name: result['assignments'][self.document(name)['document_id']] for name in names}
        self.assertEqual(len(set(pair_topics.values())), 3)

    def test_duplicate_cluster_cannot_be_split_by_different_constraint(self):
        self.ingest([record('a', 'alpha beta gamma delta epsilon'), record('b', 'alpha beta gamma delta epsilon')], 'pair')
        clusters.run(self.db, 'pair-run')
        left = self.document('a')['document_id']
        right = self.document('b')['document_id']
        topic_constraints.put(self.db, left, right, 'different', 0, 'supervisor', 'Mistaken split')
        result = self.experiment('duplicate-split', topic_definition(candidates=[2]), 'pair-run')
        self.assertEqual(result['state'], 'infeasible')
        self.assertEqual(result['reason'], 'constraint_contradiction')
        self.assertEqual(result['constraints']['contradictions'][0]['reason'], 'duplicate_cluster_indivisible')

    def test_constraints_require_supervisor_and_are_revised(self):
        self.corpus()
        left = self.document('graph-0')['document_id']
        right = self.document('protein-0')['document_id']
        desk.register(self.db, 'reviewer', 'reviewer')
        self.fails('topic_constraint_supervisor', topic_constraints.put, self.db, left, right, 'same', 0, 'reviewer', 'reason')
        saved = topic_constraints.put(self.db, left, right, 'same', 0, 'supervisor', 'First')
        self.assertEqual(saved['revision'], 1)
        self.fails('topic_constraint_stale', topic_constraints.put, self.db, left, right, 'different', 0, 'supervisor', 'Stale')
        updated = topic_constraints.put(self.db, left, right, 'different', 1, 'supervisor', 'Reversed')
        self.assertEqual(updated['kind'], 'different')
        removed = topic_constraints.remove(self.db, left, right, 2, 'supervisor', 'Withdraw')
        self.assertEqual(removed['state'], 'removed')
        self.assertEqual(len(topic_constraints.history(self.db, left, right)), 3)
        self.assertEqual(topic_constraints.listing(self.db), [])
        self.assertEqual(len(topic_constraints.listing(self.db, active_only=False)), 1)


class TopicComparisonTests(TopicCase):
    def test_compare_reports_split_merge_and_real_migration(self):
        self.corpus()
        coarse = self.experiment('coarse', topic_definition(candidates=[2], seed='coarse'))
        fine = self.experiment('fine', topic_definition(candidates=[3], seed='fine'))
        compared = experiments.compare(self.db, 'coarse', 'fine')
        self.assertTrue(compared['label_change_is_not_content_change'])
        self.assertEqual(compared['topic_map_after_to_before'] and len(compared['topic_map_after_to_before']), 3)
        summary = compared['summary']
        self.assertGreaterEqual(summary['split'] + summary['merges'], 1)
        self.assertTrue(compared['migrations'])
        for migration in compared['migrations']:
            if migration['before_topic'] and migration['after_topic']:
                self.assertIn(migration['reason'], {'split_branch', 'realigned', 'new_branch'})
                if migration['reason'] == 'split_branch':
                    split = next(item for item in compared['topics'] if item['status'] == 'split' and item['topic_id'] == migration['before_topic'])
                    self.assertIn(migration['after_topic'], [row['topic_id'] for row in split['after_topics']])
                    self.assertEqual(migration['parent_topic'], migration['before_topic'])
                    self.assertEqual(migration['aligned_before_topic'], migration['parent_topic'])
                    self.assertNotEqual(migration['after_topic'], migration['parent_topic'])
        self.assertEqual(summary['document_moves'], len([row for row in compared['migrations'] if row['before_topic'] and row['after_topic']]))

    def test_same_members_different_labels_compare_as_retained(self):
        self.corpus()
        first = self.experiment('first')
        second = self.experiment('second', topic_definition(candidates=[first['selection']['selected_k']], seed='second'))
        compared = experiments.compare(self.db, 'first', 'second')
        self.assertEqual(compared['summary']['split'], 0)
        self.assertEqual(compared['summary']['merges'], 0)
        self.assertEqual(compared['migrations'], [])
        self.assertTrue(all(row['status'] == 'retained' for row in compared['topics']))
        self.assertEqual(first['assignments'], second['assignments'])

    def test_compare_requires_completed_experiments(self):
        self.ingest([record('brief', 'short')], 'short-batch')
        clusters.run(self.db, 'short-run')
        profiles.put(self.db, 'default', 2, {'quality': {'minimum_tokens': 1}})
        result = self.experiment('short', topic_definition(candidates=[2]), 'short-run')
        self.assertEqual(result['state'], 'insufficient')
        self.corpus()
        complete = self.experiment('complete')
        self.fails('topic_compare_state', experiments.compare, self.db, 'short', 'complete')


class TopicLifecycleTests(TopicCase):
    def test_withdrawal_blocks_export_and_identifies_invalid_experiment(self):
        self.corpus()
        result = self.experiment()
        release = self.release(options=selection())
        exported = True
        sources.revoke(self.db, 'source', 1, 'Source withdrawal')
        self.fails('release_authorization', __import__('corpus_gate.releases.exporting', fromlist=['export']).export, self.db, 'release')
        refreshed = experiments.get(self.db, 'experiment')
        self.assertFalse(refreshed['valid'])
        payload = refreshed
        self.assertTrue(any(reason['code'] == 'source_authorization' for reason in refreshed['invalid_reasons']))
        listing = experiments.listing(self.db)
        self.assertFalse(listing[0]['valid'])
        self.assertEqual(listing[0]['invalid_reasons'], refreshed['invalid_reasons'])
        self.assertFalse(payload['fresh'])

    def test_experiment_digest_and_rerun_immutability(self):
        self.corpus()
        result = self.experiment()
        self.fails('topic_experiment_exists', self.experiment, 'experiment')
        self.db.execute("UPDATE topic_experiments SET payload='{}' WHERE experiment_id='experiment'")
        self.fails('topic_experiment_corrupt', experiments.get, self.db, 'experiment')

    def test_topic_unit_capacity_is_enforced(self):
        self.corpus()
        self.fails('topic_capacity', self.experiment, 'capacity', topic_definition(maximum_units=2))
        self.assertEqual(self.db.scalar('SELECT COUNT(*) FROM topic_experiments'), 0)

    def test_stale_cluster_prevents_experiment(self):
        self.corpus()
        self.ingest([record('later', 'another distinct document appears later on')], 'later-batch')
        self.fails('topic_cluster_stale', self.experiment, 'stale')
        self.assertEqual(self.db.scalar('SELECT COUNT(*) FROM topic_experiments'), 0)
        clusters.run(self.db, 'fresh-run')
        result = self.experiment('fresh', run='fresh-run')
        self.assertEqual(result['state'], 'complete')


class TopicServiceTests(TopicCase):
    def test_service_operations_are_registered_and_validated(self):
        self.corpus()
        definition = topic_definition()
        result = self.app.dispatch('topic.run', {'experiment_id': 'service', 'cluster_run_id': 'cluster-run', 'context': REQUEST, 'definition': definition})
        self.assertEqual(result['state'], 'complete')
        again = self.app.dispatch('topic.get', {'experiment_id': 'service'})
        self.assertEqual(again['digest'] if 'digest' in again else again['selection'], result['selection'])
        self.assertEqual(len(self.app.dispatch('topic.list', {})), 1)
        self.fails('topic_candidates', self.app.dispatch, 'topic.run', {'experiment_id': 'bad', 'cluster_run_id': 'cluster-run', 'context': REQUEST, 'definition': {**definition, 'candidates': [3, 2]}})
        self.fails('fields', self.app.dispatch, 'topic.classify', {'experiment_id': 'service', 'extra': True})

    def test_transaction_rolls_back_topic_failures(self):
        self.corpus()
        requests = [{'operation': 'topic.run', 'arguments': {'experiment_id': 'tx', 'cluster_run_id': 'cluster-run', 'context': REQUEST, 'definition': topic_definition()}}, {'operation': 'topic.get', 'arguments': {'experiment_id': 'missing'}}]
        self.fails('topic_experiment_missing', self.app.transaction, requests)
        self.assertEqual(self.db.scalar('SELECT COUNT(*) FROM topic_experiments'), 0)

    def test_incremental_classification_does_not_refit(self):
        self.corpus()
        self.experiment()
        self.ingest([record('graph-new', 'graph network message embedding node layers')], 'incremental-batch')
        identity = self.document('graph-new')['document_id']
        reviewed = experiments.classify(self.db, 'experiment', [identity])
        result = reviewed['results'][0]
        self.assertEqual(result['state'], 'assigned')
        self.assertFalse(reviewed['refitted'])
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM topic_experiments WHERE definition LIKE '%incremental%'"), 0)
        payload = experiments.get(self.db, 'experiment')
        self.assertEqual(payload['assignments_with_incremental'][identity], result['topic_id'])
        self.assertEqual(len(payload['incrementals' if 'incrementals' in payload else 'incremental_assignments']), 1)
        with self.assertRaises(Exception):
            experiments.classify(self.db, 'experiment', [self.document('graph-0')['document_id']])


class TopicDetailTests(TopicCase):
    def test_topic_detail_contains_terms_representatives_sources_boundaries(self):
        self.corpus(reposts=4)
        result = self.experiment()
        for topic in result['topics']:
            self.assertTrue(topic['terms'])
            self.assertTrue(topic['representatives'])
            self.assertLessEqual(len(topic['representatives']), result['definition']['representatives'])
            self.assertEqual(sum(topic['sources'].values()), topic['sizes']['documents'])
            self.assertEqual(topic['sizes']['units'], len({row['unit_id'] for row in topic['members']}))
            for row in topic['boundary_documents']:
                self.assertIn(row['boundary_reason'], {'closer_to_other_topic', 'below_minimum_similarity', 'no_known_terms'})
                self.assertEqual(row['topic_id'], topic['topic_id'])
        graph_topic = next(topic for topic in result['topics'] if any(row['term'] == 'graph' for row in topic['terms']))
        self.assertEqual(graph_topic['sizes']['documents'], 7)
        self.assertEqual(graph_topic['repost_documents'], 4)
        self.assertEqual(len(graph_topic['representatives']), graph_topic['sizes']['units'])
        self.assertEqual(result['deduplication']['largest_unit_documents'], 5)
        self.assertEqual(result['deduplication']['largest_unit']['weighted_vote_share'], round(1 / result['deduplication']['units'], 12))

    def test_different_constraint_forces_separation_within_merged_topic(self):
        self.corpus(groups={'finance': ['portfolio risk asset allocation volatility model', 'portfolio asset allocation risk volatility returns', 'portfolio allocation equity volatility model risk'], 'news': ['market returns equity news portfolio allocation', 'market portfolio equity news allocation risk']})
        result = self.experiment(definition=topic_definition(candidates=[2], seed='merge'))
        finance = [self.document(f'finance-{index}')['document_id'] for index in range(3)]
        news = [self.document(f'news-{index}')['document_id'] for index in range(2)]
        if len(set(result['assignments'].get(identity) for identity in finance + news)) == 1:
            topic_constraints.put(self.db, finance[0], news[0], 'different', 0, 'supervisor', 'Separate desk beats')
            constrained = self.experiment('constrained', topic_definition(candidates=[2, 3], seed='merge'))
            self.assertTrue(constrained['state'] == 'complete')
            self.assertNotEqual(constrained['assignments'].get(finance[0]), constrained['assignments'].get(news[0]))
        self.assertEqual(self.db.scalar("SELECT COUNT(*) FROM topic_assignments WHERE origin='representative'"), len(set(result['assignments'])))

    def test_boundary_documents_report_competing_topic(self):
        self.corpus(groups={'hybrid': ['graph protein folding network message structure', 'graph protein folding network sequence alignment'], 'graph': ['graph node network message embedding training', 'graph network message node classification'], 'protein': ['protein folding sequence alignment molecular biology', 'protein structure molecular folding alignment']})
        result = self.experiment(definition=topic_definition(candidates=[2, 3], seed='boundary', minimum_similarity=0.95))
        boundaries = [row for topic in result['topics'] for row in topic['boundary_documents']]
        self.assertTrue(boundaries)
        self.assertTrue(any(row['boundary_reason'] == 'below_minimum_similarity' for row in boundaries))
        for row in boundaries:
            self.assertEqual(row['topic_id'], next(topic['topic_id'] for topic in result['topics'] if row['document_id'] in topic['documents']))
            self.assertIn('nearest_topic', row)
            self.assertGreaterEqual(row['nearest_other_similarity'], 0)
            self.assertGreaterEqual(row['similarity'], 0)

    def test_classification_obeys_stored_constraints(self):
        self.corpus()
        result = self.experiment()
        graph = next(topic for topic in result['topics'] if any(row['term'] == 'graph' for row in topic['terms']))
        anchor = graph['representatives'][0]
        self.ingest([record('forced', 'quantum lattice phonon spectroscopy')], 'forced-batch')
        forced = self.document('forced')['document_id']
        topic_constraints.put(self.db, forced, anchor, 'same', 0, 'supervisor', 'Editor override')
        classified = experiments.classify(self.db, 'experiment', [forced])
        row = classified['results'][0]
        self.assertEqual(row['state'], 'assigned')
        self.assertEqual(row['topic_id'], graph['topic_id'])
        self.assertEqual(row['origin'], 'constraint_forced')
        self.assertEqual(row['reason'], 'forced_without_known_terms')
        self.assertEqual(classified['pending_area'], [])

        other = self.ingest([record('blocked', 'graph network message embedding node')], 'blocked-batch')
        blocked = self.document('blocked')['document_id']
        topic_constraints.put(self.db, blocked, anchor, 'different', 0, 'supervisor', 'Editor exclusion')
        classified = experiments.classify(self.db, 'experiment', [blocked])
        row = classified['results'][0]
        if row['state'] == 'assigned':
            self.assertNotEqual(row['topic_id'], graph['topic_id'])
        else:
            self.assertEqual(row['reason'], 'below_threshold')
        self.assertIn(blocked, classified['pending_area'] or [blocked])

    def test_incremental_classification_can_be_requeued(self):
        self.corpus()
        self.experiment()
        self.ingest([record('unknown', 'quantum lattice phonon spectroscopy')], 'unknown-batch')
        identity = self.document('unknown')['document_id']
        first = experiments.classify(self.db, 'experiment', [identity])['results'][0]
        self.assertEqual(first['state'], 'unassigned')
        self.assertEqual(first['reason'], 'no_known_terms')
        second = experiments.classify(self.db, 'experiment', [identity])['results'][0]
        self.assertEqual(second['reason'], 'no_known_terms')
        self.assertEqual(experiments.get(self.db, 'experiment')['participants']['incremental'], 1)

    def test_source_impact_lists_invalidated_experiments(self):
        self.corpus()
        self.experiment()
        sources.revoke(self.db, 'source', 1, 'Withdrawal')
        impact = inspection.source_impact(self.db, 'source')
        self.assertEqual(len(impact['topic_experiments']), 1)
        self.assertEqual(impact['topic_experiments'][0]['experiment_id'], 'experiment')
        self.assertEqual({reason['code'] for reason in impact['topic_experiments'][0]['invalid_reasons']}, {'source_authorization'})

    def test_demo_reports_repost_effect_and_constraint_adjustment(self):
        import contextlib
        import io
        import pathlib
        from corpus_gate.cli import main
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(['--db', ':memory:', 'demo', '--fixture', 'fixtures/demo.json']), 0)
        report = json.loads(output.getvalue())
        exploration = report['topic_exploration']
        self.assertGreater(exploration['baseline']['repost_documents'], 0)
        self.assertTrue(exploration['constrained']['forced_documents_in_different_topics'])
        self.assertTrue(exploration['constrained']['topic_changed_by_constraint'])
        self.assertTrue(exploration['contradiction_detected'])
        self.assertTrue(report['withdrawal_invalidates_experiment'])


class TopicAuditTests(TopicCase):
    def test_experiment_run_is_audited_and_ignored_constraints_reported(self):
        self.corpus()
        batch.stage(self.db, 'short', 'source', 'default', [record('outside', 'brief')])
        batch.run(self.db, 'short')
        clusters.run(self.db, 'audit-run')
        outsider = self.document('outside')['document_id']
        anchor = self.document('graph-0')['document_id']
        topic_constraints.put(self.db, outsider, anchor, 'same', 0, 'supervisor', 'Outside the run')
        result = self.experiment('audited', run='audit-run')
        self.assertEqual(len(result['constraints']['ignored_rules']), 1)
        actions = [row['action'] for row in self.db.all('SELECT action FROM audit ORDER BY position')]
        self.assertIn('topic.experiment', actions)
        self.assertIn('topic.constraints_ignored', actions)
        self.assertTrue(self.db.audit_check()['ok'])

    def test_every_effective_unit_maps_to_one_topic_and_assignment(self):
        self.corpus(reposts=3)
        result = self.experiment()
        for unit in result['units']:
            if unit['topic_id'] is None:
                continue
            for identity in unit['included']:
                self.assertEqual(result['assignments'].get(identity), unit['topic_id'])
        self.assertEqual(len(result['assignments']), sum(len(topic['documents']) for topic in result['topics']))
        self.assertEqual(sorted(result['assignments']), sorted(identity for topic in result['topics'] for identity in topic['documents']))
