import json
from corpus_gate.errors import text, canonical, fail
from corpus_gate import documents
from corpus_gate.review import desk


def pair(db, left_id, right_id):
    documents.get(db, left_id)
    documents.get(db, right_id)
    if left_id == right_id:
        fail('topic_constraint_self', 'Topic constraint requires distinct documents')
    return tuple(sorted([left_id, right_id]))


def put(db, left_id, right_id, kind, expected_revision, actor, reason):
    if kind not in {'same', 'different'}:
        fail('topic_constraint_kind', 'Topic constraint kind must be same or different')
    reason = text(reason, 'reason')
    if desk.actor(db, actor)['role'] != 'supervisor':
        fail('topic_constraint_supervisor', 'Topic constraints require a supervisor')
    with db.transaction():
        left_id, right_id = pair(db, left_id, right_id)
        row = db.one('SELECT * FROM topic_constraints WHERE left_id=? AND right_id=?', (left_id, right_id))
        revision = row['revision'] if row else 0
        if revision != expected_revision:
            fail('topic_constraint_stale', 'Topic constraint revision changed')
        db.execute("INSERT INTO topic_constraints VALUES(?,?,?,?,'active',?,?) ON CONFLICT(left_id,right_id) DO UPDATE SET kind=excluded.kind,revision=excluded.revision,state=excluded.state,reason=excluded.reason,actor=excluded.actor", (left_id, right_id, kind, revision + 1, reason, actor))
        result = get(db, left_id, right_id)
        db.execute('INSERT INTO topic_constraint_history VALUES(?,?,?,?)', (left_id, right_id, result['revision'], canonical(result)))
        db.log('topic.constraint_set', result)
        return result


def get(db, left_id, right_id):
    left_id, right_id = pair(db, left_id, right_id)
    row = db.one('SELECT * FROM topic_constraints WHERE left_id=? AND right_id=?', (left_id, right_id))
    if row is None:
        fail('topic_constraint_missing', 'Topic constraint does not exist')
    return row


def remove(db, left_id, right_id, expected_revision, actor, reason):
    reason = text(reason, 'reason')
    if desk.actor(db, actor)['role'] != 'supervisor':
        fail('topic_constraint_supervisor', 'Only a supervisor may remove topic constraints')
    with db.transaction():
        row = get(db, left_id, right_id)
        if row['revision'] != expected_revision:
            fail('topic_constraint_stale', 'Topic constraint changed')
        if row['state'] != 'active':
            fail('topic_constraint_inactive', 'Topic constraint already removed')
        db.execute("UPDATE topic_constraints SET state='removed',revision=revision+1,actor=?,reason=? WHERE left_id=? AND right_id=?", (actor, reason, row['left_id'], row['right_id']))
        result = get(db, row['left_id'], row['right_id'])
        db.execute('INSERT INTO topic_constraint_history VALUES(?,?,?,?)', (row['left_id'], row['right_id'], result['revision'], canonical(result)))
        db.log('topic.constraint_removed', result)
        return result


def listing(db, active_only=True):
    query = 'SELECT * FROM topic_constraints'
    if active_only:
        query += " WHERE state='active'"
    return db.all(query + ' ORDER BY left_id,right_id')


def history(db, left_id, right_id):
    left_id, right_id = pair(db, left_id, right_id)
    return [json.loads(row['payload']) for row in db.all('SELECT payload FROM topic_constraint_history WHERE left_id=? AND right_id=? ORDER BY revision', (left_id, right_id))]


class UnionFind:
    def __init__(self, identities=()):
        self.parents = {identity: identity for identity in identities}

    def add(self, identity):
        self.parents.setdefault(identity, identity)

    def root(self, identity):
        self.add(identity)
        trail = []
        while self.parents[identity] != identity:
            trail.append(identity)
            identity = self.parents[identity]
        for member in trail:
            self.parents[member] = identity
        return identity

    def union(self, left, right):
        a = self.root(left)
        b = self.root(right)
        if a != b:
            smaller, larger = sorted([a, b])
            self.parents[larger] = smaller
