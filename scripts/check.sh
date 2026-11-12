#!/bin/sh
set -eu
python3 -m compileall -q corpus_gate
python3 -m unittest discover -s tests -v
python3 -m corpus_gate --db :memory: demo --fixture fixtures/demo.json
