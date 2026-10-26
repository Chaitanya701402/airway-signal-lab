"""Run from repository root: python tests/smoke_application.py."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from engine import run_scenario
from experiments.hybrid_ablation import smoke_models

app = AppTest.from_file('app.py').run(timeout=60)
assert not app.exception, str(app.exception)
result = run_scenario('config.ready.json')
assert result['metadata']['diagnostics']['windows'] == len(result['times'])
print(json.dumps({'architecture_smoke': smoke_models(),
    'streamlit_initial_exceptions': len(app.exception),
    'tflite_diagnostics': result['metadata']['diagnostics'],
    'runtime': result['metadata']['runtime'],
    'model_sha256': result['metadata']['model_sha256']}, indent=2))
