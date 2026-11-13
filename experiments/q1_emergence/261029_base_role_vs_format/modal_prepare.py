"""CPU-only input staging, independent of GPU account enablement."""
import json
from pathlib import Path
import sys

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal_analysis as base

app = modal.App('persona-podb-inputs-261104', image=base.image)
prepare_cpu = app.function(cpu=(2, 2), memory=(16384, 16384), timeout=1800, retries=0,
    max_containers=1, scaledown_window=2, volumes={'/data': base.volume},
    secrets=[modal.Secret.from_name('hf-read', required_keys=['HF_TOKEN'])])(base.prepare.get_raw_f())


@app.local_entrypoint()
def main():
    call = prepare_cpu.spawn()
    record = dict(app='persona-podb-inputs-261104', volume=base.VOLUME_NAME,
                  phase='prepare', function_call_id=call.object_id, GPU=False)
    folder = base.PAYLOAD.parent
    (folder / 'prepare_call.json').write_text(json.dumps(record, indent=2) + '\n')
    print('MODAL_CALL ' + json.dumps(record), flush=True)
    result = call.get()
    (folder / 'prepare_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('raw_read_text', 'files')}), flush=True)
