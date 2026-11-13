"""Read small progress receipts from the existing Volume; never start compute."""
import argparse
import asyncio
import datetime
import io
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save')
    args = parser.parse_args()
    volume = modal.Volume.from_name('persona-podb-analysis-261104', create_if_missing=False)
    root = 'podb-261104-01a188ac'
    report = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'files': {}, 'missing': []}
    names = ['prepare_status.json', 'gpu_budget.json', 'tests_status.json', 'probe_status.json',
             'preflight.json', 'full_status.json', 'full_verification.json']
    async def read_one(name):
        try:
            stream = io.BytesIO()
            await asyncio.wait_for(volume.read_file_into_fileobj.aio(root + '/' + name, stream), timeout=20)
            if stream.tell() > 200000:
                raise RuntimeError('Progress receipt exceeds the bounded read limit')
            report['files'][name] = json.loads(stream.getvalue())
        except (FileNotFoundError, modal.exception.NotFoundError):
            report['missing'].append(name)
        except TimeoutError:
            report.setdefault('read_timeouts', []).append(name)
    async def read_all():
        await asyncio.gather(*(read_one(name) for name in names))
    asyncio.run(read_all())
    text = json.dumps(report, indent=2) + '\n'
    if args.save:
        Path(args.save).write_text(text)
    print(text)


if __name__ == '__main__':
    main()
