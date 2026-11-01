# STATUS — flux_student
Updated: 2026-11-02 (sync)

## Sync policy
- Teacher/student samples synced as `data/samples_b64.json` + `data/manifest.jsonl` (base64) so sandbox disk stays light.
- Continuous train continues locally; re-sync when new images appear.

## Metrics
- n_data: 26 (sd-turbo + prompt only)
- Student ~10M params, loss ~0.37-0.39
- continuous_loop running

## Recover images
```python
import json,base64
from pathlib import Path
obj=json.load(open('flux_student/data/samples_b64.json'))
for k,v in obj.items():
    Path(k).write_bytes(base64.b64decode(v))
```
