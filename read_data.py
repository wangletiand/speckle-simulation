"""Read raw measurements only; no preprocessing/reconstruction."""
from pathlib import Path
import json,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
def records():
    return [json.loads(s) for s in (ROOT/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
def load(index=0,noisy=True):
    r=records()[index]
    if 'measurement' in r:
        with np.load(ROOT/r['measurement'],allow_pickle=False) as a:
            speckle=a['speckle_noisy' if noisy else 'speckle_clean'].copy()
        target=np.load(ROOT/r['target'],allow_pickle=False)
    else:
        speckle=np.array(Image.open(ROOT/r['speckle']))
        target=None
    return target,speckle,r
if __name__=='__main__':
    rows=records();o,s,r=load()
    print('records',len(rows),'target',None if o is None else o.shape,'speckle',s.shape,s.dtype)
