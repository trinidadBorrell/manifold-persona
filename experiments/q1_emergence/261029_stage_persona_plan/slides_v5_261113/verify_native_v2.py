import json, re, copy
from pathlib import Path
P=Path(__file__).parent
old=json.loads((P/'raw-template-v2.json').read_text()); new=json.loads((P/'raw-output-v2.json').read_text()); spec=json.loads((P/'parsed-spec-v2.json').read_text())
ids=[f'u1113b_slide_{i:02}' for i in range(1,19)]
assert new['slides'][0]['objectId']=='p1'
assert [s['objectId'] for s in new['slides'][1:19]]==ids
assert new['slides'][19]['objectId']=='u1016b_slide_01'
assert len(new['slides'])==len(old['slides'])+2==60
def clean(v):
 if isinstance(v,dict): return {k:clean(x) for k,x in v.items() if k not in ('contentUrl','sourceUrl')}
 if isinstance(v,list): return [clean(x) for x in v]
 return v
untouched=[s for s in old['slides'] if not s['objectId'].startswith('u1113_slide_')]
assert [s['objectId'] for s in new['slides'] if s['objectId'] not in ids]==[s['objectId'] for s in untouched]
for s in untouched:
 assert clean(s)==clean(next(x for x in new['slides'] if x['objectId']==s['objectId'])), s['objectId']
def txt(e): return ''.join(t.get('textRun',{}).get('content','') for t in e.get('text',{}).get('textElements',[])).rstrip('\n')
out=[]; font=[]
for i,(sid,sp) in enumerate(zip(ids,spec),1):
 s=next(s for s in new['slides'] if s['objectId']==sid)
 els={e['objectId'].split('_')[-1]:e for e in s['pageElements']}
 actual=[]
 for role in ('question','part','title','heading','body','prompt','caption','result','table','note'):
  if role not in els: continue
  e=els[role]
  if 'shape' in e: actual.append(txt(e['shape']))
  if 'table' in e:
   actual.extend(txt(c) for r in e['table']['tableRows'] for c in r['tableCells'])
  runs=e.get('shape',{}).get('text',{}).get('textElements',[])
  for r in runs:
   if 'textRun' in r:font.append(r['textRun']['style'].get('fontSize',{}).get('magnitude',0))
 expected=[sp['part'],sp['title'],'\n'.join(sp['body']),sp['prompt'],sp['caption']]
 expected.extend(c.strip() for row in sp['table'] for c in row)
 assert sorted(x for x in actual if x)==sorted(x for x in expected if x), (sid,actual,expected)
 # Export spatial reading order: part, heading, body/prompt, table, captions.
 out.append('## Slide '+str(i)+'\n\n'+'\n\n'.join(x for x in actual if x))
 assert all(e['objectId'].startswith('u1113b_') for e in s['pageElements'])
assert min(font)>=12,min(font)
(P/'google_new_slides_text_v2.md').write_text('# Native text from the 18 new Google Slides\n\n'+'\n\n'.join(out)+'\n')
result=dict(total_slides=60,deleted=16,new=18,unchanged_original_slides=len(untouched),exact_text=True,min_native_font_pt=min(font),positions={sid:i+2 for i,sid in enumerate(ids)})
(P/'verification-v2.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
