import json, re, math
from pathlib import Path
from PIL import Image
P=Path(__file__).parent
d=json.loads((P/'raw-template-v2.json').read_text())
parts=re.split(r'\n## Slide \d+[^\n]*\n', (P/'google_slides_spec_v2.md').read_text())[1:]
spec=[]
for i,p in enumerate(parts,1):
 def val(pattern):
  m=re.search(pattern,p,re.M); return m.group(1).strip() if m else ''
 spec.append(dict(n=i,part=val(r'^Small part line: (.*)'),title=val(r'^(?:Title|Heading(?:, centred)?): (.*)'),body=re.findall(r'^- (.*)',p,re.M),figure=val(r'^Figure: (\S+\.png)'),prompt=val(r'```\n([\s\S]*?)\n```'),caption=val(r'^(?:Small caption under it|Caption under the table): (.*)'),table=[x.strip().strip('|').split('|') for x in p.splitlines() if x.startswith('|') and not x.startswith('|---')]))
req=[]; images=[]; plans=[]
def add(k,v): req.append({k:v})
def props(sl,x,y,w,h): return dict(pageObjectId=sl,size=dict(width=dict(magnitude=w,unit='PT'),height=dict(magnitude=h,unit='PT')),transform=dict(scaleX=1,scaleY=1,translateX=x,translateY=y,unit='PT'))
def rgb(h): return dict(zip(('red','green','blue'),(int(h[k:k+2],16)/255 for k in (0,2,4))))
def text(sl,id,txt,x,y,w,h,size=14,bullets=False,font='Arial',color='000000',existing=None,align='START',gap=7):
 if existing:
  add('deleteText',dict(objectId=id,textRange=dict(type='ALL')))
  add('updatePageElementTransform',dict(objectId=id,applyMode='ABSOLUTE',transform=dict(scaleX=w*12700/existing['size']['width']['magnitude'],scaleY=h*12700/existing['size']['height']['magnitude'],translateX=x,translateY=y,unit='PT')))
 else: add('createShape',dict(objectId=id,shapeType='TEXT_BOX',elementProperties=props(sl,x,y,w,h)))
 add('insertText',dict(objectId=id,insertionIndex=0,text=txt))
 add('deleteParagraphBullets',dict(objectId=id,textRange=dict(type='ALL')))
 add('updateTextStyle',dict(objectId=id,textRange=dict(type='ALL'),style=dict(fontFamily=font,fontSize=dict(magnitude=size,unit='PT'),bold=False,foregroundColor=dict(opaqueColor=dict(rgbColor=rgb(color)))),fields='fontFamily,fontSize,bold,foregroundColor'))
 add('updateParagraphStyle',dict(objectId=id,textRange=dict(type='ALL'),style=dict(alignment=align,lineSpacing=110,spaceAbove=dict(magnitude=0,unit='PT'),spaceBelow=dict(magnitude=gap,unit='PT'),indentStart=dict(magnitude=18 if bullets else 0,unit='PT'),indentEnd=dict(magnitude=0,unit='PT'),indentFirstLine=dict(magnitude=0,unit='PT')),fields='alignment,lineSpacing,spaceAbove,spaceBelow,indentStart,indentEnd,indentFirstLine'))
 if bullets: add('createParagraphBullets',dict(objectId=id,textRange=dict(type='ALL'),bulletPreset='BULLET_DISC_CIRCLE_SQUARE'))
 add('updateShapeProperties',dict(objectId=id,shapeProperties=dict(outline=dict(propertyState='NOT_RENDERED'),autofit=dict(autofitType='NONE')),fields='outline.propertyState,autofit.autofitType'))
for s in spec:
 n=s['n']; sl=f'u1113b_slide_{n:02}'; prefix=f'u1113b_{n:02}_'
 ref='u1016b_slide_'+('01' if n==1 else '03' if n==16 else '10' if n==18 else '12' if n in (12,13,14,15) else '04')
 src=next(z for z in d['slides'] if z['objectId']==ref)
 mapping={ref:sl}; mapping.update({e['objectId']:prefix+e['objectId'].split('_')[-1] for e in src['pageElements']})
 add('duplicateObject',dict(objectId=ref,objectIds=mapping))
 roles={e['objectId'].split('_')[-1]:e for e in src['pageElements']}
 keep={'title','body'} if n==1 else {'heading','body','prompt','caption'} if n==16 else {'question','heading','table','note'} if n==18 else {'heading','body'} if n in (12,13,14) else {'heading'} if n==15 else {'question','heading','result'}
 plans.append(dict(slide=sl,exemplar=ref,elements={mapping[e['objectId']]:('keep and populate' if e['objectId'].split('_')[-1] in keep else 'delete unused source media') for e in src['pageElements']}))
 for role,e in roles.items():
  if role not in keep: add('deleteObject',dict(objectId=mapping[e['objectId']] ))
 def tx(role,txt,x,y,w,h,size=14,**kw): text(sl,prefix+role,txt,x,y,w,h,size,existing=roles.get(role),**kw)
 if n==1:
  tx('title',s['title'],200,40,560,48,32,align='CENTER')
  tx('body','\n'.join(s['body']),118,130,730,330,14,bullets=True,gap=24)
  continue
 if n==15:
  tx('heading',s['title'],40,220,880,70,32,align='CENTER');continue
 if s['part']: tx('question' if n!=16 else 'part',s['part'],40,12,880,25,12,color='5F6368')
 elif 'question' in keep:add('deleteObject',dict(objectId=prefix+'question'))
 title_y=40 if s['part'] else 35
 title_h=68 if len(s['title'])>82 else 40
 tx('heading',s['title'],40,title_y,880,title_h,24 if len(s['title'])>82 else 26)
 top=title_y+title_h+8
 if 2<=n<=11:
  txt='\n'.join(s['body']); bh=66 if n in (5,10) else 53
  path=P/s['figure']; iw,ih=Image.open(path).size
  fw=min(880,(520-top-bh-8)*iw/ih);fh=fw*ih/iw
  images.append(dict(slide=sl,path=str(path.resolve()),requests=[dict(createImage=dict(objectId=prefix+'figure',url=str(path.resolve()),elementProperties=props(sl,40+(880-fw)/2,top,fw,fh)))]))
  tx('result',txt,40,top+fh+7,880,bh,14,bullets=True,gap=5)
 elif n in (12,13,14):tx('body','\n'.join(s['body']),55,max(top,112),850,390,14,bullets=True,gap=23)
 elif n==16:
  tx('body','\n'.join(s['body']),40,100,421,414,14,bullets=True,gap=10)
  tx('prompt',s['prompt'],489,102,431,250,12,font='Courier New',gap=0)
  add('updateShapeProperties',dict(objectId=prefix+'prompt',shapeProperties=dict(shapeBackgroundFill=dict(solidFill=dict(color=dict(rgbColor=rgb('EEF0F2')),alpha=1))),fields='shapeBackgroundFill'))
  tx('caption',s['caption'],489,370,431,145,12,gap=0)
 elif n==17:
  path=P/s['figure'];iw,ih=Image.open(path).size;fw=528;fh=fw*ih/iw
  images.append(dict(slide=sl,path=str(path.resolve()),requests=[dict(createImage=dict(objectId=prefix+'figure',url=str(path.resolve()),elementProperties=props(sl,40,top,fw,fh)))]))
  tx('result','\n'.join(s['body']),585,top,335,520-top,12,bullets=True,gap=7)
 elif n==18:
  tid=prefix+'table'
  add('deleteTableColumn',dict(tableObjectId=tid,cellLocation=dict(rowIndex=0,columnIndex=2)))
  add('insertTableRows',dict(tableObjectId=tid,cellLocation=dict(rowIndex=2,columnIndex=0),insertBelow=True,number=1))
  add('updatePageElementTransform',dict(objectId=tid,applyMode='ABSOLUTE',transform=dict(scaleX=1,scaleY=1,translateX=40,translateY=120,unit='PT')))
  add('updateTableColumnProperties',dict(objectId=tid,columnIndices=[0],tableColumnProperties=dict(columnWidth=dict(magnitude=440,unit='PT')),fields='columnWidth'))
  add('updateTableColumnProperties',dict(objectId=tid,columnIndices=[1],tableColumnProperties=dict(columnWidth=dict(magnitude=440,unit='PT')),fields='columnWidth'))
  for r,row in enumerate(s['table']):
   for c,cell in enumerate(row):
    loc=dict(rowIndex=r,columnIndex=c)
    if r<3:add('deleteText',dict(objectId=tid,cellLocation=loc,textRange=dict(type='ALL')))
    add('insertText',dict(objectId=tid,cellLocation=loc,text=cell.strip(),insertionIndex=0))
    add('updateTextStyle',dict(objectId=tid,cellLocation=loc,textRange=dict(type='ALL'),style=dict(fontFamily='Arial',fontSize=dict(magnitude=14,unit='PT'),bold=r==0),fields='fontFamily,fontSize,bold'))
  add('updateTableRowProperties',dict(objectId=tid,rowIndices=[0,1,2,3],tableRowProperties=dict(minRowHeight=dict(magnitude=62,unit='PT')),fields='minRowHeight'))
  tx('note',s['caption'],40,405,880,95,12)
(P/'parsed-spec-v2.json').write_text(json.dumps(spec,indent=2))
(P/'construction-plan-v2.json').write_text(json.dumps(plans,indent=2))
(P/'native-build-v2.json').write_text(json.dumps(req))
(P/'image-batches-v2.json').write_text(json.dumps(images))
print(len(req),len(images))
