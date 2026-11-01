import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import re,sys
x=open(sys.argv[1],encoding='utf8').read()
print('chars',len(x))
for m in re.finditer(r'<Block BlockType="([^"]+)" Name="([^"]*)" SID="([^"]+)"',x):
    start=m.end(); depth=1; i=start
    # take body until matching </Block> (blocks can nest)
    while depth and i<len(x):
        o=x.find('<Block ',i); c=x.find('</Block>',i)
        if o!=-1 and o<c: depth+=1; i=o+7
        else: depth-=1; i=c+8
    body=x[start:i]
    ps=re.findall(r'<P Name="([^"]+)"[^>]*>([^<]*)</P>',body)
    skip={'Position','ZOrder','Ports','BackgroundColor','ForegroundColor','LibraryVersion','SourceType','ShowName','HideAutomaticName','NamePlacement','Orientation','DropShadow','FontName','FontSize','BlockMirror','BlockRotation','IconDisplay','MaskIconFrame','MaskDisplay','RTWMemSecFuncInitTerm','RTWMemSecFuncExecute','RTWMemSecDataConstants','RTWMemSecDataInternal','RTWMemSecDataParameters'}
    keep=[(k,v) for k,v in ps if k not in skip]
    print(f'--- [{m.group(1)}] {m.group(2)!r} SID{m.group(3)}')
    for k,v in keep[:45]: print('    ',k,'=',v[:140].replace('\n',' '))
