import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__))); from _paths import ROOT, DATA, WORK  # repository paths
import xml.etree.ElementTree as ET, sys, re
root=ET.parse(sys.argv[1]).getroot()
def walk(system, path):
    names={}
    for b in system.findall('Block'):
        names[b.get('SID')]=b.get('Name').replace('\n',' ')
    print(f'== System {path}: blocks =', sorted(names.values()))
    for L in system.findall('Line'):
        if L.get('LineType')!='Connection': continue
        ends=[p.text for p in L.iter('P') if p.get('Name') in ('Src','Dst')]
        lab=[]
        for e in ends:
            m=re.match(r'(\d+)#(\w+):(\d+)',e)
            lab.append(f"{names.get(m.group(1),m.group(1))}[{m.group(2)}{m.group(3)}]")
        print('   NODE:', ' = '.join(lab))
    for b in system.findall('Block'):
        sub=b.find('System')
        if sub is not None: walk(sub, path+'/'+b.get('Name').replace('\n',' '))
walk(root.find('Model/System') if root.find('Model/System') is not None else root.find('.//System'),'top')
