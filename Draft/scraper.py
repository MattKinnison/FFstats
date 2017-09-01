import re
import csv

with open('players.txt') as ply:
    players = ply.read()

names = re.findall(r"ank\">(.*?)</a",players)
teams = re.findall(r"xs\">(.*?) -",players)
pos = re.findall(r"- (.*?)</sp",players)
projs = re.findall(r"Fw-b\">(.*?)</sp",players)
p_ranks = range(1,26)
a_ranks = [a for a in re.findall(r"\"Ta-end\"><div >(.*?)</div",players) if '.' not in a and '<' not in a]

print(len(names))
print(len(teams))
print(len(pos))
print(len(projs))
print(len(p_ranks))
a_ranks = a_ranks + [0,0]

with open('draft.csv','a',newline="") as drf:
    riter = csv.writer(drf)
    for ndx in range(len(names)):
        riter.writerow([names[ndx],teams[ndx],pos[ndx],projs[ndx],p_ranks[ndx],a_ranks[ndx]])