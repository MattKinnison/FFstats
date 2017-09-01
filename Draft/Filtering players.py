>>> with open('BMacDraft.csv') as raw:
...     a = raw.readlines()
...
>>> a = [b[:-1] for b in a]
>>> a[:20]
['New Player Note', 'Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0']
>>> b = [c for c in a if c != 'Questionable' and c != 'Suspended']
>>> len(b)
4700
>>> len(a)
4728
>>> len(b)/200
23.5
>>> 4700/8
587.5
>>> 25*25*150
93750
>>> 25*150
3750
>>> off = b[:3750]
>>> off[-25:]
['?', 'Player Note', 'Darren Sproles\xa0Phi - RB', 'Sun 12:00 pm @ Was', '?', 'FA', '16', '10', '126.81', '24%', '176', '176', '0', '0', '0', '64', '277', '1.1', '55.8', '53.5', '519', '2.3', '0', '0', '0']
>>> 20*25
500
>>> 4700-3750-500
450
>>> 450/25
18.0
>>> k = b[3750:4200]
>>> def = b[4200:]
  File "<stdin>", line 1
    def = b[4200:]
        ^
SyntaxError: invalid syntax
>>> DEF = b[4200:]
>>> z = [off[plyr:(25+plyr)] for plyr in range(150)]
>>> z[:2]
[['New Player Note', 'Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0', '0', '0', '4', '2', '?'], ['Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0', '0', '0', '4', '2', '?', 'Player Note']]
>>> z = [off[(plyr*25):(25+25*plyr)] for plyr in range(150)]
>>> z[:2]
[['New Player Note', 'Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0', '0', '0', '4', '2', '?'], ['Player Note', 'Tom Brady\xa0NE - QB', 'Thu 7:30 pm vs KC', '?', 'FA', '16', '9', '365.82', '100%', '2', '35', '4907', '39.9', '7.7', '24.6', '55.3', '1', '0', '0', '0', '0', '0', '3', '0', '?']]
>>> DEF[:21]
['?', 'No new player Notes', 'Kansas City\xa0KC - DEF', 'Thu 7:30 pm @ NE', '?', 'FA', '16', '10', '93.05', '98%', '245', '141', '357', '27.4', '0', '14.2', '6.1', '2', '0', '2', '?']
>>> off[:26]
['New Player Note', 'Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0', '0', '0', '4', '2', '?', 'Player Note']
>>> off = ['?']+off
>>> off.remove('PUP-P')
>>> len(off)
3750
>>> z = [off[(plyr*25):(25+25*plyr)] for plyr in range(150)]
>>> z[:2]
[['?', 'New Player Note', 'Aaron Rodgers\xa0GB - QB', 'Sun 3:25 pm vs Sea', '?', 'FA', '16', '8', '368.54', '100%', '1', '30', '4514', '35.9', '7.9', '51.3', '298', '3.1', '0', '0', '0', '0', '0', '4', '2'], ['?', 'Player Note', 'Tom Brady\xa0NE - QB', 'Thu 7:30 pm vs KC', '?', 'FA', '16', '9', '365.82', '100%', '2', '35', '4907', '39.9', '7.7', '24.6', '55.3', '1', '0', '0', '0', '0', '0', '3', '0']]
>>> y = [k[(plyr*18):(18+18*plyr)] for plyr in range(25)]
>>> x = [DEF[(plyr*20):(20+20*plyr)] for plyr in range(25)]
>>> az = [[e[2],e[8],e[10],e[11]] for e in z]
>>> az[1]
['Tom Brady\xa0NE - QB', '365.82', '2', '35']
>>> by = [[e[2],e[8],e[10],e[11]] for e in y]
>>> by[1]
['Matt Bryant\xa0Atl - K', '150.95', '119', '181']
>>> cx = [[e[2],e[8],e[10],e[11]] for e in x]
>>> cx[1]
['Atlanta\xa0Atl - DEF', '92.65', '246', '227']
>>> players = az + by + cx
>>> [e[0].split(' - ')+e[1:] for plyr in players]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in <listcomp>
NameError: name 'e' is not defined
>>> [plyr[0].split(' - ')+plyr[1:] for plyr in players][1]
['Tom Brady\xa0NE', 'QB', '365.82', '2', '35']
>>> players = [plyr[0].split(' - ')+plyr[1:] for plyr in players]
>>> players = [plyr[0].split('\za0')+plyr[1:] for plyr in players]
>>> players[0]
['Aaron Rodgers\xa0GB', 'QB', '368.54', '1', '30']
>>> players = [plyr[0].split('\xa0')+plyr[1:] for plyr in players]
>>> players[0]
['Aaron Rodgers', 'GB', 'QB', '368.54', '1', '30']
>>>