''' '''

import csv
import json
import pprint as pp

Roster = '{"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DEF": 1, "BENCH": 5}'

def get_max(l):
    max_val = max(l)
    return [ndx for ndx, val in enumerate(l) if val == max_val][0]

def ideal(pick,teams,roster,filename):

    with open(filename) as csvfile:
        players = [row for row in csv.DictReader(csvfile)]

    needs = json.loads(roster)

    rounds = sum(needs.values())

    picks = [teams*num+(pick if not num % 2 else teams - pick + 1) for num in range(rounds)]
    print(picks)

    options = [players[my_pick-bonus-1:int(my_pick-bonus-1+teams*1.5)] for bonus, my_pick in enumerate(picks)]

    bests = []
    for poss in options:
        to_add = dict()
        for pos in ['QB','RB','WR','TE','K','DEF']:
            by_pos = sorted([player for player in poss if player['Pos'] == pos], key=lambda x: x['Val'])
            if len(bests) and len(by_pos) > 1 and by_pos[-1] == bests[-1][pos]:
                by_pos[-1]['Backup'] = by_pos[-2]
                to_add[pos] = by_pos[-1]
            elif len(by_pos):
                to_add[pos] = by_pos[-1]
            else:
                to_add[pos] = {'Val':0}
        bests.append(to_add)

    pp.pprint(bests)

    margins = []
    for rd in bests:
        margins.append([])
        for pos in ['QB','RB','WR','TE','K','DEF']:
            tst = [float(rd[x]['Val']) for x in rd.keys() if float(rd[x]['Val']) > 0.5]
            margins[-1].append(float(rd[pos]['Val'])*len(tst)-sum(tst))

    pp.pprint(margins)

    for ndx, pos in enumerate(['QB','RB','WR','TE','K','DEF']):
        s = [x[ndx] for x in margins]
        print(pos + str(sorted(range(len(s)), key=lambda k: s[k])))



# def draft(pick,teams,roster,filename,randomness=0.15,quality=0.1,snake=True):

#     with open(filename) as csvfile:
#         players = csv.reader(csvfile)
#     # Remove Headers
#     players.pop(0)

#     needs = json.loads(roster)

#     rounds = sum(needs.values())

#     effc_picks = int(((teams-1)*effc)+0.5)

#     my_team = []
#     for round in range(1,rounds+1):
#         my_pick = pick
#         if snake and round % 2 == 0:
#             my_pick = teams - my_pick

#         effc_pick_count = effc_picks
#         for the_pick in range(1,teams+1):
#             if the_pick == my_pick:
#                 # Calculate the value of each position
#                 multi['QB'] = 1 if needs['QB'] else 1/15
#                 multi['RB'] = 1 if needs['RB'] or needs['FLEX'] else 1/15
#                 multi['WR'] = 1 if needs['WR'] or needs['FLEX'] else 1/15
#                 multi['TE'] = 1 if needs['TE'] or needs['FLEX'] else 1/15
#                 multi['K'] = 1 if needs['K'] else 1/15
#                 multi['DEF'] = 1 if needs['DEF'] else 1/15

#                 options = players[0:teams*2]
#                 worth = [options[4]*multi[options[3]] for player in options]
#                 my_team.append(options[get_max(worth)])
#             else:
#                 if effc_pick_count > 0:
#                     effc_pick_count -= 1
#                     players.pop(0)
#                 else:

