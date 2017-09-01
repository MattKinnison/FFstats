'''Fantasy Sports Predictor - Python - v3_0

    By: Matthew Joseph Kinnison (mkinnisonj@gmail.com)

    Runs calculus based 'random' simulations of all the possible results of a
    given season and outputs the results in a format easily usable for owner
    decision making.
'''

# ***** Imported Modules ***** #
import numpy as np
from scipy.integrate import cumtrapz
from scipy.stats import norm
from jsonDB import *
import sys
import csv
import pandas as pd
#from matplotlib import pyplot as plt
import pprint as pp
#import heapq

fidelity = 10000
jql = jsonDB('fantasy.db')

# ***** Helper Functions ***** #
def m_round(num):
    '''rounds num to exactly 2 digits
    in: num:    float (not rounded)
    out:    float (rounded to 2 digits)
    '''
    return float('%.4f' % num)

def dict_map(func, dict_, *args):
    '''maps every value in a dictionary through the function described in func.
        Only certain functions allowed for security.
    in: func:   string containing a valid function name
        dict_:  dictionary to have each value mapped to this function
    out:    dictionary with all func applies to all values and reassigned to
            keys.
    '''
    allowed = ['m_round','make_curve','sum','mean','stdev']
    if func in allowed:
        return {key: eval(func)(dict_[key]) for key in dict_.keys()}
    elif func == 'add':
        return {key: dict_[key] + args[0][key] for key in dict_.keys()}
    elif func == 'sub':
        return {key: dict_[key] - args[0][key] for key in dict_.keys()}
    elif func == 'mult':
        return {key: dict_[key]*args[0] for key in dict_.keys()}
    else:
        raise ValueError('invalid function. Allowed functions are {}.'.format(allowed))

def get_mean(league,ovrd):
    if ovrd:
        return {team['Name']: {stat: np.mean(jql.select('league',league,'teams',ndx,'Season',where=(stat,"",stat))[:ovrd],axis=0) for stat in team['Season'][0].keys()} for ndx,team in enumerate(jql.select('league',league,'teams'))}
    else:
        return {team['Name']: {stat: np.mean(jql.select('league',league,'teams',ndx,'Season',where=(stat,"",stat)),axis=0) for stat in team['Season'][0].keys()} for ndx,team in enumerate(jql.select('league',league,'teams'))}

def get_std(league,ovrd):
    if ovrd:
        return {team['Name']: {stat: np.std(jql.select('league',league,'teams',ndx,'Season',where=(stat,"",stat))[:ovrd],axis=0) for stat in team['Season'][0].keys()} for ndx,team in enumerate(jql.select('league',league,'teams'))}
    else:
        return {team['Name']: {stat: np.std(jql.select('league',league,'teams',ndx,'Season',where=(stat,"",stat)),axis=0) for stat in team['Season'][0].keys()} for ndx,team in enumerate(jql.select('league',league,'teams'))}

def get_sch(league):
    return {team['Name']: team['Schedule'] for team in jql.select('league',league,'teams')}

def make_w_l(league,ovrd):
    if ovrd:
        scores = {team['Name']: team['Season'][:ovrd] for team in jql.select('league',league,'teams')}
        schedule = {team['Name']: team['Schedule'] for team in jql.select('league',league,'teams')}
        return {key: sum([sum([(0.5 if val == 0 else (1 if ((key2 == 'WHIP' or key2 == 'ERA') and val < 0) or (not (key2 == 'WHIP' or key2 == 'ERA') and val > 0) else 0)) for key2,val in dict_map('sub',scores[key][ndx],scores[schedule[key][ndx]][ndx]).items()]) for ndx in range(len(scores[key]))]) for key in scores.keys()}
    else:
        scores = {team['Name']: team['Season'] for team in jql.select('league',league,'teams')}
        schedule = {team['Name']: team['Schedule'] for team in jql.select('league',league,'teams')}
        return {key: sum([sum([(0.5 if val == 0 else (1 if ((key2 == 'WHIP' or key2 == 'ERA') and val < 0) or (not (key2 == 'WHIP' or key2 == 'ERA') and val > 0) else 0)) for key2,val in dict_map('sub',scores[key][ndx],scores[schedule[key][ndx]][ndx]).items()]) for ndx in range(len(scores[key]))]) for key in scores.keys()}

# ***** Table Writer ***** #
def write_table(filename, datas):
    '''writes various odds to csv for easy visualization and application of
        conditional formating.
    in: filename:   string countaining the full filename of the csv file
        headers:    list of the headers for each of the inputed dictionaries
        *args:  dictonaries with keys containing all the teams mapped to the
            desired output
    '''
    with open(filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Teams'] + [list(odd.keys())[0] for odd in datas])
        for key in datas[0][list(datas[0].keys())[0]].keys():
            writer.writerow([key] + [data[list(data.keys())[0]][key] for data in datas])

def odds_report(league,season=False):
    get_odds(league)
    if season:
        print('Under Construction')
    else:
        write_table(league+'_Odds.csv',[{bar: {team['Name']: team['Odds'][-1][bar] for team in jql.select('league',league,'teams')}} for bar in jql.select('game type',jql.select('league','Baseball_17','schema'),0,'values','values',3,'values','values',where=('name','"*"','name'))])

# ***** Matchups Logic ***** #
def new_odds(league,ovrd=False):
    '''
    '''
    means = get_mean(league,ovrd)
    stdevs = get_std(league,ovrd)
    scored = {entry['name']: entry['scored'] for entry in jql.select('game type',jql.select('league',league,'schema'),0,'values','values',1,'values','values',where=('name','"*"','*'))}
    teams = means.keys()
    probs = {team: {opp: [np.absolute((0 if scored[stat] == -1 else 1)-norm.cdf(0,means[team][stat]-means[opp][stat],np.sqrt(stdevs[team][stat]**2+stdevs[opp][stat]**2))) for stat in means[team].keys() if scored[stat] != 0] for opp in teams if opp != team} for team in teams}
    probs = {team: {opp: np.sum(np.array([[probs[team][opp][ndx],probs[team][opp][ndx]*probs[opp][team][ndx]] for ndx in range(10)]),axis=0) for opp in teams if opp != team} for team in teams}
    for team in probs.keys():
        probs[team][''] = np.zeros(2)
    return probs

def avg_prob(league): #################################################################################################################
    '''makes a dictionary comparing each team to their average opponent.
    in: scores: dictornary with team names as keys and arrays of previous
            scores as values
    out:    dictionary containing floats representing probability
    '''
    means = get_mean(league,False)
    stdevs = get_std(league,False)
    scored = {entry['name']: entry['scored'] for entry in jql.select('game type',jql.select('league',league,'schema'),0,'values','values',1,'values','values',where=('name','"*"','*'))}
    probs = {key: {'avg':0} for key in means.keys()}
    for key in probs.keys():
        odds = 0
        for stat in scored.keys():
            raw = jql.select('league',league,'teams',where=(stat,"",stat))
            mu = np.mean(raw)
            sigma = np.std(raw)
            odds = odds + np.absolute((0 if scored[stat] == -1 else 1)-norm.cdf(0,means[key][stat]-mu,np.sqrt(sigma**2+stdevs[key][stat]**2)))
        probs[key] = odds
    return probs

def get_odds(league,week=False):

    if week == False:
        week = len(jql.select('league',league,'teams',0,'Season'))+1

    if week <= 2:
        all_odds = {cat: {team['Name']: 1/num for team in jql.select('league',league,'teams')} for cat,num in jql.select('league',league,'playoffs').items()}
        all_odds['champion'] = {team['Name']: 1/len(jql.select('league',league,'teams')) for team in jql.select('league',league,'teams')}
        return pd.DataFrame(data=all_odds)[jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name')) + ['champion']]

    wins = make_w_l(league,week)
    schedule = get_sch(league)
    hth = new_odds(league,ovrd=week)
    weeks_left = len(jql.select('league',league,'teams',0,'Schedule'))-len(jql.select('league',league,'teams',0,'Season')[:week])
    record = predict_record(schedule,hth,weeks_left)
    record_curves = make_record_curves(record,wins,schedule)
    cdf = find_cutoffs(record_curves)
    all_odds = {cat: find_top(cdf,record_curves,jql.select('league',league,'playoffs',cat)) for cat in jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name'))}
    all_odds['champion'] = playoffs(jql.select('league',league,'playoffs','playoff'),hth,cdf=cdf,pdfs=record_curves)

    return pd.DataFrame(data=all_odds)[jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name')) + ['champion']]

# ***** Using Schedule ***** #
def predict_record(schedule,hth,*args):
    '''
    '''
    if len(args) and args[0] > 0:
        return {team: np.sum([hth[team][opp] for opp in schedule[team][-1*args[0]:]],axis=0) for team in schedule.keys()}
    elif len(args) and args[0] < 0:
        return {team: np.sum([hth[team][opp] for opp in schedule[team][:args[0]]],axis=0) for team in schedule.keys()}
    else:
        return {team: np.sum([hth[team][opp] for opp in schedule[team]],axis=0) for team in schedule.keys()}

def make_record_curves(records,wins,schedule):
    '''
    '''
    sigma = {team: records[team][1]**0.5 for team in records.keys()}
    mu = {team: wins[team]+records[team][0] for team in records.keys()}

    # Normalize to a percent
    mu = {team: mu[team]/(10*len(list(filter(None,schedule[team])))) for team in mu.keys()}
    sigma = {team: sigma[team]/(10*len(list(filter(None,schedule[team])))) for team in mu.keys()}

    return {team: (np.linspace(1,0,fidelity),norm.pdf(np.linspace(1,0,fidelity),mu[team],sigma[team])) for team in records.keys()}

# ***** Getting odds from records ***** #
def find_cutoffs(pdfs):
    '''
    '''
    all_curves = np.zeros(fidelity)
    for pdf in pdfs.values():
        all_curves = np.add(all_curves,pdf[1])

    return cumtrapz(all_curves,dx=1/fidelity)

def find_top(cdf,pdfs,top):
    '''
    '''
    mask = np.where(cdf <= top)
    return {team: np.trapz(pdfs[team][1][mask],dx=1/fidelity) for team in pdfs.keys()}

def find_seed(cdf,pdfs,seed):
    '''
    '''
    mask1 = np.where(cdf <= seed)
    mask2 = np.where(cdf >= seed-1)
    mask = np.intersect1d(mask1[0],mask2[0])
    return {team: np.trapz(pdfs[team][1][mask],dx=1/fidelity) for team in pdfs.keys()}

def seeding(num_teams):
    full_brack = []
    brack = [[1]]
    for seed in range(2,num_teams+1):
        #split
        unpaired = [game for game in brack if len(game) == 1]
        if len(unpaired) == 0:
            full_brack.append(brack)
            new_brack = []
            for game in brack:
                new_brack.append([game[0]])
                new_brack.append([game[1]])
            brack = new_brack
            unpaired = brack
        #insert
        in_game = max(unpaired)
        brack = [game if game != in_game else [in_game[0],seed] for game in brack]
    full_brack.append(brack)
    full_brack.reverse()

    return full_brack

def playoffs(num_teams,hth,cdf=0,pdfs=0,contenders=0):
    '''
    '''
    brack = seeding(num_teams)
    if contenders == 0:
        contenders = {seed: find_seed(cdf,pdfs,seed) for seed in range(1,num_teams+1)}
    teams = hth.keys()
    for round_ in brack:
        new_cont = {}
        for game in round_:
            if len(game) == 1:
                winners = contenders[game[0]]
            else:
                team1 = contenders[game[0]]
                team2 = contenders[game[1]]
                win1 = {team: team1[team]*sum([team2[opp]*hth[team][opp][0] for opp in teams if opp != team]) for team in teams}
                win2 = {team: team2[team]*sum([team1[opp]*hth[team][opp][0] for opp in teams if opp != team]) for team in teams}
                winners = dict_map('add',win1,win2)
                adj = 1/sum(winners.values())
                winners = dict_map('mult',winners,adj)
                winners = dict_map('m_round',winners)
            new_cont[game[0]] =  winners
        contenders = new_cont
    return contenders[1]

# ***** Utilities ***** #
def input_scores(league,database='fantasy.db'):
    jql = jsonDB(database)
    teams = jql.select('league',league,'teams')
    stats = jql.select('game type',jql.select('league',league,'schema'),0,'values','values',1,'values','values',where=('name','"*"','name'))
    for ndx,team in enumerate(teams):
        correct = ''
        while len(correct) == 0 :
            print('----> Team: '+team['Name'])
            score = {}
            for stat in stats:
                val = input(stat+': ')
                score[stat] = (float(val) if '.' in val else int(val))
            correct = input('Correct?: ')
        team['Season'].append(score)
        jql.update(team,'league',league,'teams',ndx)
# ***** Outputs ***** #


# ***** Temp ***** #

# def temp(a):
#     league = 'Baseball_17'
#     jql = jsonDB('fantasy.db')
#     wins = make_w_l(league,ovrd=a)
#     schedule = get_sch(league)
#     hth = new_odds(league,ovrd=a)
#     weeks_left = 22 - a
#     record = predict_record(schedule,hth,weeks_left)
#     record_curves = make_record_curves(record,wins,schedule)
#     cdf = find_cutoffs(record_curves)
#     reg_odds = [{cat: find_top(cdf,record_curves,jql.select('league',league,'playoffs',cat))} for cat in jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name'))]
#     first = playoffs(jql.select('league',league,'playoffs','playoff'),hth,cdf=cdf,pdfs=record_curves)
#     all_odds = reg_odds + [{'champion': first[1]}]

# def get_odds(league,plots=False):
#     '''
#     '''
#     wins = make_w_l(league,False)
#     schedule = get_sch(league)
#     hth = new_odds(league)
#     weeks_left = len(jql.select('league',league,'teams',0,'Schedule'))-len(jql.select('league',league,'teams',0,'Season'))
#     if weeks_left > 0:
#         record = predict_record(schedule,hth,weeks_left)
#         record_curves = make_record_curves(record,wins,schedule)
#         if plots:
#             test_plot(record_curves)
#         cdf = find_cutoffs(record_curves)
#         reg_odds = [{cat: dict_map('m_round',find_top(cdf,record_curves,jql.select('league',league,'playoffs',cat)))} for cat in jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name'))]
#         first = playoffs(jql.select('league',league,'playoffs','playoff'),hth,cdf=cdf,pdfs=record_curves)
#         all_odds = reg_odds + [{'champion': first[1]}]
#     else:
#         reg_odds = [{cat: (1 if wins[cat] >= heapq.nlargest(jql.select('league',league,'playoffs',cat), wins.values())[-1] else 0)} for cat in jql.select('game type',jql.select('league',league,'schema'),2,'values',where=('name','"*"','name'))]
#         s_wins = sorted(wins,reverse=True)
#         seeds = {seed: {team: (1 if s_wins.index(team)+1 == seed else 0) for team in wins.keys()} for seed in range(1,playoff_teams+1)}
#         first = playoffs(playoff_teams,hth,contenders=seeds)

# ***** Special ***** #
# def main():
#     #print("def _538_table(league,plots=False,database = 'fantasy.db'):")
#     while True:
#         try:
#             pp.pprint(eval(input('----> ')))
#         except SyntaxError:
#             print("Syntax Error")

# if 'Y' in input('start internal: (Y/N) ') :
#     main()

# def test_plot(curves):
#     '''
#     '''
#     total = np.zeros(fidelity)
#     for team in curves.keys():
#         plt.plot(curves[team][0],curves[team][1],'k-')
#         total = total + curves[team][1]
#     plt.plot(curves[team][0],total,'g-')

#     plt.show()

# def pure_chance(league,database='fantasy.db'):
#     '''
#     '''
#     jql = jsonDB(database)
#     wins = make_w_l(league)
#     schedule = get_sch(league)
#     teams = wins.keys()
#     hth = {team: {opp: np.array([5, 2.5]) for opp in list(teams)+[''] if opp != team} for team in teams}
#     weeks_left = len(jql.select('league',league,'teams',0,'Schedule'))-len(jql.select('league',league,'teams',0,'Season'))
#     if weeks_left:
#         record = predict_record(schedule,hth,weeks_left)
#         record_curves = make_record_curves(record,wins,schedule)
#         cdf = find_cutoffs(record_curves)

#         test_plot(record_curves)

#         playoff = find_top(cdf,record_curves,jql.select('league',league,'playoffs','playoff'))
#         first = playoffs(jql.select('league',league,'playoffs','playoff'),hth,cdf,record_curves)

#     return playoff,first

# def str_of_sch(league,database='fantasy.db',remaining=True):
#     '''makes a dictionary of expected wins.
#     '''
#     jql = jsonDB(database)
#     avg = avg_prob(league)
#     schedule = get_sch(league)
#     avg[''] = 0
#     if remaining:
#         return predict_record(schedule,{key: avg for key in avg.keys()},len(jql.select('league',league,'teams',0,'Schedule'))-len(jql.select('league',league,'teams',0,'Season')))
#     else:
#         return predict_record(schedule,{key: avg for key in avg.keys()})

# def overachievers(league,database='fantasy.db'):
#     '''
#     '''
#     jql = jsonDB(database)
#     wins = make_w_l(league)
#     schedule = get_sch(league)
#     hth = new_odds(league)
#     record = predict_record(schedule,hth,len(jql.select('league',league,'teams',0,'Season'))-len(jql.select('league',league,'teams',0,'Schedule')))
#     return {team: m_round(wins[team]-record[team][0]) for team in wins.keys()}

# # ***** Plots ***** #
# def vs_avg(league,team,stat,db='fantasy.db'):
#     jql = jsonDB(db)
#     scores = [jql.select('league',league,'teams',where=('Name','"*"=="{}"'.format(name),stat)) for name in jql.select('league',league,'teams',where=('Name','"*"','Name'))]
#     avg = [np.mean([scr[ndx] for scr in scores]) for ndx in range(len(scores[0]))]
#     plt.plot(jql.select('league',league,'teams',where=('Name','"*"=="{}"'.format(team),stat)))
#     plt.plot(avg)
#     plt.legend([team,'League Average'])
#     plt.show()

# def week_vs_avg(league,team,week,db='fantasy.db'):
#     jql = jsonDB(db)
#     scores = [jql.select('league',league,'teams',where=('Name','"*"=="{}"'.format(name),week-1)) for name in jql.select('league',league,'teams',where=('Name','"*"','Name'))]
#     cats = jql.select('game type',jql.select('league',league,'schema'),0,'values','values',1,'values','values',where=('name','"*"','name'))
#     scores = [[a for a in scr if type(a) == type(dict())][0] for scr in scores]
#     for ndx,key in enumerate(cats):
#         plt.subplot(1,len(cats),ndx+1)
#         loc = np.arange(1)
#         plt.bar(loc,[[a for a in jql.select('league',league,'teams',where=('Name','"*"=="{}"'.format(team),week-1)) if type(a) == type(dict())][0][key]],0.35)
#         plt.bar(loc+0.35,[np.mean([scr[key] for scr in scores])],0.35,color='g')
#         plt.title(key)
#         frame1 = plt.gca()
#         frame1.axes.get_xaxis().set_ticks([])
#     plt.legend([team,'League Average'])
#     plt.show()

# def rolling_odds(league,team,db='fantasy.db'):
#     jql = jsonDB(db)
#     plt.title(team)
#     for bar in jql.select('game type',jql.select('league','Baseball_17','schema'),0,'values','values',3,'values','values',where=('name','"*"','name')):
#         plt.plot(jql.select('league',league,'teams',where=('Name','"*"=="{}"'.format(team),bar)))
#     plt.legend(jql.select('game type',jql.select('league','Baseball_17','schema'),0,'values','values',3,'values','values',where=('name','"*"','name')))
#     plt.show()