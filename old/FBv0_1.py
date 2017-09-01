'''Fantasy Football Predictor - Python - v2_1

    By: Matthew Joseph Kinnison (mkinnisonj@gmail.com)

    Runs calculus based 'random' simulations of all the possible results of a
    given season and outputs the results in a format easily usable for owner
    decision making.
'''

# ***** Imported Modules ***** #
import numpy as np
import math
import csv
import os.path
from statistics import mean
from statistics import stdev
from matplotlib.mlab import normpdf
from scipy.integrate import cumtrapz

fidelity = 10000

# ***** Helper Functions ***** #
def m_round(num):
    '''rounds num to exactly 2 digits
    in: num:    float (not rounded)
    out:    float (rounded to 2 digits)
    '''
    return float('%.2f' % num)

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
    elif func == 'np.mean':
        return {key: np.mean(dict_[key],axis=0) for key in dict_.keys()}
    elif func == 'np.std':
        return {key: np.std(dict_[key],axis=0) for key in dict_.keys()}
    elif func == 'mult':
        return {key: dict_[key]*args[0] for key in dict_.keys()}
    else:
        raise ValueError('invalid function. Allowed functions are {}.'.format(allowed))

def re_key(dict_,new_keys):
    '''makes a new dictionay using the values assigned to each key in new_keys
        as the new keys with the same values as the old dictionary.
    in: dict_:  dictionary to be re-keyed
        new_keys:   dictionary mapping old keys to new keys
    out:    dictornary with new keys and onld values
    '''
    return {new_keys[key]: dict_[key] for key in dict_.keys()}

def test_plot(curves):
    '''
    '''
    import matplotlib.pyplot as plt
    total = np.zeros(fidelity)
    for team in curves.keys():
        plt.plot(curves[team][0],curves[team][1],'k-')
        total = total + curves[team][1]
    plt.plot(curves[team][0],total,'g-')

    plt.show()
# ***** CSV Parsers ***** #
def parse_schedule(filename):
    '''takes in csv countaining team ids and their opponents ids each week with
        no header. Should be run on dataset only once.
    in: filename:   string countaining the full filename of the csv file
    out:schedule:   dictionary with a key for each team and a corresponding
            value which is a list countaining that teams schedule
    '''
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            schedule = {row[0]: row[1:] for row in reader}

        return schedule
    else:
        raise ValueError('{} not found'.format(filename))

def parse_scores(filename):
    '''takes in csv containing team ids and their previous points each week with
        no header. Should be run on dataset only once.
    in: filename:   string countaining the full filename of the csv file
    out:scores: dictionary with a key for each team and a corresponding array of
            their previous scores (array of floats)
    '''
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            scores = {row[0]: np.array([[float(num) for num in week.split(',')] for week in row[1:]]) for row in reader}
        return scores
    else:
        raise ValueError('{} not found'.format(filename))

def parse_other(filename):
    '''takes in csv containing team ids and their team name, their previous
        wins, and their previous ties in that order with no header. Should be
        run on dataset only once.
    in: filename:   string countaining the full filename of the csv file
    out:ids:    dictonary mapping team ids to their names (string)
        wins:   dictonary mapping team ids to their adjusted wins
            (wins + ties/2) (int)
    '''
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            other = [list(row) for row in reader]

        ids = {row[0]: row[1] for row in other}
        wins = {row[0]: int(row[2]) + int(row[3])/2 for row in other}

        return ids, wins
    else:
        raise ValueError('{} not found'.format(filename))

# ***** Table Writer ***** #
def write_table(filename, headers, *args):
    '''writes various odds to csv for easy visualization and application of
        conditional formating.
    in: filename:   string countaining the full filename of the csv file
        headers:    list of the headers for each of the inputed dictionaries
        *args:  dictonaries with keys containing all the teams mapped to the
            desired output
    '''
    with open(filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Teams'] + headers)
        for key in args[0].keys():
            writer.writerow([key] + [arg[key] for arg in args])

# ***** Matchups Logic ***** #
def get_p(mu,sigma):
    '''
    '''
    scores = np.linspace(mu-3.5*sigma,mu+3.5*sigma,1000)
    probs = normpdf(scores,mu,sigma)
    mask = np.where(scores >= 0)

    return np.trapz(probs[mask],x=scores[mask])

def new_odds(means,st_dev):
    '''
    '''
    teams = means.keys()
    probs = {team: {opp: [get_p(means[team][ndx]-means[opp][ndx],np.sqrt(st_dev[team][ndx]**2+st_dev[opp][ndx]**2)) for ndx in range(10)] for opp in teams if opp != team} for team in teams}
    for team in teams:
        for opp in probs[team].keys():
            probs[team][opp][8] = 1 - probs[team][opp][8]
            probs[team][opp][9] = 1 - probs[team][opp][9]
    probs = {team: {opp: np.sum(np.array([[probs[team][opp][ndx],probs[team][opp][ndx]*probs[opp][team][ndx]] for ndx in range(10)]),axis=0) for opp in teams if opp != team} for team in teams}
    for team in probs.keys():
        probs[team][''] = np.zeros(2)
    return probs

def avg_prob(scores,means,st_devs):
    '''makes a dictionary comparing each team to their average opponent.
    in: scores: dictornary with team names as keys and arrays of previous
            scores as values
    out:    dictionary containing floats representing probability
    '''
    raw = np.concatenate(list(scores.values()))
    mu = mean(raw)
    sigma = stdev(raw)
    probs = {team: get_p(mu-means[team],np.sqrt(sigma**2+st_dev[team]**2)) for team in teams}
    probs[''] = 0

    return probs

# ***** Using Schedule ***** #
def str_of_sch(schedule,avgs,*args):
    '''makes a dictionary of expected wins.
    in: schedule:   dictionary with a key for each team and a corresponding
            value which is a list countaining that teams schedule
        avgs:   dictionary containing floats representing the probability of
            team beating average opponent
        args[0]:    positive int representing the number of weeks left in the
            season --OR-- negative int representing weeks left in the season
    out:    dictionary with a key for each team and a corresponding value which
            is the teams expected wins above average over the specified period
    '''
    if len(args) and args[0] > 0:
        return {team: m_round(sum([avgs[opp] for opp in schedule[team][-1*args[0]:]])-len(list(filter(None,schedule[team][-1*args[0]:])))/2) for team in schedule.keys()}
    elif len(args) and args[0] < 0:
        return {team: m_round(sum([avgs[opp] for opp in schedule[team][:args[0]]])-len(list(filter(None,schedule[team][-1*args[0]:])))/2) for team in schedule.keys()}
    else:
        return {team: m_round(sum([avgs[opp] for opp in schedule[team]])-len(list(filter(None,schedule[team])))/2) for team in schedule.keys()}

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
    sigma = {team: math.sqrt(records[team][1]) for team in records.keys()}
    mu = {team: wins[team]+records[team][0] for team in records.keys()}

    # Normalize to a percent
    mu = {team: mu[team]/(10*len(list(filter(None,schedule[team])))) for team in mu.keys()}
    sigma = {team: sigma[team]/(10*len(list(filter(None,schedule[team])))) for team in mu.keys()}

    return {team: (np.linspace(1,0,fidelity),normpdf(np.linspace(1,0,fidelity),mu[team],sigma[team])) for team in records.keys()}

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

def playoffs(num_teams,hth,cdf,pdfs):
    '''
    '''
    brack = seeding(num_teams)
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
            new_cont[game[0]] =  winners
        contenders = new_cont
    return contenders


def _538_table(sch_csv, scr_csv, oth_csv,plots=False):
    '''
    '''
    scores = parse_scores(scr_csv)
    schedule = parse_schedule(sch_csv)
    ids, wins= parse_other(oth_csv)
    means = dict_map('np.mean',scores)
    stdevs = dict_map('np.std',scores)
    hth = new_odds(means,stdevs)
    weeks_left = len(list(schedule.values())[0])-len(list(scores.values())[0])
    if weeks_left:
        record = predict_record(schedule,hth,weeks_left)
        record_curves = make_record_curves(record,wins,schedule)
        if plots:
            test_plot(record_curves)

        cdf = find_cutoffs(record_curves)
        playoff = find_top(cdf,record_curves,6)
        bye = find_top(cdf,record_curves,2)
        first = playoffs(6,hth,cdf,record_curves)
    else:
        print('Work in progress')
    write_table('BBOdds.csv',['playoffs','first round bye','champion'],playoff, bye, first[1])

def pure_chance(sch_csv, scr_csv, oth_csv):
    '''
    '''
    scores = parse_scores(scr_csv)
    schedule = parse_schedule(sch_csv)
    ids, wins= parse_other(oth_csv)
    teams = ids.keys()
    hth = {team: {opp: np.array([5, 2.5]) for opp in teams if opp != team} for team in teams}
    weeks_left = len(list(schedule.values())[0])-len(list(scores.values())[0])
    if weeks_left:
        record = predict_record(schedule,hth,weeks_left)
        record_curves = make_record_curves(record,wins,schedule)
        cdf = find_cutoffs(record_curves)

        test_plot(record_curves)

        playoff = find_top(cdf,record_curves,6)
        first = playoffs(6,hth,cdf,record_curves)

    print(playoff,first[1])

def _achievers(schedule, hth, wins, scores):
    '''
    '''
    means = dict_map('np.mean',scores)
    stdevs = dict_map('np.std',scores)
    record = predict_record(schedule,hth,len(list(scores.values())[0])-len(list(schedule.values())[0]))
    return {team: m_round(wins[team]-sum(record[team])) for team in wins.keys()}

def main(sch_csv, scr_csv, oth_csv):
    scores = parse_scores(scr_csv)
    schedule = parse_schedule(sch_csv)
    ids, wins= parse_other(oth_csv)
    hth = new_odds(scores)
    record = predict_record(schedule,hth,len(list(schedule.values())[0])-len(list(scores.values())[0]))
    record_curves = make_record_curves(record,wins,schedule)
    cdf = find_cutoffs(record_curves)
    means = dict_map('mean',scores)
    stdevs = dict_map('np.std',scores)

if __name__=="__main__":
   import sys
   main(sys.argv)