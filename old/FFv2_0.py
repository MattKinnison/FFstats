'''Fantasy Football Predictor - Python - v2_0

    By: Matthew Joseph Kinnison (mkinnisonj@gmail.com)

    Runs calculus based 'random' simulations of all the possible results of a
    given season and outputs the results in a format easily usable for owner
    decision making.
'''

# ***** Imported Modules ***** #
import numpy as np
import numpy.ma as ma
import csv
import os.path
from statistics import mean
from statistics import stdev
from matplotlib.mlab import normpdf
import scipy.integrate as calc
import scipy.stats as stats

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
    allowed = ['m_round','make_curve','sum']
    if func in allowed:
        return {key: eval(func)(dict_[key]) for key in dict_.keys()}
    elif func == 'add':
        return {key: dict_[key] + args[0][key] for key in dict_.keys()}
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
    for team in curves.keys():
        plt.plot(curves[team][0],curves[team][1],'k-')

    plt.show()

# ***** Score Curve Functions ***** #
def make_curve(data):
    '''converts list of ints containing point values scored each previous week
        into a normal curve representing future score distribution in the form
        of two numpy arrays, one containing the x coordinates and one
        countaining the y coordinates of the curve. Should be run no more than
        one time per team.
    in: data:   list or array of ints with length > 1
    out:score:  array of length 100 containing the x coordinates of normal curve
        prob:   array of lenght 100 containing the y coordinates of normal curve
        * score and prob packaged in tuple for easy storage in dict later
    '''
    mu = mean(data)
    sigma = stdev(data)

    # Generate curve between 3 standard deviations (99.7% is plenty accurate)
    up_bound = mu + 3*sigma
    lo_bound = mu - 3*sigma

    score = np.linspace(lo_bound,up_bound,100)

    prob = normpdf(score,mu,sigma)

    return (score, prob)

def win_prob(home,away):
    '''generates probability of the home team beating the away team using
        each team's curve and taking a 'double' integral (riemann sum). Should
        be run no more than 1 time per combination of two teams.
        ** This formula is the secret sauce for doing this w/out random loops **
    in: home:   tuple containing two arrays (the x and y coordinates that
            define a probability distribution) (floats)
        away:   tuple containing two arrays (the x and y coordinates that
            define a probability distribution) (floats)
    out:odds:   float (not rounded) with the probability of the home team
            winning as a decimal
    '''
    riem = 0
    for ndx, val in enumerate(home[0]):
        x = np.array([x for x in away[0] if x < val]) # get away less than point
        if len(x) > 1: # make sure away curve is long enough to integrat over
            # sum areas of away behind val multiplied by their probability
            riem += np.trapz(away[1][:len(x)],x=x) * home[1][ndx]

    # normalize by multiplying by stdevs and dividing by constant
    odds = riem * (home[0][99]-home[0][0])/97.056

    return odds

def make_avg_opp(scores):
    '''combines all data to make average opponent curve.
    in: scores: dictionary with team names as keys and arrays of previous scores
            as values
    out:    ** see make_curve **
    '''
    return make_curve(np.concatenate(list(scores.values())))

def make_team_curves(scores):
    '''turns dictonary of teams and scores into a dictionary containing their
        curves.
    in: scores: dictionary with team names as keys and arrays of previous
            scores as values
    out:    dictionary with team names as keys and a tuple with x and y
            coordinates in an array
    '''
    return dict_map('make_curve',scores)

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
            scores = {row[0]: np.array(list(map(float,row[1:]))) for row in reader}

        return scores
    else:
        raise ValueError('{} not found'.format(filename))

def parse_other(filename):
    '''takes in csv containing team ids and their team name, their previous
        wins, and their previous ties in that order with no header. Should be
        run on dataset only once.
    in: filename:   string countaining the full filename of the csv file
    out:ids:    dictonary mapping team ids to their names (string)
        wins:   dictonary mapping team ids to their wins (int)
        ties:   dictonary mapping team ids to their ties (int)
    '''
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            other = [list(row) for row in reader]

        ids = {row[0]: row[1] for row in other}
        wins = {row[0]: int(row[2]) for row in other}
        ties = {row[0]: int(row[3]) for row in other}

        return ids, wins, ties
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
def HtH_prob(curves):
    '''creates a dictionary of dictionaries with the probability that the first
        team (first key) will beat the other (second key).
    in: scores: dictornary with team names as keys and arrays of previous
            scores as values
    out:hth:    dictionary containing dictionaries containing floats
            representing probability
    '''
    hth =  {team: {opp: win_prob(curves[team],curves[opp]) for opp in curves.keys() if opp != team} for team in curves.keys()}
    for team in hth.keys():
        hth[team][''] = 0

    return hth

def avg_prob(scores,curves):
    '''makes a dictionary comparing each team to their average opponent.
    in: scores: dictornary with team names as keys and arrays of previous
            scores as values
    out:    dictionary containing floats representing probability
    '''
    avgs = {team: win_prob(make_avg_opp(scores),curves[team]) for team in scores.keys()}
    avgs[''] = 0

    return avgs

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
        return {team: [hth[team][opp] for opp in schedule[team][-1*args[0]:]] for team in schedule.keys()}
    elif len(args) and args[0] < 0:
        return {team: [hth[team][opp] for opp in schedule[team][:args[0]]] for team in schedule.keys()}
    else:
        return {team: [hth[team][opp] for opp in schedule[team]] for team in schedule.keys()}

def make_record_curves(records,wins,ties,schedule):
    '''
    '''
    sigma = {team: sum([(1-abs(2*x-1))/6 for x in records[team] if x != 0]) for team in records.keys()}
    mu = {team: wins[team]+sum(records[team])+0.5*ties[team] for team in records.keys()}

    # Normalize to a percent
    mu = {team: mu[team]/len(list(filter(None,schedule[team]))) for team in mu.keys()}
    sigma = {team: sigma[team]/len(list(filter(None,schedule[team]))) for team in mu.keys()}

    return {team: (np.linspace(0,1,500),normpdf(np.linspace(0,1,500),mu[team],sigma[team])) for team in records.keys()}

# ***** Getting odds from records ***** #
def find_cutoffs(curves):
    all_curves = np.zeros(500)
    for curve in curves.values():
        all_curves = np.add(all_curves,curve[1])

    return calc.cumtrapz(all_curves,dx=1/500)

def find_old(curves,top):
    '''
    '''
    all_curves = np.zeros(500)
    for curve in curves.values():
        all_curves = np.add(all_curves,curve[1])

    ndx = -1
    area = 0
    while area <= top:
        ndx -= 1
        area = np.trapz(all_curves[:ndx],dx=1/500)

    return {team: np.trapz(curves[team][1][:ndx],dx=1/500) for team in curves.keys()}

def find_top(curves,cdf,top):
    '''
    '''
    mask = np.where(cdf >= top)
    return {team: np.trapz(curves[team][1][mask],dx=1/500) for team in curves.keys()}

def playoffs(rounds,hth,make_playoffs,curves):
    '''
    '''
    seed ={(ndx+1): find_top(curves,ndx+1) for ndx in range(rounds)}

# *************************************************** Temporary
def get_p(mu,sigma):
    scores = np.linspace(mu-3*sigma,mu+3*sigma,100)
    probs = normpdf(scores,mu,sigma)
    mask = np.where(scores > 0)
    return np.trapz(probs[mask],x = scores[mask])

def new_odds(scores):
    teams = scores.keys()
    means = {team: mean(scores[team]) for team in scores.keys()}
    st_dev = {team: stdev(scores[team]) for team in scores.keys()}
    probs = {team: {opp: get_p(means[team]-means[opp],np.sqrt(st_dev[team]**2+st_dev[opp]**2)) for opp in teams if opp != team} for team in teams}
    for team in probs.keys():
        probs[team][''] = 0
    return probs
    # get_p(c_means[team][opp],st_dev[team][opp])

def _538_table(sch_csv, scr_csv, oth_csv):
    '''
    '''
    scores = parse_scores(scr_csv)
    #team_curves = make_team_curves(scores)
    schedule = parse_schedule(sch_csv)
    #hth = HtH_prob(team_curves)
    hth = new_odds(scores)
    ids, wins, ties = parse_other(oth_csv)
    record = predict_record(schedule,hth,len(list(schedule.values())[0])-len(list(scores.values())[0]))
    record_curves = make_record_curves(record,wins,ties,schedule)
    #playoffs =find_old(record_curves,4)
    #consolation =find_old(record_curves,8)
    cdf = find_cutoffs(record_curves)
    playoffs = (find_top(record_curves,cdf,4))
    consolation =(find_top(record_curves,cdf,8))
    write_table('Odds.csv',['consolation','playoffs'],consolation,playoffs)

def _achievers(schedule, hth, wins, scores):
    record = predict_record(schedule,hth,len(list(scores.values())[0])-len(list(schedule.values())[0]))
    return {team: m_round(wins[team]-sum(record[team])) for team in wins.keys()}