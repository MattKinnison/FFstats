'''Fantasy Football Predictor - Python - v1.0'''

import csv
import statistics as stats
import numpy
import math

def make_curves(data):

    mu = stats.mean(data)
    sigma = stats.stdev(data)

    upbound = mu + 3.5 * sigma
    lobound = mu - 3.5 * sigma

    t = numpy.linspace(lobound,upbound,100)

    p = numpy.exp((-1*(t-mu)**2)/(2*sigma**2))/(sigma*math.sqrt(2*math.pi))

    return (t, p)

def parse_schedule(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        schedule = {row[0]: row[1:] for row in reader}

    return schedule

def parse_curves(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        curves = {row[0]: make_curves([int(num) for num in row[1:]]) for row in reader}

    return curves

'''def win_prob(h_curve,a_curve):

    odd = []
    for val in h_curve[0]:
        x = filter(lambda x: x < val, a_curve[0])
        if len(x) > 0
            of

    return'''

def predict_records(weeks_left, sch_csv, dat_csv):
    '''Compute expected value of games won for each team provided:
        * weeks_left - [int] - number of weeks remaining in season
        * sch_csv - [str] - filename of a csv containing the schedule for each
            team
        * dat_csv - [str] - filename of a csv coutaining previous scores for
            each team
    '''

    # Read schedule into dictionary
    schedule = parse_schedule(sch_csv)
    # Read data into curves dictionary
    curves = parse_curves(dat_csv)

    ex_wins = {}
    for team in schedule.keys():
        opponents  = schedule[team][-1*weeks_left:]
        odds = [win_prob(curves[team],curves[opp]) for opp in opponents]
        ex_wins[team] = sum(odds)

    return ex_wins

def main(args):
    print(predict_records(int(args[3]),args[1],args[2]))

if __name__=="__main__":
    import sys
    main(sys.argv)