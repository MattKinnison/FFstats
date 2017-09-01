'''Fantasy Football Predictor - Python - v1.0'''

# League ID: nfl.l.102612
# My Team ID:359.l.102612.t.9

# Client ID (Consumer Key)
# dj0yJmk9azFZU3E2eDQwejYwJmQ9WVdrOWFUbHpRM0pKTTJVbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD01NA--

# Client Secret (Consumer Secret)
# c3da9fef03ab09a049d322a9aa31f9c2c1a5ffc4

import csv
import statistics as stats
import numpy
import math

sd_range = 3

def make_curve(data):

    mu = stats.mean(data)
    sigma = stats.stdev(data)

    upbound = mu + sd_range * sigma
    lobound = mu - sd_range * sigma

    t = numpy.linspace(lobound,upbound,100)

    p = numpy.exp((-1*(t-mu)**2)/(2*sigma**2))/(sigma*math.sqrt(2*math.pi))

    return (t, p)

def win_prob(h_curve,a_curve):

    odd = 0
    count = 1
    for ndx, val in enumerate(h_curve[0]):
        x = numpy.array([x for x in a_curve[0] if x < val])
        if len(x) > 1:
            odd += (numpy.trapz(a_curve[1][:len(x)],x=x) * h_curve[1][ndx])

    odds = odd * (h_curve[0][99]-h_curve[0][0])/(95)

    return odds

def parse_schedule(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        schedule = {row[0]: row[1:] for row in reader}

    return schedule

def parse_curves(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        curves = {row[0]: make_curve([int(num) for num in row[1:]]) for row in reader}

    return curves

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

    teams = list(schedule.keys())

    if len(teams)-1 < weeks_left:
        h_to_h = {h_team: {a_team: win_prob(curves[h_team],curves[a_team])
                  for a_team in teams}
                  for h_team in teams}
        ex_wins = {team: float('%.2f' % round(sum([h_to_h[team][a_team] for a_team in schedule[team][-1*weeks_left:]]),2))
                   for team in teams}
    else:
        ex_wins = {team: float('%.2f' % round(sum(
            [win_prob(curves[team],curves[opp]) for opp in schedule[team][-1*weeks_left:]]
            ),2)) for team in teams}

    return ex_wins

def main(args):
    print(predict_records(int(args[3]),args[1],args[2]))

if __name__=="__main__":
    import sys
    main(sys.argv)