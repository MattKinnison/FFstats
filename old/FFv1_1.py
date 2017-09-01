'''Fantasy Football Predictor - Python - v1.1'''

# League ID: nfl.l.102612
# My Team ID:359.l.102612.t.9

# Client ID (Consumer Key)
# dj0yJmk9azFZU3E2eDQwejYwJmQ9WVdrOWFUbHpRM0pKTTJVbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD01NA--

# Client Secret (Consumer Secret)
# c3da9fef03ab09a049d322a9aa31f9c2c1a5ffc4

import csv
import statistics as stats
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

sd_range = 3

def make_curve(data):

    mu = stats.mean(data)
    sigma = stats.stdev(data)

    upbound = mu + sd_range * sigma
    lobound = mu - sd_range * sigma

    t = np.linspace(lobound,upbound,100)

    p = mlab.normpdf(t,mu,sigma)

    return (t, p)

def win_prob(h_curve,a_curve):

    odd = 0
    count = 1
    for ndx, val in enumerate(h_curve[0]):
        x = np.array([x for x in a_curve[0] if x < val])
        if len(x) > 1:
            odd += (np.trapz(a_curve[1][:len(x)],x=x) * h_curve[1][ndx])

    odds = odd * ((h_curve[0][99]-h_curve[0][0])/6)/(15.83333333333333)

    return odds

def m_round(num):

        return float('%.2f' % num)

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

def get_week(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        weeks_in = len(list(reader)[0])-1

    return weeks_in

def parse_ids(filename):

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        ids = {row[0]: {'Team Name': row[1], 'Wins': int(row[2]), 'Ties': int(row[3])} for row in reader}

    return ids

def write_table(cons,offs,semi,final):

    filename = 'Odds.csv'

    with open(filename,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Team','Make Consolation','Make Playoff','Make Final','Win League'])
        for key in cons.keys():
            writer.writerow([key, cons[key], offs[key], semi[key], final[key]])

    return filename

def predict_records(weeks_left, sch_csv, scr_csv):
    '''Compute expected value of games won for each team provided:
        * weeks_left - [int] - number of weeks remaining in season
        * sch_csv - [str] - filename of a csv containing the schedule for each
            team
        * scr_csv - [str] - filename of a csv coutaining previous scores for
            each team
    '''

    # Read schedule into dictionary
    schedule = parse_schedule(sch_csv)
    # Read data into curves dictionary
    curves = parse_curves(scr_csv)

    teams = list(schedule.keys())

    if len(teams)-1 < weeks_left:
        h_to_h = {h_team: {a_team: win_prob(curves[h_team],curves[a_team])
                  for a_team in teams}
                  for h_team in teams}
        ex_wins = {team: float('%.2f' % round(sum(
            [h_to_h[team][a_team] for a_team in schedule[team][-1*weeks_left:]]
            ),2)) for team in teams}
    else:
        ex_wins = {team: float('%.2f' % round(sum(
            [win_prob(curves[team],curves[opp]) for opp in schedule[team][-1*weeks_left:]]
            ),2)) for team in teams}

    return ex_wins

def predict_remaining(weeks_left, sch_csv, scr_csv,ids_csv):
    '''Compute expected value of games won for each team provided:
        * weeks_left - [int] - number of weeks remaining in season
        * sch_csv - [str] - filename of a csv containing the schedule for each
            team
        * scr_csv - [str] - filename of a csv coutaining previous scores for
            each team
    '''

    # Read schedule into dictionary
    schedule = parse_schedule(sch_csv)
    # Read data into curves dictionary
    curves = parse_curves(scr_csv)
    # Read data in team id dictionary
    ids = parse_ids(ids_csv)

    teams = list(schedule.keys())

    if len(teams)-1 < weeks_left:
        h_to_h = {h_team: {a_team: win_prob(curves[h_team],curves[a_team])
                  for a_team in teams}
                  for h_team in teams}
        ex_wins = {team:
            [h_to_h[team][a_team] for a_team in schedule[team][-1*weeks_left:]]
            for team in teams}
    else:
        ex_wins = {team:
            [m_round(win_prob(curves[team],curves[opp])) for opp in schedule[team][-1*weeks_left:]]
             for team in teams}

    return {ids[team]['Team Name']: ex_wins[team] for team in teams}

def find_top(means, st_dev, top_):

    x = np.linspace(0,1,200)

    curves = {team: mlab.normpdf(x,means[team],st_dev[team]) for team in means.keys()}

    all_curves = np.zeros(200)
    for curv in curves.values():
        #plt.plot(x, curv,'k-')
        all_curves = np.add(all_curves,curv)

    #plt.plot(x, all_curves,'g-')
    #plt.show()

    area = 0
    ndx = -1
    while area < top_:
        ndx -= 1
        area = np.trapz(all_curves[ndx:], x=x[ndx:])

    percents = {team: np.trapz(curves[team][ndx:],x=x[ndx:]) for team in curves.keys()}

    return percents

def playoffs(curves, means, st_dev):

    # Find each seed for the tourney
    seed_1 = find_top(means, st_dev, 1)
    seed_2 = {team: find_top(means, st_dev, 2)[team]-seed_1[team] for team in means.keys()}
    seed_3 = {team: find_top(means, st_dev, 3)[team]-seed_2[team]-seed_1[team] for team in means.keys()}
    seed_4 = {team: find_top(means, st_dev, 4)[team]-seed_3[team]-seed_2[team]-seed_1[team] for team in means.keys()}

    odds_1 = {team: sum([seed_1[team]*seed_4[a_team]*win_prob(curves[team],curves[a_team]) for a_team in seed_4.keys()]) for team in seed_1.keys()}
    odds_2 = {team: sum([seed_2[team]*seed_3[a_team]*win_prob(curves[team],curves[a_team]) for a_team in seed_3.keys()]) for team in seed_2.keys()}
    odds_3 = {team: sum([seed_3[team]*seed_2[a_team]*win_prob(curves[team],curves[a_team]) for a_team in seed_2.keys()]) for team in seed_3.keys()}
    odds_4 = {team: sum([seed_4[team]*seed_1[a_team]*win_prob(curves[team],curves[a_team]) for a_team in seed_1.keys()]) for team in seed_4.keys()}

    semi_results = {team: sum([odds_1[team],odds_2[team],odds_3[team],odds_4[team]]) for team in means.keys()}

    final_results = {team: semi_results[team]*(sum([semi_results[a_team]*win_prob(curves[team],curves[a_team]) for a_team in means.keys()])-semi_results[team]) for team in means.keys()}

    return semi_results, final_results

def predict_percents(sch_csv, scr_csv, dat_csv):

    # Read schedule into dictionary
    schedule = parse_schedule(sch_csv)
    # Read scores into curves dictionary
    curves = parse_curves(scr_csv)
    # Read data in team id dictionary
    ids = parse_ids(dat_csv)


    season_length = len(schedule[list(schedule.keys())[0]])
    weeks_left = season_length - get_week(scr_csv)
    teams = list(schedule.keys())

    if len(teams)-1 < weeks_left:
        h_to_h = {h_team: {a_team: win_prob(curves[h_team],curves[a_team])
                  for a_team in teams}
                  for h_team in teams}
        ex_by_game = {team:
            [h_to_h[team][a_team] for a_team in schedule[team][-1*weeks_left:]]
            for team in teams}
        sd_wins = {team: sum(
            [(1-abs(2*x-1))/6 for x in ex_by_game[team]]) for team in teams}
        ex_wins = {team: ids[team]['Wins'] + sum(ex_by_game[team]) for team in teams}
    else:
        ex_by_game = {team:
            [win_prob(curves[team],curves[opp]) for opp in schedule[team][-1*weeks_left:]]
            for team in teams}
        # sd found using chevychev on who system
        sd_wins = {team: sum(
            [((1-abs(2*x-1))/6) for x in ex_by_game[team]]) for team in teams}
        ex_wins = {team: ids[team]['Wins'] + sum(ex_by_game[team]) for team in teams}

    # Normalize (for ties)
    ex_per = {team: ex_wins[team]/(season_length-ids[team]['Ties']) for team in teams}
    sd_per = {team: sd_wins[team]/(season_length-ids[team]['Ties']) for team in teams}

    make_playoff = find_top(ex_per, sd_per, 4)
    make_consolation = find_top(ex_per, sd_per, 8)
    semi_results, winner = playoffs(curves,ex_per,sd_per)

    cons = {ids[team]['Team Name']: m_round(make_consolation[team]) for team in teams}
    play = {ids[team]['Team Name']: m_round(make_playoff[team]) for team in teams}
    semi = {ids[team]['Team Name']: m_round(semi_results[team]) for team in teams}
    final = {ids[team]['Team Name']: m_round(winner[team]) for team in teams}

    write_table(cons,play,semi,final)

#def main(args):


#if __name__=="__main__":
 #   import sys
  #  main(sys.argv)