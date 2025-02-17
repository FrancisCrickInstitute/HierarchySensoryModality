import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import math
from path import Path
from scipy.stats import rankdata, chi2_contingency, chisquare
from scipy.interpolate import interp1d
from pprint import pprint
import warnings

font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : 16}

mpl.rc('font', **font)

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def predicted_prob(Ra, Rb):

    rank_diff = Rb - Ra
    probability = 1/(1 + 10**(rank_diff/400))

    return probability

def update_elo(Ra_old, Rb_old, result, k=100):
    '''
    Ra_old, Rb_old: existing rating of players A and B to be updated
    Result: integer or float indicating game outcome for player A
    1 for win
    0.5 for draw
    0 for loss

    k = integer constant indicating the maximum number of points transferred between players in one game
    '''

    if result not in [0,0.5,1]:
        raise ValueError('Result must be either 1 (victory), 0.5 (draw) or 0 (loss).')

    Pa = predicted_prob(Ra_old, Rb_old)

    transfer = k*(result - Pa)

    Ra_new = Ra_old + transfer
    Rb_new = Rb_old - transfer

    return Ra_new, Rb_new

def concat_dicts(d1, d2):
    ''' Concatenates two dictionaries d1 and d2 which have the same keys. '''
    combined = {}

    for key in d1.keys():
        combined[key] = np.append(d1[key], d2[key])

    return combined

def stability_index(ranks, ratings, exclude_first = True):
    '''
    Calculates the hierarchy stability index modified from Neuman et al. 2011 over a 2 day sliding window
    S ranges between 0 (completely stable hierarchy) and 1 - large rank reversals every day - complete instability

    Does not work with unequal number of players on different days

    ranks = a dictionary of ranks or other scores (e.g. Elo ratings) with animals as keys
    and their ratings dictionary values formatted as lists/arrays where every element represents the rating on a
    particular date
    '''

    ranks = pd.DataFrame(ranks)
    ratings = pd.DataFrame(ratings)
    Ss = []

    for day in range(len(ranks) - 1):
        # Calculates standardised Elo ratings
        rank_window = ranks.iloc[day:day+2]
        rating_window = ratings.iloc[day:day+2]

        best_elo = rating_window.iloc[0].max()
        worst_elo = rating_window.iloc[0].min()

        num_animals = rank_window.shape[1]

        f = interp1d([worst_elo, best_elo], [0,1])

        sum_rank_changes = np.abs(rank_window.iloc[0] - rank_window.iloc[1]).sum()
        switching_animals = (rank_window.iloc[0] - rank_window.iloc[1]).astype(bool)

        if switching_animals.sum() > 0:
            highest_switcher = rating_window.iloc[0][switching_animals].max()
            weighting = f(highest_switcher)
        else:
            weighting = 0

        S = (sum_rank_changes * weighting) / num_animals
        S = 1 - S
        Ss.append(S)

    if exclude_first:
        return Ss[1:]
    else:
        return Ss



def consistency(df):
    '''
    Compares game outcomes between the same 2 animals over 2 consecutive days as a rolling window and returns a
    consistency score for each day (except the first day where consistency comparison is not possible)

    Consistency score ranges between 0 (all game outcomes between all animal pairs were inconsistent across 2 days)
    and 1 (all game outcomes were consistent across 2 days). The score is an average of the consistency score for each
    animal pair

    df = pandas dataframe containing a record of tests run on each date and the winner + loser of each test

    Returns:
        - consistency = A list of consistency scores for each day (except the first)
    '''

    consistency = []

    dates = np.unique(df['Date'])
    pair_outcomes = []

    for date in dates:

        df_window = df[df['Date'] == date]

        # Assigns each game to an animal pair
        pairs = df_window['Winner'].str.cat(df_window['Loser'], sep='')
        pairs = [''.join(sorted(pair)) for pair in pairs]

        if len(pairs) != df_window.shape[0]: raise Warning('Error in test records for date %s' % date)

        reference = pd.Series(pairs).str[0]
        outcomes = reference == df_window.reset_index(drop=True).Winner

        test = pd.DataFrame(dict(zip(['pair', 'outcome'], [pairs, outcomes])))

        # Find consensus outcome for multiple instances of the same test
        test = test.groupby('pair').mean()
        # Exclude pairs with the same number of wins and losses
        test = test[test.outcome != 0.5]
        test = test > 0.5

        pair_outcomes.append(test)

    for d1, d2 in zip(pair_outcomes[:-1], pair_outcomes[1:]):

        window = d1.merge(d2, on='pair', how='inner')
        consistent = (window.outcome_x == window.outcome_y).sum()
        if window.shape[0] > 0:
            C = consistent / window.shape[0]
        else:
            C = np.nan

        consistency.append(C)

    return consistency



def transitivity(df):

    T = []

    dates = np.unique(df['Date'])
    names = np.unique(df[['Winner', 'Loser']])

    record = []

    for day in dates:
        record = []

        # Check if full round robin was performed
        if df[df['Date'] == day].shape[0] < math.factorial(len(names) - 1):
            T.append(np.nan)
            continue

        for name in names:
            wins = len(np.where(df[df['Date'] == day]['Winner'] == name)[0])

            record.append(wins)

        sorted_wins = sorted(record)
        sorted_wins = np.array(sorted_wins)
        transitive = sorted_wins == np.arange(0, len(record))

        if all(transitive):
            T.append(True)

        else:
            T.append(False)

    return np.array(T)


def process_group_elo(path, score0 = 1000, k=100, accurate_dates = False, plotting = True, printout=False):
    '''
    Calculates the Elo ratings for a group of animals after each game

    Path = a path to an Excel (.xls) file containing the game results.
    File must contain at least 3 columns named "Date", "Winner", and "Loser" which specify the date of each game,
    name of the winner and loser of each game.

    score0 = initial score assigned to all animals

    k = scalar constant which specifies the maximum number of points transferred from the loser to the winner in one game

    accurate_dates = specifies whether to plot the Elo scores with accurate time intervals based on the testing dates.
    If not, intervals between testing points are assumed to be 1 day.

    plotting: boolean; specifies whether ranking plots should be generated
    printout, bool: Specifies whether to print out the result

    :returns
        - ratings
        - ranks
        - C
        - S
        - T
        - dates
    '''

    path = Path(path)

    # Loads the file
    try:
        df = pd.read_excel(path)

        # Finds the animal names and creates a rating and rank record
        names = np.unique(df[['Winner', 'Loser']])
        ratings = [np.array([score0])]*len(names)
        ratings = dict(zip(names,ratings))
        ranks = dict(zip(names, [[(len(names)+1)/2]]*len(names)))

        daily_ratings = ratings.copy()
        dates = np.unique(df['Date'])
        days = [0, 1]

        # Finds time differences between testing dates
        for n in range(len(dates) - 1):
            difference = (dates[n+1] - dates[n]).astype('timedelta64[D]').astype(int)
            days.append(days[-1] + difference)

        # Updates the rating and rank record for every game played in a day
        for date in dates:
            test_day = df[df['Date'] == date]

            # Updates ranking for each game
            for n in range(len(test_day)):

                game = test_day.iloc[n]

                daily_ratings[game['Winner']], daily_ratings[game['Loser']] = update_elo(daily_ratings[game['Winner']],
                                                                                         daily_ratings[game['Loser']],
                                                                                         1, k)

            # Concatenates the new daily ratings to the existing record
            ratings = concat_dicts(ratings, daily_ratings)

            # Converts the most recent ratings from dict to list
            end_of_day = np.array([])
            for animal in daily_ratings.items():
                end_of_day = np.append(end_of_day, [animal[1][-1]])

            # Ranks the players based on the most recent ratings and finds the best and worst player
            daily_ranks = rankdata(end_of_day)
            best = daily_ranks.argmax()
            best = names[best]
            worst = daily_ranks.argmin()
            worst = names[worst]
            daily_ranks = dict(zip(names, daily_ranks))
            ranks = concat_dicts(ranks, daily_ranks)


        S = stability_index(ranks, ratings)
        T = transitivity(df)
        C = consistency(df)

        if printout:
            print('Highest ranking animal is: ' + best)
            print('Lowest ranking animal is: ' + worst + '\n')

            print('Hierarchy transitive over past 3 tests?')
            if all(T[-3:] == True):
                print('✓ \n')
            else:
                print('❌ \n')

            print('Inconsistent outcomes:')
            pprint(log)

        # Saves rankings to new excel file
        #processed = pd.DataFrame(ranks)
        #processed.to_excel(path.parent.parent + '/Processed/' + path.name[:-4] + '_rankings.xls')

        if plotting:
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.right'] = False

            if accurate_dates == False:
                days = np.arange(len(dates) + 1)

            # Plots the ranks
            plt.figure(figsize=[14,11])
            plt.subplot(222)
            for animal in ranks.items():
                plt.plot(days, animal[1], label=animal[0], linewidth=2.5)

            plt.ylabel('Elo rank')
            plt.yticks(np.arange(len(names)) + 1)
            #plt.xticks(days)
            plt.xlabel('Session')

            # Plots the ratings
            plt.subplot(221)
            for animal in ratings.items():
                plt.plot(days, animal[1], label=animal[0], linewidth=2.5)

            plt.legend(ncol = len(names), loc=(0, 0.95), title = 'Mouse ID', frameon=False)
            plt.ylabel('Elo rating')
            plt.xlabel('Session')
            #plt.xticks(days)

            # Plots the consistency scores
            plt.subplot(223)
            plt.plot(days[2:], C)
            plt.ylim([0,1.1])
            plt.ylabel('Consistency Index')
            plt.xlabel('Session')
            plt.xticks(days[2:])

            #Plots the stability index
            plt.subplot(224)
            plt.plot(days[2:], S)
            plt.xticks(days[2:])
            plt.ylim([0,1.1])
            plt.ylabel('Hierarchy Stability Index')
            plt.xlabel('Session')

        return ratings, ranks, C, S, T, dates

    except FileNotFoundError:
        warnings.warn("File not found - Are you connected to CAMP?")
