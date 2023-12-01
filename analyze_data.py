from matplotlib import gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
# Load the dataset
#data = pd.read_csv('datasets/timeBased/full_data_40.csv')
data = pd.read_csv('22-375minof40%.csv')

# plots of red team vs blue team
def RedVsBlue():
    # Histogram Count for 'blueTotalGold' and 'redTotalGold'
    
    plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 2], height_ratios=[2, 2])

    # Subplot for 'blueTotalGold'
    plt.subplot(gs[0])
    plt.hist(x=data['blueTotalGold'], bins=80, color='skyblue', edgecolor='black')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Blue Total Gold @ 10 - 12mins', fontsize=14)
    plt.ylabel('Count of Games', fontsize=14)
    plt.title('Histogram of Blue Total Gold @ 10 - 12mins', fontsize=16, fontweight='bold')

    # Subplot for 'redTotalGold'
    plt.subplot(gs[1])
    plt.hist(x=data['redTotalGold'], bins=80, color='lightcoral', edgecolor='black')
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel('Red Total Gold @ 10 - 12mins', fontsize=14)
    plt.ylabel('Count of Games', fontsize=14)
    plt.title('Histogram of Red Total Gold @ 10 - 12mins', fontsize=16, fontweight='bold')
    
    # Scatter plot for Gold
    plt.subplot(gs[2:])
    sns.set_palette("deep")
    sns.scatterplot(x='blueTotalGold', y='redTotalGold', hue='blueWin', data=data, size='blueWin', sizes=[50, 150])
    plt.title('Blue vs. Red in Gold @ 10 - 12mins', fontsize=18, fontweight='bold')
    plt.xlabel('Blue Team Total Gold', fontsize=16)
    plt.ylabel('Red Team Total Gold', fontsize=16)
    plt.legend(title='Team', title_fontsize='12', loc='upper right', labels=['Blue', 'Red'])
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('BlueVsRedGold.png')
    


def ChampionKills():
    # Champion Kills advantage binary feature (e.g., more blue champion kills)
    data['blueChampionAdv'] = (data['ChampionKillsDiff'] > 0).astype(int)
    champion_adv_win_rate = data[data['blueChampionAdv'] == 1]['blueWin'].mean() * 100
    print(f"Win Rate when Blue Team has Champion Kill Advantage > 0: {champion_adv_win_rate:.2f}%")

    # Define the thresholds
    thresholds = range(0, 15, 1)


    # Initialize lists to store win rates and corresponding thresholds for both blue and red teams
    blue_win_rates = []
    red_win_rates = []
    champion_kills_diffs = []

    # Calculate win rates and champion kills differences for each threshold
    for threshold in thresholds:
        blue_condition = data['ChampionKillsDiff'] > threshold
        red_condition = data['ChampionKillsDiff'] < -threshold
        blue_win_rate = data[blue_condition]['blueWin'].mean() * 100
        red_win_rate = data[red_condition]['redWin'].mean() * 100
        blue_win_rates.append(blue_win_rate)
        red_win_rates.append(red_win_rate)
        champion_kills_diffs.append(threshold)

    # Create DataFrames from the lists
    blue_data_for_line_plot = pd.DataFrame({'ChampionKillsDiff': champion_kills_diffs, 'BlueWinRate': blue_win_rates})
    red_data_for_line_plot = pd.DataFrame({'ChampionKillsDiff': champion_kills_diffs, 'RedWinRate': red_win_rates})

   # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Line plot for blue team
    sns.lineplot(x='ChampionKillsDiff', y='BlueWinRate', marker='o', data=blue_data_for_line_plot, color='blue', ax=axes[0, 0])
    axes[0, 0].set_title('Blue Team Win Rates based on Champion Kills Difference', fontsize=16,fontweight='bold')
    axes[0, 0].set_xlabel('Champion Kills Difference', fontsize=14)
    axes[0, 0].set_ylabel('Blue Team Win Rate (%)', fontsize=14)
    axes[0, 0].tick_params(axis='x')
    axes[0, 0].tick_params(axis='both', labelsize=12)

    # Line plot for red team
    sns.lineplot(x='ChampionKillsDiff', y='RedWinRate', marker='o', data=red_data_for_line_plot, color='red', ax=axes[1, 0])
    axes[1, 0].set_title('Red Team Win Rates based on Champion Kills Difference', fontsize=16,fontweight='bold')
    axes[1, 0].set_xlabel('Champion Kills Difference', fontsize=14)
    axes[1, 0].set_ylabel('Red Team Win Rate (%)', fontsize=14)
    axes[1, 0].tick_params(axis='x')
    axes[1, 0].tick_params(axis='both', labelsize=12)

    # Scatter plot for Kills
    sns.set_palette("deep")
    sns.scatterplot(x='blueChampionKill', y='redChampionKill', hue='blueWin', data=data, size='blueWin', sizes=[50, 150], ax=axes[0, 1])
    axes[0, 1].set_title('Blue vs. Red in Kills @ 10 - 12mins', fontsize=16, fontweight='bold')
    axes[0, 1].set_xlabel('Blue Team Kills', fontsize=14)
    axes[0, 1].set_ylabel('Red Team Kills', fontsize=14)
    axes[0, 1].legend(title='Team', title_fontsize='12', loc='upper right', labels=['Blue', 'Blue'])
    axes[0, 1].tick_params(axis='x')
    axes[0, 1].tick_params(axis='both', labelsize=12)


    # Bar plot for kill difference in blue game wins
    sns.set_palette("deep")
    
    blue_team_wins = data[data['blueWin'] == 1]['blueWin'].count()
    print(f"Number of Blue Team Wins: {blue_team_wins}")
    
    positive_kills_diff_data = data[data['ChampionKillsDiff'] >= 0]
    sns.countplot(x='ChampionKillsDiff', data=positive_kills_diff_data , ax=axes[1, 1])
    
    axes[1, 1].set_title('Blue Team Wins vs. Champion Kills Difference', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Champion Kills Difference', fontsize=14)
    axes[1, 1].set_ylabel('Number of Games Blue Team Won', fontsize=14)
    axes[1, 1].tick_params(axis='x')
    axes[1, 1].tick_params(axis='both', labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("ChampionKill_Difference_Thresholds_LinePlot_Subplots.png")

def GoldDiff():
    # Define the thresholds (e.g., starting from 0 and increasing by 200 for both teams)
    thresholds = range(-7000, 9000, 600)

    # Initialize lists to store win rates and corresponding thresholds for both blue and red teams
    blue_win_rates = []
    red_win_rates = []
    gold_diffs = []

    # Calculate win rates and gold differences for each threshold
    for threshold in thresholds:
        blue_condition = data['GoldDiff'] > threshold
        red_condition = data['GoldDiff'] < -threshold  # Considering negative gold differences for the red team
        blue_win_rate = data[blue_condition]['blueWin'].mean() * 100
        red_win_rate = data[red_condition]['redWin'].mean() * 100
        blue_win_rates.append(blue_win_rate)
        red_win_rates.append(red_win_rate)
        gold_diffs.append(threshold)

    # Create DataFrames from the lists
    blue_data_for_line_plot = pd.DataFrame({'GoldDiff': gold_diffs, 'BlueWinRate': blue_win_rates})
    red_data_for_line_plot = pd.DataFrame({'GoldDiff': gold_diffs, 'RedWinRate': red_win_rates})

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Line plot for blue team
    sns.lineplot(x='GoldDiff', y='BlueWinRate', marker='o', data=blue_data_for_line_plot, color='blue', ax=axes[0])
    axes[0].set_title('Blue Team Win Rates based on Gold Difference', fontsize=16)
    axes[0].set_xlabel('Gold Difference', fontsize=14)
    axes[0].set_ylabel('Blue Team Win Rate (%)', fontsize=14)
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].tick_params(axis='both', labelsize=12)

    # Line plot for red team
    sns.lineplot(x='GoldDiff', y='RedWinRate', marker='o', data=red_data_for_line_plot, color='red', ax=axes[1])
    axes[1].set_title('Red Team Win Rates based on Gold Difference', fontsize=16)
    axes[1].set_xlabel('Gold Difference', fontsize=14)
    axes[1].set_ylabel('Red Team Win Rate (%)', fontsize=14)
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].tick_params(axis='both', labelsize=12)
    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Gold_Difference_Thresholds_LinePlot_Subplots.png")


def DragonKills():
    # Define the thresholds
    thresholds = [0, 1]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot for Blue Team Win Rate based on Dragon Kills Thresholds
    win_rates = [data[data['blueDragonKill'] > threshold]['blueWin'].mean() * 100 for threshold in thresholds]
    axes[0].bar(thresholds, win_rates, color='darkgreen', alpha=0.7)
    axes[0].set_title('Blue Team Win Rate based on Dragon Kills', fontsize=16)
    axes[0].set_xlabel('Dragon Kills Threshold', fontsize=14)
    axes[0].set_ylabel('Blue Team Win Rate (%)', fontsize=14)
    axes[0].set_xticks(thresholds)
    axes[0].tick_params(axis='both', labelsize=12)

    # Count Plot for 'blueDragonKill' and 'redDragonKill'
    colors = sns.color_palette("pastel")[:2]  # Limiting to two distinct colors
    sns.countplot(x='blueDragonKill', hue='redDragonKill', data=data, palette=colors, ax=axes[1])
    axes[1].set_xlabel('Dragon Kills @ 10 - 12mins', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count of Games', fontsize=14, fontweight='bold')
    axes[1].set_title('Dragon Kills by Blue and Red Teams @ 10 - 12mins', fontsize=16, fontweight='bold')
    axes[1].legend(title='Team', title_fontsize='12', loc='upper right', labels=['Blue', 'Red'])
    axes[1].tick_params(axis='both', labelsize=12)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("Dragon_Kills_Subplots.png")






def minionsKilledTotal():
    # Calculate win rate for each value of blueMinionsKilledTotal
    blue_minions_data_for_line_plot = (
        data.groupby('blueMinionsKilledTotal')['blueWin']
        .mean()
        .reset_index()
        .rename(columns={'blueMinionsKilledTotal': 'MinionsKilledTotal', 'blueWin': 'WinRate'})
    )
    
    # Calculate win rate for each value of redMinionsKilledTotal
    red_minions_data_for_line_plot = (
        data.groupby('redMinionsKilledTotal')['redWin']
        .mean()
        .reset_index()
        .rename(columns={'redMinionsKilledTotal': 'MinionsKilledTotal', 'redWin': 'WinRate'})
    )
    
    # Calculate win rate for each difference value
    minions_diff_data_for_line_plot = (
        data.groupby('MinionsKilledDiff')['blueWin']
        .mean()
        .reset_index()
        .rename(columns={'blueWin': 'WinRateDiff'})
    )
    
    plt.figure(figsize=(30, 25))
    
    frac=0.035
    # Smooth the data using lowess
    smoothed_values = lowess(blue_minions_data_for_line_plot['WinRate'], blue_minions_data_for_line_plot['MinionsKilledTotal'], frac=frac)[:, 1] * 100
    blue_minions_data_for_line_plot_smoothed = pd.DataFrame({'MinionsKilledTotal': blue_minions_data_for_line_plot['MinionsKilledTotal'], 'SmoothedWinRate': smoothed_values})
    
    smoothed_values_red = lowess(red_minions_data_for_line_plot['WinRate'], red_minions_data_for_line_plot['MinionsKilledTotal'], frac=frac)[:, 1]* 100
    red_minions_data_for_line_plot_smoothed = pd.DataFrame({'MinionsKilledTotal': red_minions_data_for_line_plot['MinionsKilledTotal'], 'SmoothedWinRate': smoothed_values_red})
    
    smoothed_values_diff = lowess(minions_diff_data_for_line_plot['WinRateDiff'], minions_diff_data_for_line_plot['MinionsKilledDiff'], frac=frac)[:, 1]* 100
    minions_diff_data_for_line_plot_smoothed = pd.DataFrame({'MinionsKilledDiff': minions_diff_data_for_line_plot['MinionsKilledDiff'], 'SmoothedWinRate': smoothed_values_diff})

    # Plot for blue team
    plt.subplot(3, 2, 1)
    sns.lineplot(x='MinionsKilledTotal', y='SmoothedWinRate', data=blue_minions_data_for_line_plot_smoothed, color='green')
    plt.title('Blue Team Win Rate based on Minions Killed Total', fontsize=16)
    plt.xlabel('Minions Killed Total', fontsize=14)
    plt.ylabel('Blue Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    
    '''

    Moving average smoothing using rolling
    plt.subplot(3, 2, 1)
    blue_minions_data_for_line_plot['WinRate_MA'] = blue_minions_data_for_line_plot['WinRate'].rolling(window=9).mean()
    sns.lineplot(x='MinionsKilledTotal', y='WinRate_MA', data=blue_minions_data_for_line_plot, color='green', label='Moving Average')
    plt.title('Blue Team Win Rate based on Minions Killed Total (Moving Average)', fontsize=16)
    plt.xlabel('Minions Killed Total', fontsize=14)
    plt.ylabel('Blue Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    
    '''

    # Plot for red team
    plt.subplot(3, 2, 3)
    sns.lineplot(x='MinionsKilledTotal', y='SmoothedWinRate', data=red_minions_data_for_line_plot_smoothed, color='red')
    plt.title('Red Team Win Rate based on Minions Killed Total', fontsize=16)
    plt.xlabel('Minions Killed Total', fontsize=14)
    plt.ylabel('Red Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)


    
    # Plot the difference
    plt.subplot(3, 2, 5)
    sns.lineplot(x='MinionsKilledDiff', y='SmoothedWinRate', data=minions_diff_data_for_line_plot_smoothed, color='orange')
    plt.title('Blue Team Win Rate based on Minions Killed Difference (Blue - Red)', fontsize=16)
    plt.xlabel('Minions Killed Difference', fontsize=14)
    plt.ylabel('Blue Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    
    # Plot for blue team without smoothing
    blue_minions_data_for_line_plot['WinRateScaled'] = blue_minions_data_for_line_plot['WinRate'] * 100
    red_minions_data_for_line_plot['WinRateScaled'] = red_minions_data_for_line_plot['WinRate'] * 100
    minions_diff_data_for_line_plot['WinRateDiffScaled'] = minions_diff_data_for_line_plot['WinRateDiff'] * 100
    
    plt.subplot(3, 2, 2)
    sns.lineplot(x='MinionsKilledTotal', y='WinRateScaled', data=blue_minions_data_for_line_plot, color='green', marker='o', markersize=8)
    plt.title('Blue Team Win Rate based on Minions Killed Total', fontsize=16)
    plt.xlabel('Minions Killed Total', fontsize=14)
    plt.ylabel('Blue Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    # Plot for red team without smoothing
    plt.subplot(3, 2, 4)
    sns.lineplot(x='MinionsKilledTotal', y='WinRateScaled', data=red_minions_data_for_line_plot, color='red',marker='o', markersize=8)
    plt.title('Red Team Win Rate based on Minions Killed Total', fontsize=16)
    plt.xlabel('Minions Killed Total', fontsize=14)
    plt.ylabel('Red Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    # Plot the difference without smoothing
    plt.subplot(3, 2, 6)
    sns.lineplot(x='MinionsKilledDiff', y='WinRateDiffScaled', data=minions_diff_data_for_line_plot, color='orange',marker='o', markersize=8)
    plt.title('Blue Team Win Rate based on Minions Killed Difference (Blue - Red)', fontsize=16)
    plt.xlabel('Minions Killed Difference', fontsize=14)
    plt.ylabel('Blue Team Win Rate (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
        
    
    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("MinionsKilledTotal_WinRate_LinePlot_Subplots_Smoothed.png")

    


def main():

    # Calculate the total number of games
    total_games = len(data)

    # Calculate the win rate for the blue team
    blue_win_rate = data['blueWin'].mean() * 100
    red_win_rate = data['redWin'].mean() * 100
    print(f"Total Games: {total_games}")
    print(f"Blue Win Rate: {blue_win_rate:.2f}%")
    print(f"Red Win Rate: {red_win_rate:.2f}%")

    minionsKilledTotal()
    DragonKills()
    GoldDiff()
    ChampionKills()
    RedVsBlue()


if __name__ == "__main__":
    main()