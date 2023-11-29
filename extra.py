def Towerkills():
    data['TowerKillsDiff'] = data['blueTowerKill'] - data['redTowerKill']
    
    thresholds = range(0, 3, 1)

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Initialize lists to store win rates and corresponding thresholds for both blue and red teams
    blue_win_rates = []
    red_win_rates = []
    tower_kill_diffs = []

    # Calculate win rates and tower kill differences for each threshold
    for threshold in thresholds:
        blue_condition = data['blueTowerKill'] > threshold
        red_condition = data['redTowerKill'] > threshold
        blue_win_rate = data[blue_condition]['blueWin'].mean() * 100
        red_win_rate = data[red_condition]['redWin'].mean() * 100
        blue_win_rates.append(blue_win_rate)
        red_win_rates.append(red_win_rate)
        tower_kill_diffs.append(threshold)

    # Create DataFrames from the lists
    blue_data_for_line_plot = pd.DataFrame({'TowerKillDiff': tower_kill_diffs, 'BlueWinRate': blue_win_rates})
    red_data_for_line_plot = pd.DataFrame({'TowerKillDiff': tower_kill_diffs, 'RedWinRate': red_win_rates})

    # Line plot for blue team
    sns.lineplot(x='TowerKillDiff', y='BlueWinRate', marker='o', data=blue_data_for_line_plot, color='blue', ax=axes[0])
    axes[0].set_title('Blue Team Win Rates based on Tower Kills Difference Thresholds', fontsize=16)
    axes[0].set_xlabel('Tower Kills Difference Threshold', fontsize=14)
    axes[0].set_ylabel('Blue Team Win Rate (%)', fontsize=14)
    axes[0].tick_params(axis='x')
    axes[0].tick_params(axis='both', labelsize=12)

    # Line plot for red team
    sns.lineplot(x='TowerKillDiff', y='RedWinRate', marker='o', data=red_data_for_line_plot, color='red', ax=axes[1])
    axes[1].set_title('Red Team Win Rates based on Tower Kills Difference Thresholds', fontsize=16)
    axes[1].set_xlabel('Tower Kills Difference Threshold', fontsize=14)
    axes[1].set_ylabel('Red Team Win Rate (%)', fontsize=14)
    axes[1].tick_params(axis='x')
    axes[1].tick_params(axis='both', labelsize=12)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("TowerKill_Difference_Thresholds_LinePlot_Subplots.png")