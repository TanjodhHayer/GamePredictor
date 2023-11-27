import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv('datasets/timeBased/full_data_40.csv')

# Calculate the total number of games
total_games = len(data)

# Calculate the win rate for the blue team
blue_win_rate = data['blueWin'].mean() * 100

print(f"Total Games: {total_games}")
print(f"Blue Win Rate: {blue_win_rate:.2f}%")

# Scatter plot for Kills
sns.scatterplot(x='blueChampionKill', y='redChampionKill', hue='blueWin', data=data)
plt.title('Blue vs. Red in Kills')
plt.xlabel('Blue Team Kills')
plt.ylabel('Red Team Kills')
plt.show()

# Scatter plot for Gold
sns.scatterplot(x='blueTotalGold', y='redTotalGold', hue='blueWin', data=data)
plt.title('Blue vs. Red in Gold')
plt.xlabel('Blue Team Total Gold')
plt.ylabel('Red Team Total Gold')
plt.show()

# Gold difference
data['GoldDiff'] = data['blueTotalGold'] - data['redTotalGold']

# Gold advantage binary feature (20,000 gold threshold example)
data['blueGoldAdv'] = (data['blueTotalGold'] >= 20000).astype(int)

gold_adv_win_rate = data[data['blueGoldAdv'] == 1]['blueWin'].mean() * 100

print(f"Win Rate when Blue Team has Gold Advantage: {gold_adv_win_rate:.2f}%")