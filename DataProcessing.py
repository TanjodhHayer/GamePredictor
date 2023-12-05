import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    
def main():
    
    selected_columns = ['matchID', 'fullTimeMS', 'blueChampionKill', 'blueFirstBlood', 'blueDragonKill', 
                    'blueRiftHeraldKill', 'blueTowerKill', 'blueTotalGold', 'blueMinionsKilled', 
                    'blueJungleMinionsKilled', 'blueAvgPlayerLevel', 'blueWin', 'redChampionKill', 
                    'redFirstBlood', 'redDragonKill', 'redRiftHeraldKill', 'redTowerKill', 'redTotalGold', 
                    'redMinionsKilled', 'redJungleMinionsKilled', 'redAvgPlayerLevel', 'redWin']

    # Load datasets with only 25,000 rows from each rank and select columns
    # https://zenodo.org/records/8303397 these datasets are from this site.
    data = pd.read_csv("datasets/timeBased/full_data_40.csv", usecols=selected_columns)
    

    # Check for missing values in the dataset
    missing_values_of_twenty = data.isnull().sum()
    print(missing_values_of_twenty, "Game Data")

    # Drop rows with missing values in each dataset
    data = data.dropna()
    data = data.drop_duplicates()
    
    
    
    # adding a new column called fullTimeMin so the game duration is in mins instead of ms
    data['fullTimeMin'] = data['fullTimeMS'] / 60000
    data['fullTimeMin'] = data['fullTimeMin'].astype(float)
    
    # adding new columns to find differences
    data['GoldDiff'] = data['blueTotalGold'] - data['redTotalGold']
    data['ChampionKillsDiff'] = data['blueChampionKill'] - data['redChampionKill']
    data['DragonKillsDiff'] = data['blueDragonKill'] - data['redDragonKill']
    
    # adding new column to find total minions killed
    data['blueMinionsKilledTotal'] = data['blueMinionsKilled'] + data['blueJungleMinionsKilled']
    data['redMinionsKilledTotal'] = data['redMinionsKilled'] + data['redJungleMinionsKilled']
    
    # new column checks difference between blue total minons killed and red total minion killed
    data['MinionsKilledDiff'] = data['blueMinionsKilledTotal'] - data['redMinionsKilledTotal']
    
    # Further filtering the dataset based on game duration
    filtered_data = data[(data['fullTimeMin'] >= 25) & (data['fullTimeMin'] <= 30)]

    # dropping the original 'fullTimeMS' column as its no longer needed
    filtered_data = filtered_data.drop('fullTimeMS', axis=1)

    
    filtered_data.to_csv("22-375minof40%.csv", index=False)
    
    #checking how much data remains about 16.2k games remaining
    #print(len(filtered_data))
    

if __name__ == "__main__":
    main()