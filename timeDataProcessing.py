import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


output_folder = "TimeBasedChecks"
os.makedirs(output_folder, exist_ok=True)

def save_bar_plot(missing_values, dataset_name, cleaning_status):
    plt.figure(figsize=(8, 12))
    plt.bar(missing_values.index, missing_values)
    plt.title(f"Missing Values in {dataset_name}{cleaning_status} Cleaning")
    plt.xticks(rotation='vertical')  # Rotate x-axis labels
    plt.savefig(f"{output_folder}/blue{dataset_name.replace(' ', '')}{cleaning_status}.png")
    plt.close()
    
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
    save_bar_plot(missing_values_of_twenty, "Game Data", "Before")

    # Drop rows with missing values in each dataset
    data = data.dropna()
    data = data.drop_duplicates()

    # Check for missing values after cleaning
    missing_values_after = data.isnull().sum()
    save_bar_plot(missing_values_after, "Data", "After")
    
    #checking the average of about how long each game is
    #print(data['fullTimeMS'].mean())
    
    
    # adding a new column called fullTimeMin so the game duration is in mins instead of ms
    data['fullTimeMin'] = data['fullTimeMS'] / 60000
    data['fullTimeMin'] = data['fullTimeMin'].astype(float)

    # Further filtering the dataset based on game duration
    filtered_data = data[(data['fullTimeMin'] >= 25) & (data['fullTimeMin'] <= 30)]

    # dropping the original 'fullTimeMS' column as its no longer needed
    filtered_data = filtered_data.drop('fullTimeMS', axis=1)

    
    filtered_data.to_csv("22-375minof40%.csv", index=False)
    
    #checking how much data remains about 16.2k games remaining
    #print(len(filtered_data))
    

if __name__ == "__main__":
    main()