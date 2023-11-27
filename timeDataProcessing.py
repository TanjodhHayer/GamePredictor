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
    plt.title(f"Missing Values in {dataset_name} Data {cleaning_status} Cleaning")
    plt.xticks(rotation='vertical')  # Rotate x-axis labels
    plt.savefig(f"{output_folder}/blue{dataset_name.replace(' ', '')}{cleaning_status}.png")
    plt.close()
    
def main():
    
    selected_columns = ['matchID', 'fullTimeMS', 'blueChampionKill', 'blueFirstBlood', 'blueDragonKill', 'blueRiftHeraldKill', 
                        'blueTowerKill', 'blueTotalGold', 'blueMinionsKilled', 'blueJungleMinionsKilled', 'blueAvgPlayerLevel', 'blueWin'] # Columns to keep

    # Load datasets with only 25,000 rows from each rank and select columns
    # https://zenodo.org/records/8303397 these datasets are from this site.
    twentyData = pd.read_csv("datasets/timeBased/full_data_40.csv", usecols=selected_columns)
    

    # Display basic information about each dataset
    print("Game data of 20% in Game:")
    print(twentyData.info())



    # Check for missing values in each dataset
    missing_values_of_twenty = twentyData.isnull().sum()
    save_bar_plot(missing_values_of_twenty, "Game Data", "Before")

    # Drop rows with missing values in each dataset
    twentyData = twentyData.dropna()


    # Check the number of rows after dropping missing values
    num_rows_twentyData = len(twentyData)
    
    
    print(num_rows_twentyData)


    
    twentyData.to_csv("full_data_20_cleaned.csv", index=False)

    # Display basic information about the combined dataset
    print("\ Data After Cleaning:")
    print(twentyData.info())

    # Check for missing values after cleaning
    missing_values_after = twentyData.isnull().sum()
    save_bar_plot(missing_values_after, "Data", "After")
    print(twentyData['fullTimeMS'].mean())
    twentyData['fullTimeMinutes'] = twentyData['fullTimeMS'] / 60000
    filtered_data = twentyData[(twentyData['fullTimeMinutes'] >= 25) & (twentyData['fullTimeMinutes'] <= 30)]
    filtered_data.to_csv("22-375minof40%.csv", index=False)
    print(len(filtered_data))

    


if __name__ == "__main__":
    main()