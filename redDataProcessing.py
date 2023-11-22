import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_folder = "redChecks"
os.makedirs(output_folder, exist_ok=True)

def save_bar_plot(missing_values, dataset_name, cleaning_status):
    plt.figure(figsize=(8, 12))
    plt.bar(missing_values.index, missing_values)
    plt.title(f"Missing Values in {dataset_name} Data {cleaning_status} Cleaning")
    plt.xticks(rotation='vertical')  # Rotate x-axis labels
    plt.savefig(f"{output_folder}/red{dataset_name.replace(' ', '')}{cleaning_status}.png")
    plt.close()
    
def main():
    selected_columns = ['gameId','gameDuraton', 'redWins', 'redFirstBlood', 'redFirstTower', 'redFirstBaron', 'redFirstDragon', 'redFirstInhibitor', 'redDragonKills', 
                        'redBaronKills', 'redTowerKills', 'redInhibitorKills', 'redWardPlaced', 'redWardkills', 'redKills', 'redAssist', 'redChampionDamageDealt', 
                        'redChampionDamageDealt', 'redChampionDamageDealt', 'redTotalGold', 'redTotalMinionKills', 'redTotalLevel', 'redAvgLevel', 'redJungleMinionKills', 
                        'redKillingSpree', 'redTotalHeal', 'redObjectDamageDealt']  # Columns to keep

    # Load datasets with only 25,000 rows from each rank and select columns
    challenger_data = pd.read_csv("datasets/Challenger_Ranked_Games.csv", usecols=selected_columns).sample(n=25000, random_state=42)
    grandmaster_data = pd.read_csv("datasets/GrandMaster_Ranked_Games.csv", usecols=selected_columns).sample(n=25000, random_state=42)
    master_data = pd.read_csv("datasets/Master_Ranked_Games.csv", usecols=selected_columns).sample(n=25000, random_state=42)


    # Display basic information about each dataset
    print("Challenger Data Before Cleaning:")
    print(challenger_data.info())

    print("\nGrandmaster Data Before Cleaning:")
    print(grandmaster_data.info())

    print("\nMaster Data Before Cleaning:")
    print(master_data.info())

    # Check for missing values in each dataset
    missing_values_challenger = challenger_data.isnull().sum()
    save_bar_plot(missing_values_challenger, "Challenger", "Before")

    missing_values_grandmaster = grandmaster_data.isnull().sum()
    save_bar_plot(missing_values_grandmaster, "Grandmaster", "Before")

    missing_values_master = master_data.isnull().sum()
    save_bar_plot(missing_values_master, "Master", "Before")

    # Drop rows with missing values in each dataset
    challenger_data = challenger_data.dropna()
    grandmaster_data = grandmaster_data.dropna()
    master_data = master_data.dropna()

    # Check the number of rows after dropping missing values
    num_rows_challenger = len(challenger_data)
    num_rows_grandmaster = len(grandmaster_data)
    num_rows_master = len(master_data)
    
    print(num_rows_challenger)
    print(num_rows_grandmaster)
    print(num_rows_master)

    # Find the minimum number of rows among the datasets
    min_num_rows = min(num_rows_challenger, num_rows_grandmaster, num_rows_master)

    # Ensure that each dataset has the same number of rows
    challenger_data = challenger_data.sample(n=min_num_rows, random_state=42)
    grandmaster_data = grandmaster_data.sample(n=min_num_rows, random_state=42)
    master_data = master_data.sample(n=min_num_rows, random_state=42)

    # Combine datasets into one
    all_data = pd.concat([challenger_data, grandmaster_data, master_data], ignore_index=True)
    
    all_data.to_csv("red_combined_cleaned_dataset.csv", index=False)

    # Display basic information about the combined dataset
    print("\nCombined Data After Cleaning:")
    print(all_data.info())

    # Check for missing values after cleaning
    missing_values_after = all_data.isnull().sum()
    save_bar_plot(missing_values_after, "Combined", "After")

    


if __name__ == "__main__":
    main()