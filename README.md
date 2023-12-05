# GamePredictor

## Commands
* pip3 install pandas
* pip3 install numpy
* pip3 install matplotlib
* pip3 install seaborn
* pip3 install statsmodels
* pip3 install vl-convert-python
* pip3 install altair vega_datasets

## Data Processing
-  Loading datasets with 65k games and selected columns.
-  Checking for missing values in datasets and cleaning them by dropping rows of missing values.
-  Feature engineering new columns to plot against win rate.

### Loading Datasets
- The project loads datasets with 65,000 games and selected columns.

### Missing Values
- Checks for missing values in datasets and cleans them by dropping rows with missing values and duplicates


## Analysis and Visualization
The analysis includes:

- Total Gold Distribution
- Kills and Gold Relationships
- Game Duration Analysis
- Impact of Champion Kills on Win Rates
- Impact of Gold Differentials on Win Rates
- Impact of Dragon Kills on Win Rates
- Minions Killed Analysis Win Rates smoothed and unsmoothed


## Training and Modeling
  **Model Training:**
   - Utilizing the Logistic Regression model for training on a subset of the dataset (75% of the data) using the Scikit-Learn library.
 **Model Evaluation:**
   - Assessing the model's performance on a separate test set (25% of the data), yielding an accuracy of 76%.
   - Evaluation Metrics: Computation of precision, recall, and F1-score to provide a comprehensive understanding of the model's performance.
 **Comparison with Alternative Models:**
   - Verifying the robustness of the chosen Logistic Regression model through comparison with alternative models such as K-Nearest Neighbors (KNN) and Random Forest.


## Usage
1. Clone the repository to your local machine.
2. Install the required Python libraries using the provided commands.
3. Run the project python scripts DataProcessing.py --> analyze_data.py and then either models, order doesn't matter

## Contributors
- Tanjodh Hayer
- Sameer Ali


