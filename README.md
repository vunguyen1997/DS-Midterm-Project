# Data Science Midterm Project

## Project/Goals
The aim of this inquiry is to use pre-existing US house sales data to create a predictive model
to accurately gauge future house prices.


## Process


### Phase I Knowing the data

- Open housing information inside of JSON files
- Checked file structure and made function to loop through all JSON file and extract house data
- All house data was then grouped together and converted into a normalized dataframe
- Unnecessary columns were then dropped and null values replaced

- tags were then parsed, everything except 10 most frequent tags were dropped and finally encoded 
- Mean cities & states sale prices was calculated and encoded into the train data to keep track of the location without having to make custom encodings for each city and state.

### Phase II Visualize the data
-  A heatmap and a pairwise plots were created to view the correlation between the different features in the dataframe

![Heatmap](images/heatmap.png)
![PairPlot](images/pairplot.png)

### Phase III  Model Selection and Training

- The numerical columns of the processed dataframe are used to train LinearRegression, Ridge, Decision Trees, and RandomForrest models
- The outputs of the trained model are compared to one another to select the performing one.
- The selected model(RandomForrest) under went cross-validation and hypertuning to further refine the model. 


## Results

After having hypertuned and corrected any underlying discrepancies with custom cross-validations the current iteration of our model doesn't yield a strong enough success rate to warrent commercial use.

Fitting 5 folds for each of 27 candidates, totalling 135 fits
🔍 Best params: {'max_depth': 20, 'min_samples_leaf': 2, 'n_estimators': 100}
🔍 Best CV RMSE: 410,120.31
🏁 Final Test RMSE: 525,311.80


## Challenges 
The biggest challenge in this project was manipulating the data into a workable dataframe for analysis while minimizing loss of useful data.The source files had a lot of redundant features and null values that necessacitated multiple rounds of trimming, null replacements, and encoding to be fed to ML models.

## Future Goals
A way to increase the success rate of the current MVP model would be to include socioeconomic data from the cities and states. As they could be factors that weight more in the areas home evaluation.
