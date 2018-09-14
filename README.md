# Baseball Pitch Prediction

## Summary
This data set contains 2011 baseball play by play information. The goal for this project is to build a model that can predict the next pitch. As there are a variety of pitches that a pitcher can throw this will be a multi-class classification model. There is also a big class imbalance between pitches. The most common pitch thrown is a fastball(44%).

Pitch distribution
![Pitch Distribution](https://github.com/haileab/baseball_pitches/blob/master/images/pitch_types.png)

Contour graph of different pitches. That's pretty nice!
![Pitch countour graph](https://github.com/haileab/baseball_pitches/blob/master/images/contour_of_pitches.png)


# Feature Engineering
Created both a pitch percentage and batter percentage of past pitches for each pitcher and batter. These features ranked high in the feature importance of the final model.

## Results
In comparison to the baseline model of simply predicting the most common pitch (fastball) the resulting model improved on both accuracy and f1-score.

--------- Baseline Model

Accuracy =  0.45

            precision    recall  f1-score
avg/total       0.20      0.45      0.28     

---------- Random Forest

Accuracy =  0.57

            precision    recall  f1-score   
avg/total       0.53      0.57      0.49    

## Future Work

- To further improve upon the model of predicting the next pitch we can also perform grid search using ROC-score as the metric to find optimal parameters for our model.
- We can compare results with other classification models.
- I wanted to do the data wrangling for this project, but it is also possible to collect last years data for each player and add that to the model.
- Feature selection can be done to reduce the number of features that are used in the model to improve performance.
- Time series cross validation can also be used to help in improving the training of the model
