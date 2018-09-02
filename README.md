# Baseball Pitch Prediction

## Summary
This data set contains 2011 baseball play by play information. The goal for this project is to build a model that can predict the next pitch. As there are a varitey of pitches that a pitcher can throw this will be a multiclass classification model. There is also a big class imbalance between pitches as the most common pitch throw is a fastball. 

## Results
In comparison to the baseline model of simply predicting the most common pitch (fastball) the resulting model improved on both accuracy and f1-score. 

--------- Baseline Model

Accuracy =  0.451442339870156

  precision    recall  f1-score 

avg / total       0.20      0.45      0.28     

---------- Random Forest

Accuracy =  0.5713302325581395

  precision    recall  f1-score   

avg / total       0.53      0.57      0.49    

## Future Work

- To futher improve upon the model of predicting the next pitch we can also perform gridsearch using ROC-score as the metric to find optimal parameters for our model.
- We can compare results with other classification models.
- I wanted to do the data wrangling for this project, but it is also possible to add last years data for each player
- Feature selection can be done to reduce the number of features that are used in the model to improve performance.
- Time series cross validation can also be used to help in improving the training of the model
