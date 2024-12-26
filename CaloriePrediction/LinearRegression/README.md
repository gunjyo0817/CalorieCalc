* To directly see the result of calorie prediction, please run: 
```bash
python regression.py
```
* Referred dataset:
    1. train.xml: manually labeled data
    2. test.xml: data processed by segmentation model
    1. train.csv: data used to train regression model, transferred from train.xml
    2. test.csv: data used to test regression model, transferred from test.xml
       
* Illustration of other files:
    1. gen_test_csv.py: generate test.csv from test.xml, which comes from segmentation part
    2. gen_train_csv.py: generate train.csv from train.xml
    3. results folder:
       - performance of linear, quadratic, interactive terms, logarithmic, exponential, cosine

       - .png: the line chart of calorie prediction value vs ground truth of 20 test data

       - .txt: MAE, RMSE, MAPE, R^2, Accuracy < 50, Accuracy < 100, Accuracy < 150(kcal) 

