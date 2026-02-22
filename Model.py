import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tpot import TPOTClassifier
import time
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
#downloading the dataset
df = pd.read_csv("https://drive.google.com/uc?id=1CppeqGbiBzX61jx56gcXyGGDNEeLlJ-W")
print(df.shape)
print(df.info())
print(df.head())
#We can see that we have categorical variables, we also notice that, unfortunately, there are missing values in all but 5 of the 13 columns.
print(df["Loan_Status"].value_counts())
#There appears to be a 70-30 split for the classes of our target variable loan status. This is not as extreme as say a 95-5 split where we would definitely use something like SMOTE.
#However, we should still note this imbalance and may return to address it depending on model results. 

df = df.drop(columns=["Loan_ID"]) #Loan ID should not have any predictive power at all so we can drop it.
df = df.dropna() # We keep 78% of the rows if we drop the missing values. Not the best way to handle this, but it might work for our needs

# we will need to modify the dependents column for tpot. We will change 3+ dependents to 4 so the entire column can be numeric 
df["Dependents"] = df["Dependents"].replace("3+","4")

LE = LabelEncoder()
y = LE.fit_transform(df["Loan_Status"])
x = df.drop(columns=["Loan_Status"])



x = pd.get_dummies(x, drop_first=False) #df is now all numeric for tpot

#train test split 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size = 0.20,
    random_state = 42,
    stratify = y
)
#running tpot
start_time = time.time()

tpot = TPOTClassifier(
    generations = 2,
    cv= 5,
    random_state= 42,
    verbose= 2,
    max_time_mins=5
)
tpot.fit(x_train, y_train)
end_time = time.time()
print("TPOT classifier finished in %.2f seconds" %(end_time - start_time))

#best pipeline on test set
y_pred = tpot.predict(x_test)
#evaluations
print("Best pipeline test accuracy:",accuracy_score(y_test, y_pred))
print("\nBest pipeline test confusion matrix:\n",confusion_matrix(y_test, y_pred))
print("\nBest pipeline test classification report:\n",classification_report(y_test, y_pred))