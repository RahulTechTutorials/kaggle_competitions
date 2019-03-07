import os
import pandas as pd
import seaborn as sns
#%pylab inline  
#import matplotlib._layoutbox as layoutbox

os.chdir('/Users/rahuljain/Desktop/Python/kaggle/titanic')

train_df = pd.read_csv('./input/train.csv', header = 0)
test_df = pd.read_csv('./input/test.csv', header = 0)
train_df.describe()
#train_df.Sex.value_counts().plot(kind='bar')
#sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train_df)

train_df.describe()

def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

# make bins for ages and name them for ease:
def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

#keep only the first letter (similar effect as making bins/clusters):
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

# make bins for fare prices and name them:
def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df


# createa all in transform_features function to be called later:
def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = drop_features(df)
    return df

# create new dataframe with different name:
train_df2 = transform_features(train_df)
test_df2 = transform_features(test_df)



from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
train_df2, test_df2 = encode_features(train_df2, test_df2)
train_df2.head()


X_train = train_df2.drop(["Survived", "PassengerId"], axis=1)
Y_train = train_df2["Survived"]
X_test  = test_df2.drop("PassengerId", axis=1).copy()


X_train.shape, Y_train.shape , X_test.shape

##1) regression 2) decision tree and 3) random forests.


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# Decision Tree

# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# Random Forest

# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest


#Creating a csv with the predicted scores (Y as 0 and 1's for survival)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

# But let's print it first to see if we don't see anything weird:
