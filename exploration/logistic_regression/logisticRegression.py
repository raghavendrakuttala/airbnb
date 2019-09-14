import airbnb.airbnb.utils as utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,f1_score

train_df,test_df, age_gender_df,countries_df,session_df =  utils.load_data("../exploration/data/")

train_df = utils.training_feature(train_df)
print(train_df.shape,test_df.shape)

#one hot encoding
categorical_features = list(train_df.select_dtypes('object').columns)
categorical_features.append('signup_flow')
categorical_features.remove('id')
categorical_features.remove('country_destination')
print(categorical_features,train_df.columns)
train_df = pd.get_dummies(train_df,columns=categorical_features,drop_first=True)

X_train,X_test,y_train,y_test = train_test_split(train_df.drop(['country_destination','id'],axis=1),train_df['country_destination'],test_size=0.3,random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
lg = LogisticRegression().fit(X_train,y_train)

lg.score(X_test,y_test)
y_pred = lg.predict(X_test)
confusion_matrix(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average='weighted')
print(f1,'f1 score')