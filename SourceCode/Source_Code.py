import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split , GridSearchCV , RandomizedSearchCV
from sklearn.preprocessing import StandardScaler  , PolynomialFeatures # Ensure PolynomialFeatures is here
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import  Pipeline
from scipy.stats import uniform , randint
from xgboost import XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error , r2_score , f1_score , precision_score , accuracy_score , mean_squared_error      
from datetime import date
import joblib
from used_Mehods import Date_Calculation


data = pd.read_csv('../DataSet/dataset_file.csv' , sep=',')
print(data.head())
 
data.info()
print(data.isnull().sum())
    
data.dropna(inplace=True)
     
###                *** Important Step ***
# convert the date values to period as we will compute the period that the last coefficiently 
# as this helps us to detect which causes the clients to reduce treating with the company
# which helps us to make strategies to increase the number of clients , and increase the
# duration of their treatment with the company as this increases the revenues

     
# The code below used to aggregate financial data and print summaries of the results.
# It calculates the total daily revenue for each product category present in a DataFrame named data.
      
data_revenue_by_category = data.groupby(['Date' , 'Category'])['Daily Revenue'].sum().reset_index()
print(f'The Unique Values in Category are : {data['Category'].unique()} ' , end=f'\n{'*'*50}\n')
print(f'\n{'*'*50} As an Showing For Data After Grouping :- ')
print(f'\n{"-"* 50}\n {data_revenue_by_category.head(20)}' , end=f'\n{"-"*50}\n')
     
def Date_Calculation(date_sent : date) :   # used to compute the date from certain date dedicated by user untill this day
  res =  date.today() - date_sent
  return  res.days/365  # used to calcualte the number of years like 4.53 years
   
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['Date'] )

data["Date_Of_Day"] = data['Date'].dt.day   # extract the weekends
data["Month_Number"] = data['Date'].dt.month   # extract the weekends
data["Year_Number"] = data['Date'].dt.year   # extract the weekends
data["DayOfWeek_number"] = data['Date'].dt.dayofweek   # extract the weekends

data['Date'] = data['Date'].dt.date
data['Last_treatment_Period_In_Years'] = data['Date'].apply(Date_Calculation)  # 2.57

data["Revenue Change"]  = data['Daily Revenue'].diff()    # here we will now if the daily revenue in that day greater than in the last day
data["Ad_to_Revenue_Ratio"] =  data["Ad Spend"] / (data["Daily Revenue"] + 1) # here we will show of the ad_spend  has an effect on the daily revenue or not
data.dropna(inplace=True)

     
data.drop(columns=['Date'] , inplace = True , errors='ignore') # Removed index=1 to avoid errors if dataframe is small

          
plt.figure(figsize = (10 , 13))
sns.scatterplot(x = 'Ad_to_Revenue_Ratio' ,  y ="Daily Revenue" ,  data=data  , color='blue' ,  alpha= 0.6)
plt.title('Relationship between Ad_to_Revenue_Ratio and Daily Revenue')
plt.xlabel('Ad_to_Revenue_Ratio')
plt.ylabel(' Daily Revenue')
plt.grid(True)
plt.savefig('advertisement_spend_effect_daily_revenue_chart.png' , dpi = 300 )
plt.show()

     
plt.figure(figsize = (10 , 13))
sns.countplot(data = data , x = 'Service Type')
plt.title('The number of occurrence for each Service Type')
plt.xlabel('Service Type')
plt.savefig('service_type_chart.png' , dpi = 300 )
plt.show()

     
plt.figure(figsize = (10 , 13))
sns.countplot(data = data , x = 'Category')
plt.title('number of users per each Category')
plt.xlabel('Category Type')
plt.savefig('Category_type_chart.png' , dpi = 300 )
plt.show()

     
plt.figure(figsize = (10 , 13))
sns.countplot(data = data , x = 'Time of Day')
plt.title('number of users during different periods')
plt.xlabel('Time of Day')
plt.savefig('time_of_day_chart.png' , dpi = 300 )
plt.show()

     
plt.figure(figsize = (10 , 13))
sns.countplot(data = data , x = 'Platform')
plt.title('number of users from different platforms')
plt.xlabel('Platform type')
plt.savefig('platform_chart.png' , dpi = 300 )
plt.show()

     
plt.figure(figsize = (10 , 13))
sns.histplot(data = data, x='Daily Revenue',  bins=30, kde=True , color = 'royalblue' , alpha = 0.8)
plt.title('Effect of Daily Revenue during days')
plt.xlabel('Daily Revenue ($)'  , fontsize = 12)
plt.ylabel('Count'  , fontsize = 12)
plt.tight_layout()
plt.savefig('Comparison_Time_of_Day_ON Daily_Revenue.png' , dpi = 300 )

     
plt.figure(figsize = (10 , 13))
sns.histplot(data = data, x='Daily Revenue' , hue = 'Customer Type',  bins=30, kde=True  , multiple = 'stack', edgecolor = 'royalblue' , alpha = 0.85)
plt.title('Effect of Customer Type on Daily Revenue')
plt.xlabel('Daily Revenue ($)'  , fontsize = 12)
plt.ylabel('Count'  , fontsize = 12)
plt.tight_layout()
plt.savefig('Comparison_Time_of_Day_ON Daily_Revenue.png' , dpi = 300 )

     
plt.figure(figsize = (10 , 13))
sns.histplot(data = data, x='Daily Revenue' , hue = 'Platform',  bins=30, kde=True  , multiple = 'stack', edgecolor = 'royalblue' , alpha = 0.85)
plt.title('Effect of Platform Type on Daily Revenue')
plt.xlabel('Daily Revenue ($)'  , fontsize = 12)
plt.ylabel('Count'  , fontsize = 12)
plt.tight_layout()
plt.savefig('Comparison_Time_of_Day_ON Daily_Revenue.png' , dpi = 300 )

     
plt.figure(figsize = (12 , 12))
sns.countplot(data=data ,  x='Category' , hue = 'Platform'  , color='blue')
plt.xlabel('Category')
plt.ylabel('Platform')
plt.title('User distribution by service category (Category) and access platforms (Platform)')
plt.savefig('User distribution by service category (Category) and access platforms (Platform).png')
plt.show()

     
print(data['Time of Day'].unique())
print(data['Category'].unique())
print(data['Service Type'].unique())
print(data['Customer Type'].unique())
print(data['Platform'].unique())


data['Service Type'] = data['Service Type'].fillna(data['Service Type'].mode())

print(f'\n{'*'*50}\nThe Number of Nulls Values in Service Type Column : ' , end='')
print(data['Service Type'].isnull().sum() , end=f'\n{'~'*50}\n')

data.head()

     
data['Time of Day'] = data['Time of Day'].map({'Morning' : 0 , 'Afternoon' : 1  , 'Evening' : 2  , 'Night' : 3  })
data['Customer Type'] = data['Customer Type'].map({'New' : 0  , 'Returning' : 1})
data = pd.get_dummies(data= data , columns=['Platform'] , prefix='Platform', prefix_sep='-')
data = pd.get_dummies(data= data , columns=['Category'] , prefix='Category', prefix_sep='-')
data = pd.get_dummies(data= data , columns=['Service Type'] , prefix='ServiceType', prefix_sep='-')

     
print(data['Last_treatment_Period_In_Years'].unique())
print(data['Date_Of_Day'].unique())
print(data['Month_Number'].unique())
print(data['Year_Number'].unique())
print(data['DayOfWeek_number'].unique())
print(data['Revenue Change'].unique())
print(data['Ad_to_Revenue_Ratio'].unique())

     
for column in data.select_dtypes(include=bool).columns :
    data[column] =  data[column].map({True : 1  , False :0})

     
data.info()

     
#     Data Searching For Outlyers

     
q1 = data['Time of Day'].quantile(0.25)
q3 = data['Time of Day'].quantile(0.75)
IQR = q3 - q1
low = q1 - 1.5 * IQR
high = q3 + 1.5 * IQR
outlayers = data[(data['Time of Day'] < low) | (data['Time of Day'] > high)]
print(f'The percentage of the outlayers for Time of Day is {(outlayers.shape[0] / data.shape[0])*100} %')

     
data.describe()

     
outs = {}
for column in data.drop(columns=['Daily Revenue']).columns: # Exclude the target variable for clarity in outlier check
    # looking for outlayers
    q1 = data[str(column)].quantile(0.25)
    q3 = data[str(column)].quantile(0.75)
    IQR = q3 - q1
    low = q1 - 1.5 * IQR
    high = q3 + 1.5 * IQR
    outlayers = data[(data[str(column)] < low) | (data[str(column)] > high)]
    per = round((outlayers.shape[0] / data.shape[0])*100, 2)
    if per > 21 :
        outs[str(column)] = per
        # Applying condition to keep values within bounds. You used a broken condition.
        # It's generally better to clip or use median imputation, but based on your original logic:
        # data.loc[(data[str(column)] < low) | (data[str(column)] > high), str(column)] = np.nan
        # data[str(column)].fillna(data[str(column)].median(), inplace=True)
        # Since you mentioned outlayers is small, we will skip the deletion part for simplicity.

for item in outs :
    print(f'{item} -> {outs[item]}')



     
data.info()

     
#     Data Correlation -> Showing Relationship between Features and Target`***

     
# see the realtionship between the target 'Daily Revenue' and all columns
plt.figure(figsize = (25 , 25))
sns.heatmap(data = data.corr() , annot = True , cmap = 'coolwarm'  , linewidth = 0.5 )
plt.title('RelationShip Between the Target \' Daily Revenue \' and the other Columns')
plt.savefig('Correlation_Between_The target_and_the_other_columns.png' , dpi = 300)
plt.show()
     
data.describe()
    
data.info()
     
data.columns = data.columns.str.strip()

print(data.isnull().sum())

     
#   Data Dividing To Train and Test

x  = data.drop(columns = ['Daily Revenue'])
y = data['Daily Revenue']
     
poly = PolynomialFeatures(degree=2)
x = poly.fit_transform(x)

x_train , x_test  , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state = 42 )
scaler = StandardScaler() 

x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test) # Use transform, not fit_transform on test data

     
# Corrected n_estimator to n_estimators
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression(
        n_jobs = 20 
    ))
])

pipeline.fit(x_train , y_train)
y_train_predict = pipeline.predict(x_train)
y_test_predict  = pipeline.predict(x_test)

train_accuracy =  r2_score(y_true = y_train , y_pred = y_train_predict)
test_accuracy =  r2_score(y_true = y_test , y_pred = y_test_predict)

print(f'r2-Score for Training Model is {train_accuracy*100} %')
print(f'r2-Score for Testing Model is {test_accuracy * 100} %' , end=f'\n{'*'*50}\n')

print(f'Mean Absolute Error is {mean_absolute_error(y_test , y_test_predict)}')
print(f' RMSE  : {np.sqrt(mean_squared_error(y_test , y_test_predict))}')

     
plt.scatter(y_test ,  y_test_predict  , color = 'blue'  )
plt.xlabel(' True Values ')
plt.ylabel(' Predicted Values ')
plt.title(' Polynomianl Regression Results ')
plt.show()

     
# Corrected n_estimator to n_estimators
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(
        n_neighbors= 5 , weights='distance' , metric='minkowski'
    ))
])

pipeline.fit(x_train , y_train)
y_train_predict = pipeline.predict(x_train)
y_test_predict  = pipeline.predict(x_test)

train_accuracy =  r2_score(y_true = y_train , y_pred = y_train_predict)
test_accuracy =  r2_score(y_true = y_test , y_pred = y_test_predict)

print(f'r2-Score for Training Model is {train_accuracy*100} %')
print(f'r2-Score for Testing Model is {test_accuracy * 100} %' , end=f'\n{'*'*50}\n')

print(f'Mean Absolute Error is {mean_absolute_error(y_test , y_test_predict)}')
print(f' RMSE  : {np.sqrt(mean_squared_error(y_test , y_test_predict))}')


     
# Corrected n_estimator to n_estimators
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=300 , max_depth=10 , random_state=42))
])

pipeline.fit(x_train , y_train)
y_train_predict = pipeline.predict(x_train)
y_test_predict  = pipeline.predict(x_test)

train_accuracy =  r2_score(y_true = y_train , y_pred = y_train_predict)
test_accuracy =  r2_score(y_true = y_test , y_pred = y_test_predict)

print(f'r2-Score for Training Model is {train_accuracy*100} %')
print(f'r2-Score for Testing Model is {test_accuracy * 100} %' , end=f'\n{'*'*50}\n')

print(f'Mean Absolute Error is {mean_absolute_error(y_test , y_test_predict)}')
print(f' RMSE  : {np.sqrt(mean_squared_error(y_test , y_test_predict))}')


     
plt.scatter(y_test ,  y_test_predict  , color = 'blue'  )
plt.xlabel(' True Values ')
plt.ylabel(' Predicted Values ')
plt.title(' Polynomianl Regression Results ')
plt.show()

     
# Corrected n_estimator to n_estimators
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

pipeline.fit(x_train , y_train)
y_train_predict = pipeline.predict(x_train)
y_test_predict  = pipeline.predict(x_test)

train_accuracy =  r2_score(y_true = y_train , y_pred = y_train_predict)
test_accuracy =  r2_score(y_true = y_test , y_pred = y_test_predict)


print(f'r2-Score for Training Model is {train_accuracy*100} %')
print(f'r2-Score for Testing Model is {test_accuracy * 100} %' , end=f'\n{'*'*50}\n')

print(f'Mean Absolute Error is {mean_absolute_error(y_test , y_test_predict)}')
print(f' RMSE  : {np.sqrt(mean_squared_error(y_test , y_test_predict))}')


     
#  Then The used Model Based On The Resulta is XGBRegressor

     
plt.scatter(y_test ,  y_test_predict  , color = 'blue'  )
plt.xlabel(' True Values ')
plt.ylabel(' Predicted Values ')
plt.title(' Polynomianl Regression Results ')
plt.show()

     
#  Here We Will use Grid Search For Model Of XGBRegressor

param_grid = {
    'n_estimators':randint(100 , 500),
    'max_depth': randint(3  , 10),
    'learning_rate': uniform(0.01 ,  0.2),
    'model__subsample': uniform(0.6 , 0.4),
    'model__colsample_bytree': uniform(0.6 , 0.4)
}

random_xgb = RandomizedSearchCV(
    estimator = XGBRegressor(objective= 'reg:squarederror' , random_state=42),
    param_distributions=param_grid,
    n_iter= 50 , 
    cv=5,                # 5-Fold Cross Validation
    scoring='r2',        # Regression Score
    n_jobs=-1,           
    verbose=1 , 
    random_state= 42
)

random_xgb.fit(x_train , y_train)

print(f'Best Score is {random_xgb.best_score_}')

print(f'Best Parameters is {random_xgb.best_params_}')

y_pred_xgb_train = random_xgb.predict(x_train)  
y_pred_xgb_test = random_xgb.predict(x_test)


print('Results of XGB Regresor After Using Grid Search')

print(f'R2-Score (Accuarcy) for train : {r2_score(y_train , y_pred_xgb_train)*100} %')
print(f'R2-Score (Accuarcy) for test : {r2_score(y_test , y_pred_xgb_test)*100} %')
print(f' MAE : {mean_absolute_error(y_test , y_pred_xgb_test)}')
print(f' RMSE  : {np.sqrt(mean_squared_error(y_test , y_pred_xgb_test))}')


     
#     NEW STEP: Save the list of raw features for order enforcement in Streamlit <br> Save Model and Scalaer to be used in Building UI in Streamlit

     
joblib.dump(pipeline ,"model_project.pkl")

joblib.dump(scaler , 'scaler_project.pkl')

joblib.dump(poly, 'poly_transformer.pkl')

joblib.dump(data.drop(columns = ['Daily Revenue']).columns, 'features_project.pkl')  # here we will save the features names as only the name of columns 

data.info()