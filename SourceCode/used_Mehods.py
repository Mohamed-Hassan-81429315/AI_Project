from datetime import date
import pandas as pd

# convert  the date values to period as we will compute the period that  the last coefficiently
# as this helps us to detect which causes the clients to reduce treating with the company
# which helps us to make strategies to increase the number of clients , and increase the 
# duration of their treatment with the company as this increases the revenues

def Date_Calculation(date_sent : date) :   # used to compute the date from certain date dedicated by user untill this day
  years = (date.today().year - date_sent.year)
  months = (date.today().month - date_sent.month)
  days = (date.today().day - date_sent.day)
  years = years if years > 0 else 0
  remainder = 0
  if months >= 0 :
        months =  months
        remainder = months / 12
  else :
        years -=1
        months = date.today().month
        remainder = months / 12

  return round((years + remainder) , 2) # used to calcualte the number of years like 4.53 years

@property
def resources() :
    data = pd.read_csv('dataset_file.csv' , sep=',')
    global values 
    values = data["Daily Revenue"]
    return values

def re_use_resources():
    data = pd.read_csv('dataset_file.csv' , sep=',')
    return data["Daily Revenue"]

def add_value(value :float) :
     values.add(value)

def last_value_added() :  
     return float(values.values[-1][-1])