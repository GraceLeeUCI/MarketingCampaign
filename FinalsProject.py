import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(layout="centered")

st.title("Finding Correlation(s) in the Marketing Campaign Data")
st.write("GitHub repository : https://github.com/GraceLeeUCI?tab=repositories")

st.subheader("Introduction to the Data Set")
'''
Data Source: https://www.kaggle.com/imakash3011/customer-personality-analysis

In this app, it dives into the marketing campaign data gathered from a company,
and tries to understand perhaps the possible target audience that the company should
aim to advertise to. 

By comparing the age, marital status, and education to total spendings, we will
be able to obtain a specific target audience, in which the company should try to focus on. 
'''

'''
Attributes:

    ID: Customer's unique identifier 
    Year_Birth: Customer's birth year 
    Education: Customer's education level 
    Marital_Status: Customer's marital status 
    Income: Customer's yearly household income 
    Kidhome: Number of children in customer's household 
    Teenhome: Number of teenagers in customer's household 
    Dt_Customer: Date of customer's enrollment with the company 
    Recency: Number of days since customer's last purchase 
    Complain: 1 if customer complained in the last 2 years, 0 otherwise Products

    MntWines: Amount spent on wine in last 2 years 
    MntFruits: Amount spent on fruits in last 2 years 
    MntMeatProducts: Amount spent on meat in last 2 years 
    MntFishProducts: Amount spent on fish in last 2 years 
    MntSweetProducts: Amount spent on sweets in last 2 years 
    MntGoldProds: Amount spent on gold in last 2 years Promotion

    NumDealsPurchases: Number of purchases made with a discount 
    AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise 
    AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise 
    AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise 
    AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise 
    AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise 
    Response: 1 if customer accepted the offer in the last campaign, 0 otherwise Place

    NumWebPurchases: Number of purchases made through the company’s web site 
    NumCatalogPurchases: Number of purchases made using a catalogue 
    NumStorePurchases: Number of purchases made directly in stores 
    NumWebVisitsMonth: Number of visits to company’s web site in the last month
'''
st.write("#")

st.write("Marketing Campaign Data Table")
#reading the data
 
df = pd.read_csv("marketing_campaign.csv",sep='\t', na_values = " ")

if st.button("Observe Original Data Table"):
    df = pd.read_csv("marketing_campaign.csv",sep='\t', na_values = " ")
    st.dataframe(df)

st.write("#")

#Data Cleaning
st.subheader("**Data Cleaning**")

st.write("Data cleaning allows us to take a look at just the information that we will be needing, and taking out things that is irrelevant. By taking out the column(s) and row(s) that we will not be using and are irrelevant to our data, we will be able to have a clearner table and dataset to look and work with. ")
st.write("First lets take a look at the columns and description of the data and see if we can simplify anything or remove anything")
st.caption("Remember we also need to get rid of rows that have na values, ")

#Pulling out the list of columns, to see what we are working with
columns = list(df.columns)
st.write("Columns",columns)

'''
In the columns we can see that there are several columns with the acceptedcmp,
There were a total of 5 campaigns, but we are just going to combine all by summing
all 5 campaigns together. Meaning 5 would be the highest and 0 would be the lowest values.

We are going to the same for amount of spendings on each product.
By summing all the amounts of products (winess, fruits, fish, meat, sweets, and gold), 
we will be able to see the total spendings of each customer.
And then drop the individual columns from the data table that were summed up.

Also we are going to sum of the number of purcahses from deals, web, catalog, store.

In addition, lets change the Year_Birth to the age of the customers, so that we can
have more simple numbers, and ages are more reletively easier to deal with.

'''
#Adding up spendings on Products
df['TotalSpending'] = df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis = 1, skipna = True)

#Adding up Accepted Campaigns 1~5
df['AcceptedCmps1~5'] = df[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2']].sum(axis = 1, skipna = True)

#Adding up Number of Purchases from deals, Web, Catalog, and Store
df['TotNumPurchases'] = df[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'
       ]].sum(axis = 1, skipna = True)


#Converting Birth_Year to represent Age
df['Age'] = pd.to_datetime(df['Dt_Customer']).max().year - df['Year_Birth']


#Finding NA values, Description of the dataframe
col1, col2 = st.columns(2)
    
with col1:
    st.write("Are there any NA values? True = No NA values; False = Has NA values", df.notna().all(axis=1))
    
with col2:
    st.write("Let's take a look at the data's description", df.describe())


#Explanation
'''
We can see that there was a column in which had a standard deviation of 0. 
That means there is only one value variety. Therfore, it will not be much of 
use in our data since all customers provided the same answer.

We can also see that income was the only column that had contained NA values.
However, we won't be using the income data, so it will de dropped.

Now lets drop the columns that we will not be needing which are:
    ID, Income, Kidhome, Teenhome, Response, Dt_Customer, AcceptedCmp1,2,3,4,5, 
    Birth_Year, Z_CostContact, Z_Revenue, NumWebVisitsMonth, complain. 
    
Lets keep some of the other columns in case, we might need them to further test our data.
'''

#creating clean dataframe
df_clean = df.drop(['ID', 'Recency', 'Income', 'Kidhome', 'Teenhome', 'Response', 'Dt_Customer', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2','Year_Birth', 'Z_CostContact', 'Z_Revenue', 'NumWebVisitsMonth', 'Complain','NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
       'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'], axis=1)
st.dataframe(df_clean)

st.write("#")
#Creating a dummies df, 
'''
This is an extra step where we are seperating all of the different categories in  Education and Marital Status into their own column
This allows us to see the different cateogries within each column in a broader view!
'''
df_dum = pd.get_dummies(df_clean, 
                       prefix=['Education', 'Marital_Status'], 
                       columns = ['Education', 'Marital_Status'],
                       drop_first=True)
st.write("Dummified Data set. In this dataset you will be able to see the separation of the values in Marital_Status and Education into their own columns! This will help us later to get a more accurate and detailed models.", df_dum)

#spacer
st.write("##")

#EDA
st.header("EDA - Exploratory Data Analysis")

'''
Now onto Data Analysis! 
'''
st.write("##")


st.subheader("Drawing a Logistic Regression")
'''
We are going to build a Logisitic Regression to help predict test set results and calculating the accuracy.
We are going to train and test the dummified data set, but we are just going to test 20% of the data set for evaluating 
the classifer. Definition of what is being tested is written for better understanding of the logistic regression.
Logistic regression helps to predict binary classes which is great for our type of dataset! This is a predictive
algorithm that uses independent variables to predict the dependent variable, wehre the dependent variables are 
categorical.

Cross-validated scores: There is a risk of overfitting on the test set which is not good. So in order to fix that
the cross-validation training is done on the traning set, after evulation is done on the validation set, and when 
the experiment seems to be successful and not overfitting, the final evaluation is done on the test set.

Average score: Gives the mean of the cross-validated score

Training score: Accuracy of the prediction
    
Test score: Probaility of each column based on the AcceptedCmps 1~5. 
'''
  # diving data into trainning and testing
X = df_dum.copy()
y = X.pop('AcceptedCmps1~5')

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2, 
    random_state=10)
    
    # standardlize the train/test sets of data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train),
                       columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), 
                      columns=X_train.columns)

if st.button("Classify", key="classify"):
    st.text("Logistic Regression data")
    st.write(("X_train shape, y_train shape"),(X_train.shape, y_train.shape))
    st.write(("X_train shape, y_train shape"), (X_test.shape, y_test.shape))
    
    # Classfier Model1 - Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    cv = cross_val_score(logreg, X_train, y_train, cv=5)
    st.write('----------------------------------------------------------------------------------------------')
    st.write('Cross-validated scores:', cv)
    st.write('Average score:', cv.mean())
    st.write('Trainning Score:', logreg.score(X_train, y_train))
    st.write('Test Score:', logreg.score(X_test, y_test))
    
    
    # Classfier Model2 - RF Classifier
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)
    cv_rfc = cross_val_score(rfc, X_train, y_train, cv=5)
    st.write('----------------------------------------------------------------------------------------------')
    st.write("Using Random Forest Classifier to get a further look into our data. How Random Forest works is each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model’s prediction. We are able to see that the Random Forest Classifier had better prediction percentage compared to the first model of logistic regression used. While the fist model was almost overfitted, we can see that here in the Rnadmon Forest it a much more safer prediction!")
    st.write('Cross-validated scores:', cv_rfc)
    st.write('Average score:', cv_rfc.mean())
    st.write('Trainning Score:', rfc.score(X_train, y_train))
    st.write('Test Score:', rfc.score(X_test, y_test))



'''
We can conclude from the logistic regression models that the Random Forest Classifier
provides a more reliable prediction on the data. Overall, none of the models provided
an overfitting result.

Probability: ~%80
'''



st.write("##")


#Creating a visual representation
st.subheader("Visual Representations of the Data")

'''
Now we are going to create some visual representation to help predict and
make more possible suggesstions about the dataset.

This bar chart below shows how many people had accepted how many campaigns.
For instance, we can see that 1777 customers had not accepted any campaign.
And only 11 people had accepted 4 of the campaigns.
This bar charts just help to gain a better understanding of how many people are
actually accpeting campaigns.

We can see that not many people accept campaigns at all. Perhaps the company
needs a change in their campaign to further fit the customers taste.
'''
st.bar_chart(df_clean['AcceptedCmps1~5'].value_counts())

st.write("#")
'''
Below there are two graphs, a scatter plot and a histogram

The scatter plot the Age and Total Spending. The colors are organized by their marital status.
We can see that the ages range from 20-60, where the largest spenders were in the early 20's. 
Hovering over the data points, will allow you to see the marital status and education of the 
point on the graph.
'''

my_graph = alt.Chart(df_clean).mark_circle().encode(
    x = 'Age',
    y = 'TotalSpending',
    tooltip=['Marital_Status','Education'],
    color = "Marital_Status"
    )
st.write(my_graph)

st.write("#")

'''
Here is another graph, but a histgram graph. In this histogram we are looking at the correlation
for age and total accepted campaigns. We can see that from the age of 20-80
most pepole had accepted a total of 4 campaigns. 
'''

my_graph2 = alt.Chart(df_clean).mark_bar().encode(
    alt.X("Age", bin=True),
    y='AcceptedCmps1~5',
    )
    
st.altair_chart(my_graph2)


'''
Comparing all graphs/charts, we can conclude that the company should aim to 
advertise towards the people of ages 20-70, but rather more specifically,
those in early 20's, 40's and 60's seem to spend the most too.

We can also seem to make a conclusion that those that had accepted 4 campaigns,
have a higher total spending!
'''


st.subheader("Conclusion")

'''
We can conclude that the company should place their target audience of the 
ages 20-50 and possible change their campaigns so that it is more suited
to that area of age!
'''

