# qml

This repo contains machine learning projects and web apps for various use-cases. The experimentation is done on Jupyter notebook and the models are deployed online using Dash Plotly. 
Dash is a python based open-source and also an enterprise platform for development and deployment of machine learning and data analytics solutions <a href="https://plotly.com/dash/" target="blank"> more on Dash</a>. 

## Use-Cases

### <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">1. Telco Customer Churn</a>
The first project in this repo is based on the <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">Telco Customer Churn </a>use-case. In this use-case we use machine learning to predict wether a an existing customer will churn or not.
We begin by collecting and cleaning the data. We then explore the data to find the insights and the distribution of the data. We preprocess and perform feature engineering of the data.
Once we have the correct features we proceed to machine learning modeling where we split our data into training, validation and testing sets. We train various models and compare there performance. <br>

We perform hyperparameter tunning using Grid Search approach to optimize our model and find the optimal hyperparameters for each algorithm. After training the model with best hyerparameters we evaluate the models 
using various model evaluation techniques (Accuracy, Precision, Recall, F1-Score and UAC ROC). We save the best performing model based on the evaluation metric and use it for deployment.
In deployment we create a dash web app and deploy our model on heroku server. We create a user interface where the model can be used to predict with single or batch data upload and prediction.
The web app has three web interface; The data Exploration section, The Model training and Feature selection section and The Model Prediction and results presentation section.
The project is hosted here <a href="https://qaml.herokuapp.com/apps/telco_customer_churn" target="blank">Telco Customer Churn ML Project</a>
![Telco Customer Churn Data Exploration](https://raw.githubusercontent.com/SammyOngaya/Customer-Churn-Prediction/master/Notebooks/Churn%20Models/Telco%20Customer%20Churn%20Data%20Exploration.PNG)



### <a href="https://qaml.herokuapp.com/apps/customer_lifetime_value" target="blank">2. Customer Lifetime Value</a>

Customer lifetime value is the total worth to a business of a customer over the whole period of their relationship. It’s an important metric as it costs less to keep existing customers than it does to acquire new ones, so increasing the value of your existing customers is a great way to drive growth. Knowing the CLV helps businesses develop strategies to acquire new customers and retain existing ones while maintaining profit margins. <br><br>
CLV=Expected No. of Transaction * Revenue per Transaction * Margin <br>
Where;<br>
Expected No. of Transaction is calculated using BG/NBD Model<br>
Revenue per Transaction is calculated using Gama Gama Model and<br> 
Margin is provided by historical transaction or we can take a standard value of 5%.<br><br>
Model<br>
For this use-case we will use the lifetimes library. 
Lifetimes is used to analyze your users based on a few assumption:<br>

1. Users interact with you when they are "alive".<br>
2. Users under study may "die" after some period of time.

<a href="https://lifetimes.readthedocs.io/en/master/index.html"> more on library</a>
<br><br>
Datasets<br>
We will use the online retail  transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers <a href="http://archive.ics.uci.edu/ml/datasets/online+retail">more on dataset</a>.
The use-case is hosted here <a href="https://qaml.herokuapp.com/apps/customer_lifetime_value" target="blank"> Customer Lifetime Value</a>

![Customer Lifetime Value](https://raw.githubusercontent.com/SammyOngaya/qml/main/qml/assets/customer_lifetime_value/customer-lifetime-value.PNG)


