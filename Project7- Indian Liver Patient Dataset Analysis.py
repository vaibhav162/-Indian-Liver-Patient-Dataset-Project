#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries & Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


liver_df= pd.read_csv(r"C:\Users\shruti\Desktop\indian_liver_patient.csv")


# In[3]:


liver_df.head(2)


# In[4]:


liver_df.tail(2)


# # Exploratory Data Analysis (EDA)

# In[5]:


liver_df.info()


# In[6]:


liver_df.describe()


# In[7]:


# Checking for Null-Values

liver_df.isnull().sum()


# In[8]:


liver_df.isna().sum()


# # Data Visualization

# In[9]:


# Plotting Number of patient with Liver disease Vs Number of pateint with no Liver disease

sns.countplot(data=liver_df, x="Dataset", label= "Count")

LD, NLD= liver_df["Dataset"].value_counts()
print("Number of patient diagnosed with Liver disease: ", LD)
print("Number of patient not diagnosed with Liver disease: ", NLD)


# In[10]:


# Plotting Number of Male & Female patient

sns.countplot(data=liver_df, x="Gender", label= "Count")

M, F= liver_df["Gender"].value_counts()
print("Number of Male patient: ", M)
print("Number of Female patient: ", F)


# In[11]:


# Plotting Patient over Age Vs Gender
# In Dataset, 1 implies the patient have Liver disease; 2 implies the patients do not have Liver disease

sns.barplot(data=liver_df, x="Age", y="Gender", hue="Dataset")


# In[12]:


liver_df[["Gender", "Age", "Dataset"]].groupby(["Dataset", "Gender"], 
                                               as_index= False).mean().sort_values(by="Dataset", ascending= False)


# In[13]:


# Plotting Age Vs Gender

g= sns.FacetGrid(liver_df, col="Dataset", row= "Gender", margin_titles=True)
g.map(plt.hist, "Age", color="lightgreen")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Disease by Gender and Age")


# In[14]:


# Plotting Gender(Male/Female) along with Total_Bilirubin and Direct_Bilirubin

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Direct_Bilirubin", "Total_Bilirubin", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[15]:


# Plotting Total_Bilirubin Vs Direct_Bilirubin

sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data= liver_df, kind="reg")


# In[16]:


# Plotting Gender(Male/Female) along with Alamine_Aminotransferase and Aspartate_Aminotransferase

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Aspartate_Aminotransferase", "Alamine_Aminotransferase", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[17]:


# Plotting Aspartate_Aminotransferase Vs Alamine_Aminotransferase

sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data= liver_df, kind="reg")


# In[18]:


# Plotting Gender(Male/Female) along with Alkaline_Phosphotase and Alamine_Aminotransferase

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Alkaline_Phosphotase", "Alamine_Aminotransferase", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[19]:


# Plotting Alkaline_Phosphotase Vs Alamine_Aminotransferase

sns.jointplot("Alkaline_Phosphotase", "Alamine_Aminotransferase", data= liver_df, kind="reg")


# In[20]:


# Plotting Gender(Male/Female) along with Total_Protiens and Albumin

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Total_Protiens", "Albumin", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[21]:


# Plotting Total_Protiens Vs Albumin

sns.jointplot("Total_Protiens", "Albumin", data= liver_df, kind="reg")


# In[22]:


# Plotting Gender(Male/Female) along with Albumin and Albumin_and_Globulin_Ratio

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Albumin", "Albumin_and_Globulin_Ratio", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[24]:


# Plotting Albumin Vs Albumin_and_Globulin_Ratio

sns.jointplot("Albumin", "Albumin_and_Globulin_Ratio", data= liver_df, kind="reg")


# In[25]:


# Plotting Gender(Male/Female) along with Albumin_and_Globulin_Ratio and Total_Protiens

g= sns.FacetGrid(liver_df, col="Gender", row= "Dataset", margin_titles=True)
g.map(plt.scatter, "Albumin_and_Globulin_Ratio", "Total_Protiens", color="lightgreen", edgecolor="blue")
plt.subplots_adjust(top=0.9)


# In[26]:


# Plotting Albumin_and_Globulin_Ratio Vs Total_Protiens

sns.jointplot("Albumin_and_Globulin_Ratio", "Total_Protiens", data= liver_df, kind="reg")


# # Feature Engineering

# In[27]:


liver_df.head(3)


# In[28]:


pd.get_dummies(liver_df["Gender"], prefix="Gender").head()


# In[31]:


# Concatenating

liver_df= pd.concat([liver_df,pd.get_dummies(liver_df["Gender"], prefix="Gender")], axis=1)


# In[37]:


liver_df.head(3)


# In[33]:


liver_df.describe()


# In[34]:


liver_df[liver_df["Albumin_and_Globulin_Ratio"].isnull()]


# In[35]:


liver_df["Albumin_and_Globulin_Ratio"]= liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df["Albumin_and_Globulin_Ratio"].mean())


# In[ ]:


X=liver_df.drop(["Gender", "Dataset"], axis=1)
X.head(2)


# In[40]:


y= liver_df["Dataset"]


# # Correlation between all features

# In[43]:


liver_corr= X.corr()
liver_corr


# In[46]:


# Plotting Heatmaps for correlation between all features

plt.figure(figsize=(12,8))
sns.heatmap(liver_corr, cbar=True, square=True, annot=True, fmt=".2f", annot_kws={"size":12}, cmap="coolwarm")
plt.title("Correlation between all the features")


# # Splitting data into Train & Test

# In[50]:


X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# # Model Building

# ### 1. Logistic Regression

# In[58]:


logreg= LogisticRegression()

# Training model using training sets and check score
logreg.fit(X_train, y_train)

# Predict Output
log_predicted= logreg.predict(X_test)

logreg_score= round(logreg.score(X_train, y_train)*100, 2)
logreg_score_test= round(logreg.score(X_test, y_test)*100, 2)

# Equation coefficient & Intercept
print("Logistic Regression Training Score: \n", logreg_score)
print("Logistic Regression Training Score: \n", logreg_score_test)

print("Accuracy: \n", accuracy_score(y_test, log_predicted))
print("Confusion Matrix: \n", confusion_matrix(y_test, log_predicted))
print("Classification Report: \n", classification_report(y_test, log_predicted))


# #### Confusion Matrix

# In[59]:


sns.heatmap(confusion_matrix(y_test, log_predicted), annot=True, fmt="d")


# ### 2. Gaussian Naive Bayes

# In[60]:


gaussian= GaussianNB()
gaussian.fit(X_train, y_train)

gauss_predicted= gaussian.predict(X_test)

gauss_score= round(gaussian.score(X_train, y_train)*100,2)
gauss_score_test= round(gaussian.score(X_test, y_test)*100,2)

print("Gaussian Score: \n", gauss_score)
print("Gaussian Score: \n", gauss_score_test)

print("Accuracy: \n", accuracy_score(y_test, gauss_predicted))
print("Confusion Matrix: \n", confusion_matrix(y_test, gauss_predicted))
print("Classification Report: \n", classification_report(y_test, gauss_predicted))


# #### Confusion Matrix

# In[61]:


sns.heatmap(confusion_matrix(y_test, gauss_predicted), annot=True, fmt="d")


# ### 3. Random Forest

# In[62]:


random_forest= RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

rf_predicted= gaussian.predict(X_test)

random_forest_score= round(random_forest.score(X_train, y_train)*100,2)
random_forest_score_test= round(random_forest.score(X_test, y_test)*100,2)

print("Random Forest Score: \n", random_forest_score)
print("Random Forest Score: \n", random_forest_score_test)

print("Accuracy: \n", accuracy_score(y_test, rf_predicted))
print("Confusion Matrix: \n", confusion_matrix(y_test, rf_predicted))
print("Classification Report: \n", classification_report(y_test, rf_predicted))


# #### Confusion Matrix

# In[63]:


sns.heatmap(confusion_matrix(y_test, rf_predicted), annot=True, fmt="d")


# # Model Evaluation

# In[65]:


# Comparing all Models

models= pd.DataFrame({
    "Model": ["Logistic Regression", "Gaussian Naive Bayes", "Ramdom Forest"],
    "Score": [logreg_score, gauss_score, random_forest_score],
    "Test Score": [logreg_score_test, gauss_score_test, random_forest_score_test]})
models.sort_values(by="Test Score", ascending=False)


# # Conclusion

# ### Among all the Models, Logistic Regression Model perform the best Model Building on this Dataset
