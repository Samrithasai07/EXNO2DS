## NAME:SAMRITHA R
## REG NO:212224040286

# EXNO2-DS
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dt = pd.read_csv("/content/titanic_dataset.csv")
dt
```

<img width="1411" height="465" alt="image" src="https://github.com/user-attachments/assets/b8609ca2-bb3c-4f6f-a7bf-4fcb8538c5f6" />

```
dt.info()
```

<img width="488" height="380" alt="image" src="https://github.com/user-attachments/assets/5e01e7a4-d3be-45d1-bb23-b16d4792cfa3" />

```
dt.shape
```

<img width="223" height="37" alt="image" src="https://github.com/user-attachments/assets/adee6962-5780-4f5c-a256-8d3cd7a96a98" />

```
dt.set_index("PassengerId",inplace=True)
dt
```
<img width="1400" height="508" alt="image" src="https://github.com/user-attachments/assets/ee7b0103-f577-48e6-ab32-823c0ca29c55" />

```
print("Data types:")
display(dt.dtypes.to_frame("Data Type"))
```

<img width="377" height="447" alt="image" src="https://github.com/user-attachments/assets/06c727fc-2f04-40e2-93f3-4a3548c3a831" />


```
print("Missing values:")
missing=pd.DataFrame({
    "Missing count":dt.isnull().sum(),
    "Missing percentage":(dt.isna().mean()*100).round(2)
})
display(missing.sort_values("Missing percentage",ascending=False))
```

<img width="538" height="457" alt="image" src="https://github.com/user-attachments/assets/5c5bcdcd-d968-460f-987c-aacbafa6a449" />

```
print("Exact duplicate rows:",dt.duplicated().sum())
print("Numeric summary")
display(dt.describe())
```

<img width="730" height="368" alt="image" src="https://github.com/user-attachments/assets/67b256af-b428-4bd5-a9fe-94eb8763d0df" />

```
dt.nunique()
```

<img width="338" height="475" alt="image" src="https://github.com/user-attachments/assets/50c4b7c7-8834-4c31-b3db-42ac33273d3b" />

```
dt["Survived"].value_counts()
```

<img width="267" height="201" alt="image" src="https://github.com/user-attachments/assets/c89e1e9f-93fb-460d-a1c4-dca15ee29488" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
<img width="241" height="190" alt="image" src="https://github.com/user-attachments/assets/5f3bcb87-5af2-4a31-81e1-14d5a3c0266b" />

```
sns.countplot(data=dt, x="Survived")
```

<img width="693" height="516" alt="image" src="https://github.com/user-attachments/assets/745e3a79-b119-4a26-a276-5f9926a1c77c" />

```
dt.Pclass.unique()
```
<img width="196" height="28" alt="image" src="https://github.com/user-attachments/assets/84561acb-40e6-4cbb-93c4-daa3afce6972" />

```
dt.rename(columns={'Sex':'Gender'},inplace=True) 
dt
```

<img width="1455" height="507" alt="image" src="https://github.com/user-attachments/assets/3cba7eb3-e53c-430b-8984-c99852b7a37c" />

```
sns.catplot(x='Gender',col='Survived',kind='count',data=dt,height=5,aspect=.7)
```

<img width="912" height="583" alt="image" src="https://github.com/user-attachments/assets/a76b37ed-a7c0-4adb-ae66-5afb81def961" />

```
sns.catplot(x='Survived', hue="Gender", data=dt, kind = "count")
```

<img width="712" height="573" alt="image" src="https://github.com/user-attachments/assets/3e2b618e-0dcb-4bc1-bb38-f0d1d0ee2ed2" />

```
dt.boxplot(column='Age',by='Survived')
```

<img width="777" height="542" alt="image" src="https://github.com/user-attachments/assets/e46a273a-72ff-4ecc-8a47-94ac76693876" />


```
sns.scatterplot(x=dt['Age'],y=dt['Fare'])
```

<img width="795" height="515" alt="image" src="https://github.com/user-attachments/assets/35cf9886-fa59-47ca-b5ac-354c92a9d27e" />

```
sns.jointplot(x="Age",y="Fare", data=dt)
```

<img width="901" height="692" alt="image" src="https://github.com/user-attachments/assets/b257d6fd-ebd3-4f0b-8828-adda91fe6312" />

```
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x="Pclass",y='Age', hue='Gender', data=dt)
```

<img width="930" height="515" alt="image" src="https://github.com/user-attachments/assets/b7b82f77-69b1-49c1-b0a6-a57276e0a296" />

```
sns.catplot(data=dt, col ="Survived", x = "Gender", hue="Pclass", kind = "count")
```

<img width="1295" height="582" alt="image" src="https://github.com/user-attachments/assets/b6243e66-631e-4ce9-a2f0-e79015337162" />


```
corr=dt.corr(numeric_only=True)

sns.heatmap(corr,annot=True,cmap='coolwarm')
```

<img width="675" height="496" alt="image" src="https://github.com/user-attachments/assets/3872d7e1-d88f-4435-8be4-b5aa6bd0bedc" />


```
sns.pairplot(dt)
```
<img width="782" height="763" alt="image" src="https://github.com/user-attachments/assets/3befec85-31b1-48a6-9469-1e56034b8ef7" />

# SUMMARY

### **SUMMARY**

In this experiment, I performed Exploratory Data Analysis (EDA) on the Titanic dataset using Python. First, I imported Pandas, NumPy, Seaborn and Matplotlib libraries and loaded the dataset. I checked the data types, missing values, duplicate records and statistical information. I used countplot, catplot and boxplot to analyze passenger survival, age, gender and passenger class. I used scatterplot and jointplot to understand the relationship between age and fare. Finally, I used heatmap and pairplot to identify correlations and visualize relationships between variables. Thus, I understood the dataset and identified useful patterns through different visualization techniques.


# RESULT
The Program and The Output has been sucessfully Verified along with the Visualization.
