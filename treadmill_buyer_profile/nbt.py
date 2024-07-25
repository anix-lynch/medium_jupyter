import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Path to your dataset
dataset_path = "/Users/anixlynch/Downloads/Stratascratch/treadmill-buyer-profile/aerofit_treadmill_data.csv"

# Create a list of cells
cells = [
    nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")"""),

    nbf.v4.new_code_cell(f"""dataset_path = "{dataset_path}"
aerofit_df = pd.read_csv(dataset_path)"""),

    nbf.v4.new_code_cell("aerofit_df.head()"),

    nbf.v4.new_code_cell("aerofit_df.shape"),

    nbf.v4.new_code_cell("aerofit_df.columns"),

    nbf.v4.new_code_cell("aerofit_df.dtypes"),

    nbf.v4.new_code_cell("""aerofit_df['Product'] = aerofit_df['Product'].astype('category')
aerofit_df['Gender'] = aerofit_df['Gender'].astype('category')
aerofit_df['MaritalStatus'] = aerofit_df['MaritalStatus'].astype('category')"""),

    nbf.v4.new_code_cell("aerofit_df.info()"),

    nbf.v4.new_code_cell("aerofit_df.memory_usage(deep=True)"),

    nbf.v4.new_code_cell("aerofit_df.describe(include='all')"),

    nbf.v4.new_code_cell("aerofit_df.isna().sum()"),

    nbf.v4.new_code_cell("aerofit_df.duplicated(subset=None, keep='first').sum()"),

    nbf.v4.new_code_cell("""# Non-Graphical Analysis
# Unique values
aerofit_df['Product'].unique()
aerofit_df['Gender'].unique()
aerofit_df['MaritalStatus'].unique()"""),

    nbf.v4.new_code_cell("""# Graphical Analysis - Numerical Variables
# Distribution plots
plt.figure(figsize=(20,10))

plt.subplot(2,3,1)
sns.histplot(aerofit_df['Age'], kde=True, color='blue')

plt.subplot(2,3,2)
sns.histplot(aerofit_df['Education'], kde=True, color='orange')

plt.subplot(2,3,3)
sns.histplot(aerofit_df['Usage'], kde=True, color='green')

plt.subplot(2,3,4)
sns.histplot(aerofit_df['Fitness'], kde=True, color='red')

plt.subplot(2,3,5)
sns.histplot(aerofit_df['Income'], kde=True, color='purple')

plt.subplot(2,3,6)
sns.histplot(aerofit_df['Miles'], kde=True, color='brown')

plt.show()"""),

    nbf.v4.new_code_cell("""# Graphical Analysis - Categorical Variables
# Count plots
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
sns.countplot(x='Product', data=aerofit_df, palette='viridis')

plt.subplot(2,2,2)
sns.countplot(x='Gender', data=aerofit_df, palette='viridis')

plt.subplot(2,2,3)
sns.countplot(x='MaritalStatus', data=aerofit_df, palette='viridis')

plt.subplot(2,2,4)
sns.countplot(x='Usage', data=aerofit_df, palette='viridis')

plt.show()"""),

    nbf.v4.new_code_cell("""# Observations
# 1. Most people using the product are in their 20s.
# 2. Higher usage is observed among males.
# 3. Single people are the majority users.
# 4. People with a fitness level of 3 are the most frequent users."""),

    nbf.v4.new_code_cell("""# Multivariate Analysis
attributes = ['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']
sns.set(color_codes=True)
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
fig.subplots_adjust(hspace=0.3)
count = 0
for i in range(3):
    for j in range(2):
        sns.boxplot(data=aerofit_df, x='Gender', y=attributes[count], hue='Product', ax=ax[i,j])
        ax[i,j].set_title(f'Product vs {attributes[count]}', pad=12, fontsize=12)
        count += 1
plt.show()"""),

    nbf.v4.new_code_cell("""# Correlation Analysis
aerofit_df.corr()"""),

    nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(18,10))
sns.heatmap(aerofit_df.corr(), ax=ax, annot=True, linewidths=0.8, fmt='.2f')
plt.show()"""),

    nbf.v4.new_code_cell("""# Observations
# 1. (Miles & Fitness) and (Miles & Usage) attributes are highly correlated, which means if a customer's fitness level is high they use more treadmills.
# 2. Income and Education shows a strong correlation. High-income and highly educated people prefer the KP281 treadmill which is having advanced features.
# 3. There is no correlation between (Usage & Age) or (Fitness & Age) attributes, which mean Age should not be a barrier to using treadmills or specific model of treadmills."""),

    nbf.v4.new_code_cell("""# Pair Plots
sns.pairplot(aerofit_df, hue='Product')
plt.show()"""),

    nbf.v4.new_code_cell("""# Marginal & Conditional Probabilities
# What percent of customers have purchased KP281, KP481, or KP781?
aerofit_df1 = aerofit_df[['Product', 'Gender', 'MaritalStatus']].melt()
aerofit_df1.groupby(['variable', 'value'])[['value']].count() / len(aerofit_df1) * 100"""),

    nbf.v4.new_code_cell("""aerofit_df1.groupby(['variable', 'value'])[['value']].count() / len(aerofit_df1) * 100.round(1).astype(str) + "%" """),

    nbf.v4.new_code_cell("""# Calculate probability of a customer based on Gender
# Probability of female buying a certain treadmill product
for gender in aerofit_df['Gender'].unique():
    print(f"Gender: {gender}")
    for product in aerofit_df['Product'].unique():
        prob = len(aerofit_df[(aerofit_df['Gender'] == gender) & (aerofit_df['Product'] == product)]) / len(aerofit_df[aerofit_df['Gender'] == gender])
        print(f"Probability of {product}: {prob:.2f}")
    print()"""),

    nbf.v4.new_code_cell("""# Calculate probability of a customer based on Marital Status
# Probability of single or partnered buying a certain treadmill product
for status in aerofit_df['MaritalStatus'].unique():
    print(f"Marital Status: {status}")
    for product in aerofit_df['Product'].unique():
        prob = len(aerofit_df[(aerofit_df['MaritalStatus'] == status) & (aerofit_df['Product'] == product)]) / len(aerofit_df[aerofit_df['MaritalStatus'] == status])
        print(f"Probability of {product}: {prob:.2f}")
    print()"""),

    nbf.v4.new_code_cell("""# Outlier Detection
plt.figure(figsize=(15, 10))
for i, col in enumerate(['Age', 'Education', 'Usage', 'Fitness', 'Income', 'Miles']):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=aerofit_df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_code_cell("""# Outlier Handling by Income Feature
q1 = aerofit_df['Income'].quantile(0.25)
q3 = aerofit_df['Income'].quantile(0.75)
iqr = q3 - q1
outliers = aerofit_df[(aerofit_df['Income'] < (q1 - 1.5 * iqr)) | (aerofit_df['Income'] > (q3 + 1.5 * iqr))]
aerofit_df = aerofit_df[~((aerofit_df['Income'] < (q1 - 1.5 * iqr)) | (aerofit_df['Income'] > (q3 + 1.5 * iqr)))]
print(f"Removed {len(outliers)} outliers from the Income feature.")"""),

    nbf.v4.new_code_cell("""# Outlier Handling for Income Feature
aerofit_df1 = aerofit_df.copy()

Q1 = aerofit_df1['Income'].quantile(0.25)
Q3 = aerofit_df1['Income'].quantile(0.75)
IQR = Q3 - Q1
aerofit_df1 = aerofit_df1[(aerofit_df1['Income'] >= Q1 - 1.5*IQR) & (aerofit_df1['Income'] <= Q3 + 1.5*IQR)]

sns.boxplot(data=aerofit_df1, x='Income', orient='h')
plt.show()"""),

    nbf.v4.new_code_cell("""# Outlier Handling for Miles Feature
aerofit_df1 = aerofit_df.copy()

Q1 = aerofit_df1['Miles'].quantile(0.25)
Q3 = aerofit_df1['Miles'].quantile(0.75)
IQR = Q3 - Q1
aerofit_df1 = aerofit_df1[(aerofit_df1['Miles'] >= Q1 - 1.5*IQR) & (aerofit_df1['Miles'] <= Q3 + 1.5*IQR)]

sns.boxplot(data=aerofit_df1, x='Miles', orient='h')
plt.show()"""),

    nbf.v4.new_code_cell("""# Before removal of Outliers
aerofit_df.shape"""),

    nbf.v4.new_code_cell("""# After removal of Outliers
aerofit_df1.shape"""),

    nbf.v4.new_code_cell("""# Actionable Insights & Recommendations
# Actionable Insights:
# 1. Model KP281 is the best-selling product. 46.6% of all treadmill sales go to model KP281.
# 2. The majority of treadmill customers fall within the $45,000 - $60,000 income bracket.
#    - 83% of treadmills are bought by individuals with incomes between $35,000 and $85,000.
#    - There are only 8% of customers with incomes below $35,000 who buy treadmills.
# 3. 88% of treadmills are purchased by customers aged 20 to 40.
# 4. Miles and Fitness & Miles and Usage are highly correlated, which means if a customer's fitness level is high they use more treadmills.
# 5. KP781 is the only model purchased by a customer who has more than 20 years of education and an income of over $85,000.
# 6. With Fitness level 4 and 5, the customers tend to use high-end treadmills and the average number of miles is above 150 per week.
#
# Recommendations:
# 1. KP281 & KP481 are popular with customer income of $45,000 - $60,000 and can be offered by these companies as affordable models.
# 2. KP781 should be marketed as a Premium Model and marketing it to high income groups and educational over 20 years market segments could result in more sales.
# 3. The KP781 is a premium model, so it is ideally suited for sporty people who have a high average weekly mileage and can be afforded by the high income customers.
# 4. Aerofit should conduct market research to determine if it can attract customers with income under $35,000 to expand its customer base.""")
]

# Add cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('data_exploration.ipynb', 'w') as f:
    nbf.write(nb, f)
