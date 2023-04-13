---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Fisher Linear Discriminant Analysis (LDA)

LDA is a widely used dimensionality reduction technique built on Fisher’s linear discriminant.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Dataset

```python
raw_data = pd.read_csv('data/A_multivariate_study_of_variation_in_two_species_of_rock_crab_of_genus_Leptograpsus.csv')

data = raw_data.rename(columns={
    'sp': 'Species',
    'sex': 'Sex',
    'index': 'Index',
    'FL': 'Frontal Lobe',
    'RW': 'Rear Width',
    'CL': 'Carapace Midline',
    'CW': 'Maximum Width',
    'BD': 'Body Depth'})

data['Species'] = data['Species'].map({'B':'Blue', 'O':'Orange'})
data['Sex'] = data['Sex'].map({'M':'Male', 'F':'Female'})
data['Class'] = data.Species + data.Sex

data_columns = ['Frontal Lobe',
                'Rear Width',
                'Carapace Midline',
                'Maximum Width',
                'Body Depth']
```

```python
# generate a class variable for all 4 classes
data['Class'] = data.Species + data.Sex

print(data['Class'].value_counts())
data.head(5)
```

```python
# normalize data columns
data_norm = data.copy()
data_norm[data_columns] = MinMaxScaler().fit_transform(data[data_columns])

data_norm.describe().T
```

## 2-Dimensional Plot

```python
no_components = 2

lda = LinearDiscriminantAnalysis(n_components = no_components)
data_lda = lda.fit_transform(data_norm[data_columns].values , y=data_norm['Class'])

data_norm[['LDA1', 'LDA2']] = data_lda

data_norm.head(1)
```

|  | Species | Sex | Index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth | Class | LDA1 | LDA2 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 0.056604 | 0.014599 | 0.042553 | 0.050667 | 0.058065 | BlueMale | 1.538869 | -0.808137 |

```python
fig = plt.figure(figsize=(10, 8))
sns.scatterplot(x='LDA1', y='LDA2', hue='Class', data=data_norm)
```

![Fisher Linear Discriminant Analysis (LDA)](https://github.com/mpolinowski/fisher-discriminant-analysis/blob/master/assets/Linear_Discriminant_Analysis_01.png)

![Fisher Linear Discriminant Analysis (LDA)](https://github.com/mpolinowski/fisher-discriminant-analysis/blob/master/assets/nice.gif)


## 3-Dimensional Plot

```python
no_components = 3

lda = LinearDiscriminantAnalysis(n_components = no_components)
data_lda = lda.fit_transform(data_norm[data_columns].values , y=data_norm['Class'])

data_norm[['LDA1', 'LDA2', 'LDA3']] = data_lda

data_norm.head(1)
```

|  | Species | Sex | Index | Frontal Lobe | Rear Width | Carapace Midline | Maximum Width | Body Depth | Class | LDA1 | LDA2 | LDA3 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| 0 | Blue | Male | 1 | 0.056604 | 0.014599 | 0.042553 | 0.050667 | 0.058065 | BlueMale | 1.538869 | -0.808137 | 1.18642 |

```python
class_colours = {
    'BlueMale': '#0027c4', #blue
    'BlueFemale': '#f18b0a', #orange
    'OrangeMale': '#0af10a', # green
    'OrangeFemale': '#ff1500', #red
}

colours = data_norm['Class'].apply(lambda x: class_colours[x])

x=data_norm.LDA1
y=data_norm.LDA2
z=data_norm.LDA3

fig = plt.figure(figsize=(10,10))
plt.title('Linear Discriminant Analysis')
ax = fig.add_subplot(projection='3d')

ax.scatter(xs=x, ys=y, zs=z, s=50, c=colours)
```

![Fisher Linear Discriminant Analysis (LDA)](https://github.com/mpolinowski/fisher-discriminant-analysis/blob/master/assets/Linear_Discriminant_Analysis_02.png)