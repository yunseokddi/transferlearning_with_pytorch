import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large'}

sns.set_style('whitegrid')
sns.set_context('talk')

plt.rcParams.update(params)

battles_df = pd.read_csv('../dataset/battles.csv')

# sns.countplot(y='year', data=battles_df)
# sns.countplot(x='region', data=battles_df)
# plt.title('Battle Distribution over Years')
# plt.show()

# attacker_king = battles_df.attacker_king.value_counts()
# attacker_king.name = ''
# attacker_king.plot.pie(figsize=(6,6), autopct='%.2f')
plt.title('Battle Distribution over Years')
plt.show()