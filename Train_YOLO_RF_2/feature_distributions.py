# feature distributions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

feature_names = ['prebreaking', 'curling', 'splashing', 'whitewash', 'crumbling']
plunging_percent = np.load('.\\plunging_labels.npy')
spilling_percent = np.load('.\\spilling_labels.npy')

#print(spilling_percent)

df_plunging = pd.DataFrame(plunging_percent, columns=feature_names)
df_plunging = df_plunging.mask(df_plunging < 0.01)

df_spilling = pd.DataFrame(spilling_percent, columns=feature_names)
df_spilling = df_spilling.mask(df_spilling < 0.01)


plt.figure(0)
plt0 = sns.kdeplot(df_plunging, palette = 'colorblind')
#plt0 = sns.kdeplot(filtered_month, x="percent", hue = "year", bw_adjust=2, palette = 'colorblind')
plt.xlabel("$percent$")

plt.show()

plt.figure(1)
plt0 = sns.kdeplot(df_spilling, palette = 'colorblind')
#plt0 = sns.kdeplot(filtered_month, x="percent", hue = "year", bw_adjust=2, palette = 'colorblind')
plt.xlabel("$percent$")

plt.show()

#plt.hist(plunging_percent)