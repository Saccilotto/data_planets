# Bring data into workspace and replicate plots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


from sklearn.metrics import mean_squared_error

df_adv = pd.read_csv('planets.csv')
df_adv = df_adv.iloc[1: , :] #remove primeira linha.
df_adv.head(9)

df_adv.describe()

# Based on RotationPeriod

df_adv.hist(column=["RotationPeriod", "DistancefromSun","Perihelion", "Aphelion"], layout=(1,4), figsize=(16,4))

df_adv.plot.scatter(x='DistancefromSun', y='RotationPeriod')
df_adv.plot.scatter(x='Perihelion', y='RotationPeriod')
df_adv.plot.scatter(x='Aphelion', y='RotationPeriod')

df_adv.plot.scatter(x='DistancefromSun', y='RotationPeriod')
df_adv.plot.scatter(x='Perihelion', y='RotationPeriod')
df_adv.plot.scatter(x='Aphelion', y='RotationPeriod')

sns.regplot(x = df_adv.DistancefromSun, y = df_adv.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df_adv.Perihelion, y = df_adv.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df_adv.Aphelion, y = df_adv.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})

# Based on LengthofDay

df_adv.hist(column=["LengthofDay", "DistancefromSun","Perihelion", "Aphelion"], layout=(1,4), figsize=(16,4))

df_adv.plot.scatter(x='DistancefromSun', y='LengthofDay')
df_adv.plot.scatter(x='Perihelion', y='LengthofDay')
df_adv.plot.scatter(x='Aphelion', y='LengthofDay')

df_adv.plot.scatter(x='DistancefromSun', y='LengthofDay')
df_adv.plot.scatter(x='Perihelion', y='LengthofDay')
df_adv.plot.scatter(x='Aphelion', y='LengthofDay')

sns.regplot(x = df_adv.DistancefromSun, y = df_adv.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df_adv.Perihelion, y = df_adv.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df_adv.Aphelion, y = df_adv.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})

# Based on RotationPeriod

est = smf.ols('RotationPeriod ~ DistancefromSun', df_adv).fit()
print('Sumario do modelo DistanFromSun em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.DistancefromSun, df_adv.RotationPeriod, "b.")
plt.plot(df_adv.DistancefromSun, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("DistancefromSun", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14)

#
est = smf.ols('RotationPeriod ~ Perihelion', df_adv).fit()
print('Sumario do modelo Perihelion em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.Perihelion, df_adv.RotationPeriod, "b.")
plt.plot(df_adv.Perihelion, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Perihelion", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14) 

#
est = smf.ols('RotationPeriod ~ Aphelion', df_adv).fit()
print('Sumario do modelo Aphelion em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.Aphelion, df_adv.RotationPeriod, "b.")
plt.plot(df_adv.Aphelion, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Aphelion", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14)

# Based on LengthofDay

est = smf.ols('LengthofDay ~ DistancefromSun', df_adv).fit()
print('Sumario do modelo DistancefromSun em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.DistancefromSun, df_adv.LengthofDay, "b.")
plt.plot(df_adv.DistancefromSun, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("DistancefromSun", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14)

#
est = smf.ols('LengthofDay ~ Perihelion', df_adv).fit()
print('Sumario do modelo Perihelion em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.Perihelion, df_adv.LengthofDay, "b.")
plt.plot(df_adv.Perihelion, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Perihelion", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14) 

#
est = smf.ols('LengthofDay ~ Aphelion', df_adv).fit()
print('Sumario do modelo Aphelion em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df_adv.Aphelion, df_adv.LengthofDay, "b.")
plt.plot(df_adv.Aphelion, est.predict(df_adv).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Aphelion", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14)

print('Correlação entre os preditores: ', df_adv.corr())

# Based on RotationPeriod
df_train, df_test = train_test_split(df_adv,test_size=0.10, random_state=24)

X_train = df_train[["DistancefromSun","Perihelion", "Aphelion"]].values.reshape(-1,np.size([["DistancefromSun","Perihelion", "Aphelion"]]))
X_test = df_test[["DistancefromSun","Perihelion", "Aphelion"]].values.reshape(-1,np.size([["DistancefromSun","Perihelion", "Aphelion"]]))
y_train = df_train['RotationPeriod'].values.reshape(-1,1)
y_test = df_test['RotationPeriod'].values.reshape(-1,1)

modeloRotation = LinearRegression()
modeloRotation.fit(X_train, y_train)

print('Coeficientes estimados pelo método linear para cada feature (Rotation Model): ', modeloRotation.coef_)

y_train_modeloRotation = modeloRotation.predict(X_train)
y_test_modeloRotation = modeloRotation.predict(X_test)

MSE_train = mean_squared_error(y_train_modeloRotation, y_train)
MSE_test = mean_squared_error(y_test_modeloRotation, y_test)

MSE_modelo = [MSE_train, MSE_test]

# Based on LengthofDay
df_train, df_test = train_test_split(df_adv,test_size=0.10, random_state=24)

X_train = df_train[["DistancefromSun","Perihelion", "Aphelion"]].values.reshape(-1,np.size([["DistancefromSun","Perihelion", "Aphelion"]]))
X_test = df_test[["DistancefromSun","Perihelion", "Aphelion"]].values.reshape(-1,np.size([["DistancefromSun","Perihelion", "Aphelion"]]))
y_train = df_train['LengthofDay'].values.reshape(-1,1)
y_test = df_test['LengthofDay'].values.reshape(-1,1)

modeloDayLength = LinearRegression()
modeloDayLength.fit(X_train, y_train)

print('Coeficientes estimados pelo método linear para cada feature (DayLength Model): ', modeloDayLength.coef_)

y_train_modeloDayLength = modeloDayLength.predict(X_train)
y_test_modeloDayLength = modeloDayLength.predict(X_test)

MSE_train = mean_squared_error(y_train_modeloDayLength, y_train)
MSE_test = mean_squared_error(y_test_modeloDayLength, y_test)

MSE_modelo = [MSE_train, MSE_test]

print('Projeções de Treino e Teste para o modelo (modeloDayLength): ', MSE_modelo)
