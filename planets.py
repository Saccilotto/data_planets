import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('planets.csv')
print(df.head(9))
print(df.describe())

# Based on RotationPeriod
df.hist(column=["RotationPeriod", "DistancefromSun","Perihelion", "Aphelion"], layout=(1,4), figsize=(16,4))

df.plot.scatter(x='DistancefromSun', y='RotationPeriod')
df.plot.scatter(x='Perihelion', y='RotationPeriod')
df.plot.scatter(x='Aphelion', y='RotationPeriod')

df.plot.scatter(x='DistancefromSun', y='RotationPeriod')
df.plot.scatter(x='Perihelion', y='RotationPeriod')
df.plot.scatter(x='Aphelion', y='RotationPeriod')

sns.regplot(x = df.DistancefromSun, y = df.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df.Perihelion, y = df.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df.Aphelion, y = df.RotationPeriod, order=1, ci=None, scatter_kws={'color':'r', 's':9})

# Based on LengthofDay
df.hist(column=["LengthofDay", "DistancefromSun","Perihelion", "Aphelion"], layout=(1,4), figsize=(16,4))

df.plot.scatter(x='DistancefromSun', y='LengthofDay')
df.plot.scatter(x='Perihelion', y='LengthofDay')
df.plot.scatter(x='Aphelion', y='LengthofDay')

df.plot.scatter(x='DistancefromSun', y='LengthofDay')
df.plot.scatter(x='Perihelion', y='LengthofDay')
df.plot.scatter(x='Aphelion', y='LengthofDay')

sns.regplot(x = df.DistancefromSun, y = df.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df.Perihelion, y = df.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})
sns.regplot(x = df.Aphelion, y = df.LengthofDay, order=1, ci=None, scatter_kws={'color':'r', 's':9})

# Based on RotationPeriod
est = smf.ols('RotationPeriod ~ DistancefromSun', df).fit()
print('Sumario do modelo DistanFromSun em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.DistancefromSun, df.RotationPeriod, "b.")
plt.plot(df.DistancefromSun, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("DistancefromSun", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14)

#
est = smf.ols('RotationPeriod ~ Perihelion', df).fit()
print('Sumario do modelo Perihelion em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.Perihelion, df.RotationPeriod, "b.")
plt.plot(df.Perihelion, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Perihelion", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14) 

#
est = smf.ols('RotationPeriod ~ Aphelion', df).fit()
print('Sumario do modelo Aphelion em Relacao a RotationPeriod: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.Aphelion, df.RotationPeriod, "b.")
plt.plot(df.Aphelion, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Aphelion", fontsize=14) 
plt.ylabel("RotationPeriod", fontsize=14)

# Based on LengthofDay
est = smf.ols('LengthofDay ~ DistancefromSun', df).fit()
print('Sumario do modelo DistancefromSun em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.DistancefromSun, df.LengthofDay, "b.")
plt.plot(df.DistancefromSun, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("DistancefromSun", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14)

#
est = smf.ols('LengthofDay ~ Perihelion', df).fit()
print('Sumario do modelo Perihelion em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.Perihelion, df.LengthofDay, "b.")
plt.plot(df.Perihelion, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Perihelion", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14) 

#
est = smf.ols('LengthofDay ~ Aphelion', df).fit()
print('Sumario do modelo Aphelion em Relacao a LengthofDay: ', est.summary())
print('Resíduo: ', np.sqrt(est.mse_resid))

plt.plot(df.Aphelion, df.LengthofDay, "b.")
plt.plot(df.Aphelion, est.predict(df).values.reshape(-1,1), "r--", linewidth=1, label="val")
plt.xlabel("Aphelion", fontsize=14) 
plt.ylabel("LengthofDay", fontsize=14)

print('Correlação entre os preditores: ', df.corr())

# Based on RotationPeriod
df_train, df_test = train_test_split(df,test_size=0.10, random_state=24)

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
df_train, df_test = train_test_split(df,test_size=0.10, random_state=24)

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

#
plt.show()

