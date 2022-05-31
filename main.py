import pandas as pd
import numpy as np

bsas = pd.read_csv('bsasMin.csv')
ero = pd.read_csv('eroMin.csv')


jvg = pd.read_csv('jvgMin.csv')
lj = pd.read_csv('lujanMin.csv')
lq = pd.read_csv('LQ.csv')

monte = pd.read_csv('monteMin.csv')

salta = pd.read_csv('saltaMin.csv')
sanco = pd.read_csv('Sanco.csv')
tf = pd.read_csv('tafivMin.csv')

print(tf.describe())


laviña = pd.read_csv('laviñaMin.csv')



df = pd.read_csv('dataMin.csv', sep=";")



df['KTR'] = df['KTR'].str.replace(',', '.', regex=True)
df['KTR'] = pd.to_numeric(df['KTR'])
    

df = df.sort_values(by=['ALT'], ascending=False)


df = df[df['ALT']>1000]



def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

# aArray = np.arange(0.0, 0.01, 0.01)
# bArray = np.arange(0.5, 0.6, 0.0001 )
# cArray = np.arange(0.00, 0.55, 0.001)

# mirRMSE = 25
# miA=1
# miB=1
# miC=1
# for a in aArray:
#     for b in bArray:
#         for c in cArray:
#             values = []
#             n_kt = [(a + b*A**c)  for A in df['ALT']]
#             rrmse = relative_root_mean_squared_error(df['KTR'], n_kt)
#             print(f" a: {a} b:{b}  c:{c} rrmse:{rrmse * 100} ")
#             if(rrmse < mirRMSE):
#                 miA = a
#                 miB = b
#                 miC = c
#                 mirRMSE = rrmse



# f = np.polyfit(df['ALT'], df['KTR'], 3)
g = np.polyfit(df['ALT'], df['KTR'], 3)
# h = np.polyfit(df['ALT'], df['KTR'], 4)


# p = np.poly1d(f)
q = np.poly1d(g)
# r = np.poly1d(h)
print(q(4515))

# predict_3 = [p(i) for i in range(0, 5000)]
# predict_2 = [q(i) for i in range(0, 5000)]
# predict_4 = [r(i) for i in range(0, 5000)]
# predict_5 = [0.5614*i**0.0510 for i in range(0, 5000)]





import matplotlib.pyplot as plt
plt.style.use('classic')
plt.scatter(df['ALT'], df['KTR'])
#plt.plot(predict_3, 'r' ,label="n=3", linewidth=2)
#plt.plot(predict_2, 'g' ,label="n=2", linewidth=2)
#plt.plot(predict_5, 'b' ,label="n=4", linewidth=2)
plt.ylabel('KTRP')
plt.xlabel('ALTURA')
plt.legend()
plt.grid()
plt.title('ALT vs KTRP')

ax = plt.gca()
ax.set_xlim([0, 5000])
ax.set_ylim([0.3, 1])

plt.show()









