from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
  
import torch

#练习的数据：
models_releate_matrix=torch.load('models_releate_matrix_1000.pt')
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releate_matrix_1000.pdf', format='pdf')
plt.close()  
# plt.show()


models_releate_matrix=torch.load('models_releate_matrix_2000.pt')
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releate_matrix_2000.pdf', format='pdf')
plt.close() 



models_releate_matrix=torch.load('models_releate_matrix_3000.pt')
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releate_matrix_3000.pdf', format='pdf')
plt.close()