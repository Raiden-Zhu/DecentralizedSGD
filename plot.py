from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
  
import torch


models_linear_tensor=torch.load('models_linear_tensor_1000.pt')
models_linear_tensor_difference = models_linear_tensor - models_linear_tensor.mean(0).unsqueeze(0).repeat(16,1,1)
models_linear_tensor_difference = models_linear_tensor_difference[:,0,:].unsqueeze(1)
models_releate_matrix = models_linear_tensor_difference.transpose(1,2).bmm(models_linear_tensor_difference)
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releatesingle_matrix_1000.pdf', format='pdf')
plt.close()  
# plt.show()


models_linear_tensor=torch.load('models_linear_tensor_2000.pt')
models_linear_tensor_difference = models_linear_tensor - models_linear_tensor.mean(0).unsqueeze(0).repeat(16,1,1)
models_linear_tensor_difference = models_linear_tensor_difference[:,0,:].unsqueeze(1)
models_releate_matrix = models_linear_tensor_difference.transpose(1,2).bmm(models_linear_tensor_difference)
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releatesingle_matrix_2000.pdf', format='pdf')
plt.close()  
# plt.show()


models_linear_tensor=torch.load('models_linear_tensor_3000.pt')
models_linear_tensor_difference = models_linear_tensor - models_linear_tensor.mean(0).unsqueeze(0).repeat(16,1,1)
models_linear_tensor_difference = models_linear_tensor_difference[:,0,:].unsqueeze(1)
models_releate_matrix = models_linear_tensor_difference.transpose(1,2).bmm(models_linear_tensor_difference)
data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

#绘制热度图：
plot=sns.heatmap(data)
plt.plot()
plt.savefig('models_releatesingle_matrix_3000.pdf', format='pdf')
plt.close()  
# plt.show()



# #练习的数据：
# models_releate_matrix=torch.load('models_releate_matrix_1000.pt')
# data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

# #绘制热度图：
# plot=sns.heatmap(data)
# plt.plot()
# plt.savefig('models_releate_matrix_1000.pdf', format='pdf')
# plt.close()  
# # plt.show()


# models_releate_matrix=torch.load('models_releate_matrix_2000.pt')
# data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

# #绘制热度图：
# plot=sns.heatmap(data)
# plt.plot()
# plt.savefig('models_releate_matrix_2000.pdf', format='pdf')
# plt.close() 



# models_releate_matrix=torch.load('models_releate_matrix_3000.pt')
# data=pd.DataFrame(models_releate_matrix.mean(0).numpy())

# #绘制热度图：
# plot=sns.heatmap(data)
# plt.plot()
# plt.savefig('models_releate_matrix_3000.pdf', format='pdf')
# plt.close()