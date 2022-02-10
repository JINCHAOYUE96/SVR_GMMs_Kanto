
import numpy as np 
from sklearn.svm import SVR    
from sklearn.model_selection import cross_val_score  
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  
import pandas as pd  
import matplotlib.pyplot as plt
import xlwt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# read the data of Kanto region
raw_data = pd.read_csv(r'C:\Users\a\Desktop\SVR-GMMs-Kanto\SVR_GMMs_IA\IA\DataAI2.csv',encoding='GBK')
X = raw_data.iloc[:, [1,2,3]].values
y = raw_data.iloc[:,[0,4]].values
print("X",X[0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print("X_train",len(X_train))
print("X_test",len(X_test))
print(type(X_train))

# Training set
data_df_xunlian=pd.DataFrame(y_train)
data_df_xunlian.columns=['IA','zh']
data_xunlian=pd.DataFrame(X_train)
data_xunlian.columns=['ln_distance','M','ln_Vs30']
writer_xunlian=pd.ExcelWriter(r'C:\Users\a\Desktop\SVR-GMMs-Kanto\SVR_GMMs_IA\IA\train.xls')
data_df_xunlian.to_excel(writer_xunlian)
data_xunlian.to_excel(writer_xunlian,startcol=3)
writer_xunlian.save()

#Testing set
data_df_ceshi=pd.DataFrame(y_test)
data_df_ceshi.columns=['IA','zh']
data_ceshi=pd.DataFrame(X_test)
data_ceshi.columns=['ln_distance','M','ln_Vs30']
writer_ceshi=pd.ExcelWriter(r'C:\Users\a\Desktop\SVR-GMMs-Kanto\SVR_GMMs_IA\IA\test.xls')
data_df_ceshi.to_excel(writer_ceshi)
data_ceshi.to_excel(writer_ceshi,startcol=3)
writer_ceshi.save()



workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
worksheet.write(0, 0, label="Actual Value")
worksheet.write(0, 1, label="SVR Value")
raw_data = pd.read_csv(r'C:\Users\a\Desktop\SVR-GMMs-Kanto\SVR_GMMs_IA\IA\DataAI2.csv',encoding='GBK')
X = raw_data.iloc[:, [1,2,3]].values
y = np.log(raw_data.iloc[:,0].values)  
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
print(y_train)

# Training regression model
n_folds = 6  # Sets the number of cross-checks
model_svr = SVR(kernel="rbf",gamma=0.279,C=7.386,epsilon=0.129)  # Set the hyperparameters of SVR GMMs
model_names = ['SVR']
model_dic = [model_svr]
cv_score_list = []
pre_y_list = []
for model in model_dic:
    scores = cross_val_score(model, X_train, y_train, cv=n_folds)
    cv_score_list.append(scores)
    pre_y_list.append(model.fit(X_train, y_train).predict(X_train))
print(model_svr)

#模型效果指标评估
n_samples, n_features = X_train.shape  
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
model_metrics_list = []
for i in range(1):
    tmp_list = []  #
    for m in model_metrics_name:
        tmp_score = m(y_train, pre_y_list[i])
        tmp_list.append(tmp_score)
    model_metrics_list.append(tmp_list)
df1 = pd.DataFrame(cv_score_list, index=model_names)
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])
print ('samples: %d \t features: %d' % (n_samples, n_features))
print (70 * '-')
print ('cross validation result:')
print (df1)
print (70 * '-')
print ('regression metrics:')
print (df2)
print (70 * '-')
print ('short name \t full name')
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')

for i in range(len(y_train)):
    worksheet.write(i+1, 0, label=y_train[i])
    worksheet.write(i+1, 1, label=pre_y_list[0][i])
workbook.save('SVR_true_predict_training_set.xls')


# 模型效果可视化
plt.plot(np.arange(X_train.shape[0]), y_train, color='k', label='true y')
color_list = [ 'g']
linestyle_list = [ 'v']
plt.plot(np.arange(X_train.shape[0]), pre_y_list[0], color_list[0], label=model_names[0])
plt.title('regression result comparison')
plt.legend(loc='upper right')
plt.ylabel('real and predicted value')
plt.show()


# test
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet2')
worksheet.write(0, 0, label="Actual Value")
worksheet.write(0, 1, label="SVR Value")
new_pre_SVR_y = []
new_pre_br_y = []
new_pre_lr_y = []
new_pre_etc_y = []
new_pre_gbr_y=[]
for i, new_point in enumerate(X_test):
    new_point=np.array(new_point).reshape(1,3)
    pre_SVR_y = model_svr.predict(new_point)
    new_pre_SVR_y.append(pre_SVR_y)
   
print(type(y_test))
data=[]
for i in range(len(np.array(new_pre_SVR_y))):
    aa=len(np.array(new_pre_SVR_y)[i])
    for j in range(aa):
        data.append(np.array(new_pre_SVR_y)[i][j])
print(len(y_test-data))
print("standard deviation=",np.std(y_test-data))
print(type(pre_y_list[0][0]))
print("MAE=",sum(np.abs(y_test-data))/len(y_test))

# 数据准备
for i in range(len(y_test)):
    worksheet.write(i+1, 0, label=y_test[i])
    worksheet.write(i+1, 1, label=new_pre_SVR_y[i][0])
workbook.save('SVR_true_predict_test_set.xls')


# prediction
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet3')
worksheet.write(0, 0, label="Kanto")
worksheet.write(0, 1, label="SVR")
worksheet.write(0, 2, label="log(distance)")
worksheet.write(0, 3, label="M")
worksheet.write(0, 4, label="log(Vs30)")
new_pre_SVR_y = []
raw_data = pd.read_csv(r'C:\Users\a\Desktop\SVR-GMMs-Kanto\SVR_GMMs_IA\IA\shujv2222-11.csv',encoding='GBK') # input the information for prediction (R,M,Vs30)
X_test1 = raw_data.iloc[:, [1,2,3]].values
y_test=np.array(raw_data.iloc[:, 0].values)
print(y_test)
for i, new_point in enumerate(X_test1):
    new_point=np.array(new_point).reshape(1,3)
    pre_SVR_y = model_svr.predict(new_point)
    print(pre_SVR_y)
    print(i)
    new_pre_SVR_y.append(pre_SVR_y)
print(np.array(new_pre_SVR_y))
data=[]
for i in range(len(y_test)):  
     worksheet.write(i+1,1, label=new_pre_SVR_y[i][0])
     worksheet.write(i+1,2,label=X_test1[i][0])
     worksheet.write(i+1,3,label=X_test1[i][1])
     worksheet.write(i+1,4,label=X_test1[i][2])
workbook.save('qvxian1.xls') # The prediction results given by SVR GMMs
print(X_test1[0][0])