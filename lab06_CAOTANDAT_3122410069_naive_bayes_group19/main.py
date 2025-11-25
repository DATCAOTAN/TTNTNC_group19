import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
import time
from Naive_Bayes import Naive_Bayes


# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features.values
y = iris.data.targets.values.flatten()
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

#==================================================================

# # fetch dataset 
# letter_recognition = fetch_ucirepo(id=59) 
  
# # data (as pandas dataframes) 
# X = letter_recognition.data.features.values
# y = letter_recognition.data.targets.values.flatten()
  
# # metadata 
# print(letter_recognition.metadata) 
  
# # variable information 
# print(letter_recognition.variables) 

#==================================================================


# Chia set thành training và test data (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CLASSIFIER
print("Bắt đầu huấn luyện...")
start_time = time.time()
nb = Naive_Bayes(X_train, y_train)
train_time = time.time() - start_time
print(f"Huấn luyện xong trong: {train_time:.4f} giây")
print("Bắt đầu kiểm tra trên tập Test...")
# Predict (Batch processing - dự đoán 1 lần 4000 mẫu)
accuracy, predictions = nb.score(X_test, y_test)
print("-" * 30)
print(f"Độ chính xác (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print("-" * 30)

# In thử một vài mẫu dự đoán sai để kiểm tra
print("Một số mẫu dự đoán SAI:")
incorrect_indices = np.where(predictions != y_test)[0]
for i in incorrect_indices[:5]: # In 5 lỗi đầu tiên
    print(f"Input: {X_test[i]} -> Dự đoán: {predictions[i]}, Thực tế: {y_test[i]}")