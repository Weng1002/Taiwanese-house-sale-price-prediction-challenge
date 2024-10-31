# 智能所-Machine Learning Lab1-台灣房價預測回歸模型
## Author：國立陽明交通大學 資訊管理與財務金融學系財務金融所碩一 313707043 翁智宏

本次是機器學習第一次作業，是拿台灣真實房價的資料來去訓練一個模型，然後透過公開的房屋成交價來去預測準確的對應價格。  
參考資料：[點擊這裡](https://plvr.land.moi.gov.tw/Index) 內政部不動產成交案件實際資訊資料供應系統。

---

## 任務說明

1. **全台房價統計分析**  
   對各主要城市（如台北市、新竹市等）進行房價數據的統計分析。

2. **建立房價預測模型**  
   使用 `train.xlsx` 和 `valid.xlsx` 中的數據進行模型訓練，數據包含每筆房屋交易的記錄（包括 `id`、`price` 及多種房屋參數）。

3. **進行房價預測**  
   將 `test.xlsx` 中的房屋參數代入訓練好的模型，預測其房價。

4. **提交預測結果**  
   上傳預測結果至系統（從 “Submit Predictions” 連結）。  
   系統將計算 Mean Absolute Error（MAE），評估預測房價與實際房價的平均絕對差距。

5. **優化模型**  
   若 MAE 分數不夠理想，嘗試改進模型，提升預測準確度。

---

## 安裝依賴

請使用以下指令安裝本專案所需的依賴套件：

```bash
# 基本套件
!pip install numpy
!pip install matplotlib
!pip install pandas
!pip install openpyxl
!pip install scikit-learn
!pip install seaborn

# 深度學習框架
!pip install tensorflow
!pip install torch

# 載入部分模型
!pip install xgboost
!pip install lightgbm
!pip install catboost
```

---
## 實作

### 第一步：資料預處理
#### 1. 載入數據集
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Importing the dataset
df_train = pd.read_excel('Data/train.xlsx')
df_valid = pd.read_excel('Data/valid.xlsx')
df_test = pd.read_excel('Data/test-reindex-test.xlsx')



