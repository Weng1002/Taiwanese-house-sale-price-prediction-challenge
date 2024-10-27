import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# Importing the dataset
df_train = pd.read_excel('Data/train.xlsx')
df_valid = pd.read_excel('Data/valid.xlsx')
df_test = pd.read_excel('Data/test-reindex-test.xlsx')


# 重新調整train、valid資料集比例(80/20)
df_all = pd.concat([df_train, df_valid], axis=0)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
df_valid_ratio = 0.2
df_valid_size = int(len(df_all) * df_valid_ratio)
df_train = df_all[:-df_valid_size]
df_valid = df_all[-df_valid_size:]


# 刪除掉raw data中有"main use"的row
df_train = df_train[~df_train['主要用途'].str.contains("main use", na=False)]

# 刪除不重要的特徵
df_train = df_train.drop(['編號', '解約情形', '棟及號', '交易標的','移轉層次','總樓層數', '非都市土地使用編定' , '有無管理組織' ,'備註', '建案名稱', '建築完成年月', '交易年月日'], axis=1)
df_valid = df_valid.drop(['編號', '解約情形', '棟及號', '交易標的','移轉層次','總樓層數', '非都市土地使用編定' , '有無管理組織' ,'備註', '建案名稱', '建築完成年月', '交易年月日'], axis=1)
df_test = df_test.drop(['編號', '解約情形', '棟及號', '交易標的','移轉層次','總樓層數', '非都市土地使用編定' , '有無管理組織' ,'備註', '建案名稱', '建築完成年月', '交易年月日'], axis=1)
df_train = df_train.drop(['鄉鎮市區'], axis=1)
df_valid = df_valid.drop(['鄉鎮市區'], axis=1)
df_test = df_test.drop(['鄉鎮市區'], axis=1)
df_train = df_train[df_train['土地移轉總面積平方公尺'] < 8000]
df_train = df_train[df_train['建物移轉總面積平方公尺'] < 15000] 
df_train = df_train[df_train['車位移轉總面積平方公尺'] < 4000]
df_train = df_train[df_train['車位總價元'] < 14000000]

# 處理空值
# 合併土地使用分區，然後空值填0
df_train['合併後土地使用分區'] = df_train['都市土地使用分區'].combine_first(df_train['非都市土地使用分區']).fillna('其他')
df_valid['合併後土地使用分區'] = df_valid['都市土地使用分區'].combine_first(df_valid['非都市土地使用分區']).fillna('其他')
df_test['合併後土地使用分區'] = df_test['都市土地使用分區'].combine_first(df_test['非都市土地使用分區']).fillna('其他')

# 將主要用途填滿
def fill_main_usage(df):
    conditions = [
        (df['合併後土地使用分區'] == '住'),
        (df['合併後土地使用分區'].isin(['商', '工'])),
        (df['合併後土地使用分區'] == '農'),
        (df['合併後土地使用分區'] == '其他'),
        (df['合併後土地使用分區'] == '工業區'),
        (df['合併後土地使用分區'] == '鄉村區')
    ]
    choices = ['住家用', '工商用','農業用', '工商用', '工業用', '住家用']   
    fill_values = pd.Series(np.select(conditions, choices, default=df['主要用途']), index=df.index)
    df['主要用途'] = df['主要用途'].combine_first(fill_values)
   
    return df

df_train = fill_main_usage(df_train)
df_valid = fill_main_usage(df_valid)
df_test = fill_main_usage(df_test)
df_train = df_train.drop(['都市土地使用分區', '非都市土地使用分區'], axis=1)
df_valid = df_valid.drop(['都市土地使用分區', '非都市土地使用分區'], axis=1)
df_test = df_test.drop(['都市土地使用分區', '非都市土地使用分區'], axis=1)
df_train[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']] = df_train[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].fillna(df_train[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].mean())
df_valid[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']] = df_valid[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].fillna(df_valid[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].mean())
df_test[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']] = df_test[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].fillna(df_test[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺']].mean())
df_train['單價元平方公尺'] = df_train['單價元平方公尺'].fillna(df_train['單價元平方公尺'].mean())
df_valid['單價元平方公尺'] = df_valid['單價元平方公尺'].fillna(df_valid['單價元平方公尺'].mean())
df_test['單價元平方公尺'] = df_test['單價元平方公尺'].fillna(df_test['單價元平方公尺'].mean())

# 將主要建材空值填0
df_train = df_train.dropna(subset=['主要建材'])
df_valid = df_valid.dropna(subset=['主要建材'])
df_test['主要建材'] = df_test['主要建材'].fillna("見使用執照")

# 將車位類別空值填0
df_train['車位類別'] = df_train['車位類別'].fillna("無")
df_valid['車位類別'] = df_valid['車位類別'].fillna("無")
df_test['車位類別'] = df_test['車位類別'].fillna("無")

# 將建物現況格局空值填0
for col in ['建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛']:
    median_value = df_train[col].median()
    df_train[col].fillna(0, inplace=True)
    df_valid[col].fillna(0, inplace=True)
    df_test[col].fillna(0, inplace=True)


# 針對 縣市進行 label embedding
city_to_price = {
    '臺北市': 217000,  
    '台北市': 217000,  
    '新北市': 70000,   
    '新竹市': 40000,  
    '新竹縣': 36000,   
    '基隆市': 30000,  
    '桃園市': 30000,   
    '桃園縣': 30000,   
    '臺中市': 32000,   
    '台中市': 32000,   
    '高雄市': 25000,   
    '臺南市': 25000,   
    '台南市': 25000,
    '彰化縣': 20000,   
    '苗栗縣': 22000,   
    '南投縣': 20000,   
    '嘉義市': 22000,  
    '嘉義縣': 20000,   
    '宜蘭縣': 22000,  
    '雲林縣': 20000,  
    '屏東縣': 20000, 
    '花蓮縣': 18000, 
    '臺東縣': 18000,   
    '金門縣': 18000,
    '澎湖縣': 19000,  
}

# 提取 "土地位置建物門牌" 的前面三個字作為 "縣市"
df_train['縣市'] = df_train['土地位置建物門牌'].str[:3]
df_valid['縣市'] = df_valid['土地位置建物門牌'].str[:3]
df_test['縣市'] = df_test['土地位置建物門牌'].str[:3]

# 只保留那些 "縣市" 是 city_to_price 字典中的資料，其他捨棄
df_train = df_train[df_train['縣市'].isin(city_to_price.keys())]
df_valid = df_valid[df_valid['縣市'].isin(city_to_price.keys())]
df_test = df_test[df_test['縣市'].isin(city_to_price.keys())]

# 將 "縣市" 欄位替換成對應的平均房價
df_train['平均房價'] = df_train['縣市'].replace(city_to_price)
df_valid['平均房價'] = df_valid['縣市'].replace(city_to_price)
df_test['平均房價'] = df_test['縣市'].replace(city_to_price)


scaler = StandardScaler()

features_train = df_train[['平均房價']]
scaled_features_train = scaler.fit_transform(features_train)
df_train['平均房價'] = scaled_features_train[:, 0]

features_valid = df_valid[['平均房價']]
scaled_features_valid = scaler.transform(features_valid)
df_valid['平均房價'] = scaled_features_valid[:, 0]

features_test = df_test[['平均房價']]
scaled_features_test = scaler.transform(features_test)
df_test['平均房價'] = scaled_features_test[:, 0]

# 移除原本的 "土地位置建物門牌" 和 "縣市" 欄位
df_train = df_train.drop(['土地位置建物門牌', '縣市'], axis=1)
df_valid = df_valid.drop(['土地位置建物門牌', '縣市'], axis=1)
df_test = df_test.drop(['土地位置建物門牌', '縣市'], axis=1)

# 針對 移轉樓層和總樓層 進行 label embedding、'土地移轉總面積平方公尺', '建物移轉總面積平方公尺'進行PCA降維合併
pca = PCA(n_components=1)
area_features_train = df_train[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺','車位移轉總面積平方公尺']]
pca_train_features = pca.fit_transform(area_features_train)
df_train['PCA_Area'] = pca_train_features

area_features_valid = df_valid[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺','車位移轉總面積平方公尺']]
pca_valid_features = pca.transform(area_features_valid)
df_valid['PCA_Area'] = pca_valid_features

area_features_test = df_test[['土地移轉總面積平方公尺', '建物移轉總面積平方公尺','車位移轉總面積平方公尺']]
pca_test_features = pca.transform(area_features_test)
df_test['PCA_Area'] = pca_test_features

# 生成一個新特徵 土地移轉總面積平方公尺*單價元平方公尺 和 建物平方公尺乘上單價元
df_train['建物平方公尺乘上單價元'] = df_train['建物移轉總面積平方公尺'] * df_train['單價元平方公尺']
df_valid['建物平方公尺乘上單價元'] = df_valid['建物移轉總面積平方公尺'] * df_valid['單價元平方公尺']
df_test['建物平方公尺乘上單價元'] = df_test['建物移轉總面積平方公尺'] * df_test['單價元平方公尺']

# 針對 主要用途 進行 label embedding
use_to_price = {
    '住家用': 60000,     
    '商業用': 90000,    
    '住商用': 75000,    
    '工商用': 70000,     
    '工業用': 50000,  
    '住工用': 55000,     
    '國民住宅': 35000,   
    '停車空間': 20000,   
    '農業用': 15000,     
    '見使用執照': 60000,  
    '見其他登記事項': 60000,  
    '見其它登記事項': 60000,  
}
df_train['主要用途'] = df_train['主要用途'].replace(use_to_price)
df_train['主要用途'] = df_train['主要用途'].fillna(np.mean(list(use_to_price.values())))
scaler = StandardScaler()
df_train[['主要用途']] = scaler.fit_transform(df_train[['主要用途']])

df_valid['主要用途'] = df_valid['主要用途'].replace(use_to_price)
df_valid['主要用途'] = df_valid['主要用途'].fillna(np.mean(list(use_to_price.values())))
df_valid[['主要用途']] = scaler.transform(df_valid[['主要用途']])

df_test['主要用途'] = df_test['主要用途'].replace(use_to_price)
df_test['主要用途'] = df_test['主要用途'].fillna(np.mean(list(use_to_price.values())))
df_test[['主要用途']] = scaler.transform(df_test[['主要用途']])

# 針對 主要建材 進行 label embedding
material_to_price = {
    '鋼筋混凝土造': 50.0,
    '鋼筋混凝土構造': 50.0,
    '鋼骨混凝土造': 55.0,
    '鋼骨鋼筋混凝土造': 60.0,
    '鋼筋混凝土加強磚造': 45.0,
    '混凝土造': 40.0,
    '鋼造': 35.0,
    '磚造': 30.0,
    '預力混凝土造': 42.0,
    '石造': 28.0,
    '木造': 25.0,
    '見使用執照': 40.0,  
    '見其他登記事項': 40.0,
    '見其它登記事項': 40.0,  
}
df_train['主要建材'] = df_train['主要建材'].replace(material_to_price)
df_train['主要建材'] = df_train['主要建材'].fillna(np.mean(list(material_to_price.values())))

scaler = StandardScaler()
df_train[['主要建材']] = scaler.fit_transform(df_train[['主要建材']])
df_valid['主要建材'] = df_valid['主要建材'].replace(material_to_price)
df_valid['主要建材'] = df_valid['主要建材'].fillna(np.mean(list(material_to_price.values())))
df_valid[['主要建材']] = scaler.transform(df_valid[['主要建材']])

df_test['主要建材'] = df_test['主要建材'].replace(material_to_price)
df_test['主要建材'] = df_test['主要建材'].fillna(np.mean(list(material_to_price.values())))
df_test[['主要建材']] = scaler.transform(df_test[['主要建材']])

# 針對 建物型態 進行 label embedding
type_to_price = {
    '住宅大樓(11層含以上有電梯)': 50.0,
    '華廈(10層含以下有電梯)': 45.0,
    '店面(店鋪)': 100.0,
    '辦公商業大樓': 80.0,
    '公寓(5樓含以下無電梯)': 35.0,
    '透天厝': 30.0,
    '廠辦': 40.0,
    '工廠': 35.0,
    '套房(1房1廳1衛)': 40.0,
    '倉庫': 30.0,
    '農舍': 20.0,
    '其他': 40.0,
}

df_train['建物型態'] = df_train['建物型態'].replace(type_to_price)
df_train['建物型態'] = df_train['建物型態'].fillna(np.mean(list(type_to_price.values())))

scaler = StandardScaler()
df_train[['建物型態']] = scaler.fit_transform(df_train[['建物型態']])
df_valid['建物型態'] = df_valid['建物型態'].replace(type_to_price)
df_valid['建物型態'] = df_valid['建物型態'].fillna(np.mean(list(type_to_price.values())))
df_valid[['建物型態']] = scaler.transform(df_valid[['建物型態']])

df_test['建物型態'] = df_test['建物型態'].replace(type_to_price)
df_test['建物型態'] = df_test['建物型態'].fillna(np.mean(list(type_to_price.values())))
df_test[['建物型態']] = scaler.transform(df_test[['建物型態']])

# 針對 車位類別 進行 label embedding、再對車位屬性進行PCA降維
parking_type_to_price = {
    '坡道平面': 150.0,
    '塔式車位': 80.0,
    '坡道機械': 100.0,
    '升降機械': 70.0,
    '升降平面': 90.0,
    '一樓平面': 200.0,
    '其他': 100.0,
    '無': 0.0,
}

def process_parking_features(df):
    average_price = np.mean(list(parking_type_to_price.values()))
    df['車位類別'] = df['車位類別'].replace(parking_type_to_price)
    df['車位類別'] = df['車位類別'].fillna(average_price)
    df['車位移轉總面積平方公尺'] = df['車位移轉總面積平方公尺'].fillna(df['車位移轉總面積平方公尺'].median())
    df['車位總價元'] = df['車位總價元'].fillna(df['車位總價元'].median())

    # 計算車位單價（每平方公尺價格）
    df['車位單價元平方公尺'] = df['車位總價元'] / df['車位移轉總面積平方公尺']
    df['車位單價元平方公尺'] = df['車位單價元平方公尺'].replace([np.inf, -np.inf], np.nan).fillna(df['車位單價元平方公尺'].median())
    features = df[['車位類別', '車位移轉總面積平方公尺', '車位單價元平方公尺']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled

parking_features_train = process_parking_features(df_train)
parking_features_valid = process_parking_features(df_valid)
parking_features_test = process_parking_features(df_test)

pca = PCA(n_components=1)
parking_pca_train = pca.fit_transform(parking_features_train)
parking_pca_valid = pca.transform(parking_features_valid)
parking_pca_test = pca.transform(parking_features_test)
df_train['Parking_PCA'] = parking_pca_train
df_valid['Parking_PCA'] = parking_pca_valid
df_test['Parking_PCA'] = parking_pca_test

df_train['車位'] = df_train['交易筆棟數'].str.extract(r'車位(\d+)').astype(float)
df_valid['車位'] = df_valid['交易筆棟數'].str.extract(r'車位(\d+)').astype(float)
df_test['車位'] = df_test['交易筆棟數'].str.extract(r'車位(\d+)').astype(float)


def calculate_total_prices(df):
    df['車位'] = df['交易筆棟數'].str.extract(r'車位(\d+)').astype(float)
    df['車位總價'] = df['車位'] * df['車位總價元']

    return df

df_train = calculate_total_prices(df_train)
df_valid = calculate_total_prices(df_valid)
df_test = calculate_total_prices(df_test)

df_train = df_train.drop(['交易筆棟數'], axis=1)
df_valid = df_valid.drop(['交易筆棟數'], axis=1)
df_test = df_test.drop(['交易筆棟數'], axis=1)

df_train = df_train.drop(['車位'], axis=1)
df_valid = df_valid.drop(['車位'], axis=1)
df_test = df_test.drop(['車位'], axis=1)

# 針對 都市土地使用分區 進行 label embedding
zone_to_price = {
    '住': 50.0,        
    '商': 80.0,        
    '工': 30.0,        
    '農': 15.0,        
    '其他': 45.0,      
    '山坡地保育區': 20.0,  
    '特定農業區': 15.0,    
    '鄉村區': 25.0,      
    '一般農業區': 15.0,  
    '工業區': 30.0,     
    '特定專用區': 40.0,  
    '風景區': 60.0      
}

df_train['合併後土地使用分區'] = df_train['合併後土地使用分區'].replace(zone_to_price)
df_train['合併後土地使用分區'] = df_train['合併後土地使用分區'].fillna(np.mean(list(zone_to_price.values())))

scaler = StandardScaler()
df_train[['合併後土地使用分區']] = scaler.fit_transform(df_train[['合併後土地使用分區']])

df_valid['合併後土地使用分區'] = df_valid['合併後土地使用分區'].replace(zone_to_price)
df_valid['合併後土地使用分區'] = df_valid['合併後土地使用分區'].fillna(np.mean(list(zone_to_price.values())))
df_valid[['合併後土地使用分區']] = scaler.transform(df_valid[['合併後土地使用分區']])

df_test['合併後土地使用分區'] = df_test['合併後土地使用分區'].replace(zone_to_price)
df_test['合併後土地使用分區'] = df_test['合併後土地使用分區'].fillna(np.mean(list(zone_to_price.values())))
df_test[['合併後土地使用分區']] = scaler.transform(df_test[['合併後土地使用分區']])


# 對 '建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛' 進行加總
for col in ['建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛']:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

df_train['總間數'] = df_train['建物現況格局-房'] + df_train['建物現況格局-廳'] + df_train['建物現況格局-衛']
df_valid['總間數'] = df_valid['建物現況格局-房'] + df_valid['建物現況格局-廳'] + df_valid['建物現況格局-衛']
df_test['總間數'] = df_test['建物現況格局-房'] + df_test['建物現況格局-廳'] + df_test['建物現況格局-衛']

df_train = df_train.drop(['建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛'], axis=1)
df_valid = df_valid.drop(['建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛'], axis=1)
df_test = df_test.drop(['建物現況格局-房', '建物現況格局-廳', '建物現況格局-衛'], axis=1)
numeric_features = ['總間數']

scaler = StandardScaler()
df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
df_valid[numeric_features] = scaler.transform(df_valid[numeric_features])
df_test[numeric_features] = scaler.transform(df_test[numeric_features])


# 針對 建物現況格局-隔間 進行 target encoding
df_combined = pd.concat([df_train, df_valid], ignore_index=True)
df_combined['隔間_編碼'] = np.nan
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, val_index in kf.split(df_combined):
    df_train_fold = df_combined.iloc[train_index]
    df_val_fold = df_combined.iloc[val_index]
    target_mean = df_train_fold.groupby('建物現況格局-隔間')['總價元'].mean()
    df_combined.loc[val_index, '隔間_編碼'] = df_val_fold['建物現況格局-隔間'].map(target_mean)

overall_mean = df_combined['總價元'].mean()
df_combined['隔間_編碼'] = df_combined['隔間_編碼'].fillna(overall_mean)

scaler = StandardScaler()
df_combined['隔間_編碼'] = scaler.fit_transform(df_combined[['隔間_編碼']])

df_train = df_combined.iloc[:len(df_train)].reset_index(drop=True)
df_valid = df_combined.iloc[len(df_train):].reset_index(drop=True)

target_mean_full = df_train.groupby('建物現況格局-隔間')['總價元'].mean()
df_test['隔間_編碼'] = df_test['建物現況格局-隔間'].map(target_mean_full)
df_test['隔間_編碼'] = df_test['隔間_編碼'].fillna(overall_mean)
df_test['隔間_編碼'] = scaler.transform(df_test[['隔間_編碼']])

df_train = df_train.drop(['建物現況格局-隔間'], axis=1)
df_valid = df_valid.drop(['建物現況格局-隔間'], axis=1)
df_test = df_test.drop(['建物現況格局-隔間'], axis=1)


# 訓練模型
# 用XGBoost
X_train = df_train.drop(columns=['總價元'])  # 特徵欄位
y_train = df_train['總價元']                # 目標變數

X_valid = df_valid.drop(columns=['總價元'])  # 驗證特徵
y_valid = df_valid['總價元']                # 驗證目標變數


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(df_test)

params = {
    'objective': 'reg:absoluteerror',  # 使用 MAE 進行回歸
    'learning_rate': 0.1,
    'max_depth': 10,
    'eval_metric': 'mae',  
    'lambda': 1,   # L2 正則化
    'alpha': 1     # L1 正則化
}

evallist = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=2000, evals=evallist)
predictions = model.predict(dtest)

result_df = pd.DataFrame({
    '編號': df_test.index + 1,  
    '總價元': predictions
})

# 用LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression_l1', 
                              n_estimators=2000, learning_rate=0.1, max_depth=6)

model_lgb.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              eval_metric='mae') 

predictions_lgb = model_lgb.predict(df_test)



# 用CatBoost
model_cat = CatBoostRegressor(iterations=2000, depth=6, learning_rate=0.1, loss_function='MAE', verbose=10)
model_cat.fit(X_train, y_train, 
              eval_set=[(X_valid, y_valid)], 
              early_stopping_rounds=10)


predictions_cat = model_cat.predict(df_test)

# 使用集成學習，將三個模型的預測結果進行加權平均
xgb_weight = 0.8
lgb_weight = 0.1
cat_weight = 0.1

final_predictions = (xgb_weight * predictions +
                     lgb_weight * predictions_lgb +
                     cat_weight * predictions_cat)


result_df_combined = pd.DataFrame({
    '編號': df_test.index + 1,
    '總價元': final_predictions
})
result_df_combined.to_csv('house_price_predictions_combined.csv', index=False, encoding='utf-8-sig')

print("加權平均完成，結果已保存至 CSV。")



