# 挑異常值 抓到anom
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]

# 用nan將異常值取代
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 檢查貸款人車齡
plt.hist(app_train[~app_train.OWN_CAR_AGE.isnull()]['OWN_CAR_AGE'])
app_train[app_train['OWN_CAR_AGE'] > 50]['OWN_CAR_AGE'].value_counts()
＊上面是說先把own_car_age>50的挑起來 然後再帶回own_car_age數有多少


# 再把只有 2 值 (通常是 0,1) 的欄位去掉
numeric_columns = list(app_train[numeric_columns].columns[list(app_train[numeric_columns].apply(lambda x:len(x.unique())!=2 ))])


# 檢視這些欄位的數值範圍
for col in numeric_columns:
    app_train[col].plot.hist(title=col)
    plt.show()

# 離群值造成的直方圖看不清楚
# 選擇 OBS_60_CNT_SOCIAL_CIRCLE 小於 20 的資料點繪製

loc_a = app_train["OBS_60_CNT_SOCIAL_CIRCLE"]<20
loc_b = 'OBS_60_CNT_SOCIAL_CIRCLE'

app_train.loc[loc_a, loc_b].hist()
plt.show()

# 挑出非空值的row
app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY']

# 如果欄位中有 NA, describe 會有問題
app_train['AMT_ANNUITY'].describe()

# np.percentile()
a = np.array([[10, 7, 4], [3, 2, 1]])
np.percentile(a, 50) # 全部的50%分位數
np.percentile(a, 50, axis=0) # 縱列50% (10+3)/2=3.5
np.percentile(a, 50, axis=1) # 橫列50% 分別的50%分位數
np.percentile(a, 50, axis=1, keepdims=True) # 保持維度不變

* 後者為加上keepdims的結果	     array([ 7.,  2.]) vs array([[ 7.],[ 2.]])

# 計算四分位數
Ignore NA, 計算五值

five_num = [0, 25, 50, 75, 100]
quantile_5s = [np.percentile(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'], q = i) for i in five_num]
print(quantile_5s)

# 利用np get median
np.median(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])

# 計算眾數
from scipy.stats import mode
mode_get = mode(app_train[~app_train['AMT_ANNUITY'].isnull()]['AMT_ANNUITY'])

# 挑選空值用 中位數填補
app_train.loc[app_train['AMT_ANNUITY'].isnull(),'AMT_ANNUITY'] = q_50

# 數值的出現的次數
app_train['AMT_ANNUITY'].value_counts()
# 次數的index數值
value = list(app_train['AMT_ANNUITY'].value_counts().index)
value[0] ==> 基本上是眾數(可用這個眾數來填補)

# 沿縱軸合併
res = pd.concat([df1, df2, df3])

# 沿橫軸合併
result = pd.concat([df1, df4], axis = 1)
result
* 基本上這個方法 合併後其他的會留空
* 下面這個方法 可以用硬串接 基本上就是有值的才串接
result = pd.concat([df1, df4], axis = 1, join = 'inner') # 硬串接

# pd.merge() how=?
pd.merge(df1, df2, on='id', how='outer') 以id這欄做全合併
pd.merge(df1, df2, on='id', how='inner') 以id這欄做部分合併 (id match才做合併)

# 欄-列 逐一解開
df.melt()

# 篩選條件後,用loc
# 取 AMT_INCOME_TOTAL 大於平均資料中，SK_ID_CURR, TARGET 兩欄
sub_df = app_train.loc[app_train['AMT_INCOME_TOTAL'] > app_train['AMT_INCOME_TOTAL'].mean(), ['SK_ID_CURR', 'TARGET']]
sub_df.head()

# GroupBy用法
## groupby後看size
app_train.groupby(['NAME_CONTRACT_TYPE']).size()
## groupby後看他某一col的25%50%....etc
app_train.groupby(['NAME_CONTRACT_TYPE'])['AMT_INCOME_TOTAL'].describe()
## groupby後看他某col的mean
app_train.groupby(['NAME_CONTRACT_TYPE'])['TARGET'].mean()

# 教你怎麼取0:10000的指定欄位
app_train.loc[0:10000, ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']]

# cut()
##如果我們今天有一些連續性的數值，可以使用cut&qcut進行離散化 cut函数是利用數值區間將數值分類，qcut則是用分位數。換句話說，cut用在長度相等的類別，qcut用在大小相等的類別。
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = bin_cut)
## cut 可以用串列帶入,或是用np.linspace(start, stop, segment point)來製造區間

# random
## random seed
np.random.seed(1) 使得亂數可預測
## random int 
x = np.random.randint(0, 50, 1000) // 0 ~ 50 產生100個亂數
## 常態亂數
y = np.random.normal(0, 10, 1000)
## correlation x, y 相關係數
np.corrcoef(x, y)
## 畫散布圖
plt.scatter(x, y)

# corr做出來是陣列 怎麼辦
corr = np.corrcoef(sub_df['DAYS_EMPLOYED'] / (-365), sub_df['AMT_INCOME_TOTAL'])
print(corr)  -> print("Correlation: %.4f" % (corr[0][1])) // 只取小數點後四位，跟x,y的相關係數
x -> x		 x -> y
[[1.         0.01300472]
 [0.01300472 1.        ]]

# 如果直接畫散布圖 - 看不出任何趨勢或形態 ？
## 將y軸改成log-scale
np.log10(sub_df['AMT_INCOME_TOTAL'] )

# 看col類別
app_train[col].dtype

# 如果只有兩個值的類別欄位就做LE
## 種類 2 種以下的類別型欄位轉標籤編碼 (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

## 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # 紀錄有多少個 columns 被標籤編碼過
            le_count += 1
 

# 相關係數
app_train.corr()['TARGET'] # 列出target與所有欄位的相關係數
ext_data_corrs = ext_data.corr()  # 若沒有指定哪一個欄位 則是變成全部相比

# sort some series
corr_vs_target.sort_values(ascending=False)

# data := <class 'pandas.core.series.Series'>
## 找小15
data.head(15)
## 找大15
data.tail(15)

# boxplot
## {dataset}.boxplot(column="y", by="x")
app_train.boxplot(column="EXT_SOURCE_3", by="TARGET")
plt.show()

# matplotlib theme
plt.style.use(‘default’) # 不需設定就會使⽤用預設
plt.style.use('ggplot')
plt.style.use(‘seaborn’)# 或採⽤用 seaborn 套件繪圖

# plot懶人包
## 改變繪圖樣式 (style)
plt.style.use('ggplot') 
## 改變樣式後再繪圖一次, 比較效果
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins = 25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');
plt.show()
## 設定繪圖區域的長與寬
plt.figure(figsize = (10, 8))

# kde by sns
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Gaussian esti.', kernel='gau')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Cosine esti.', kernel='cos')
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'Triangular esti.', kernel='tri')
plt.show()

* app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365 這裡的意思是說 先loc出target=0的值對上DAYS_BIRTH再除以365

# 完整分布圖 (distplot) : 將 bar 與 Kde 同時呈現
sns.distplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
plt.legend() // 顯示圖例
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.show()

# EDA: 把連續型變數離散化
## 主要的⽅法:
等寬劃分：按照相同寬度將資料分成幾等份。缺點是受到異常值的影響比較⼤。
ex:
# 新增欄位 "equal_width_age", 對年齡做等寬劃分, 切成四等份
ages["equal_width_age"] = pd.cut(ages["age"], 4)

等頻劃分：將資料分成幾等份，每等份資料裡⾯的個數是一樣的。
ex:
# 新增欄位 "equal_freq_age", 對年齡做等頻劃分, 切成四等份
ages["equal_freq_age"] = pd.qcut(ages["age"], 4)

聚類劃分：使⽤用聚類演算法將資料聚成幾類，每⼀個類為⼀個劃分。

# sort_index
ex1:
ages["customized_age_grp"].value_counts().sort_values()
>>
(20, 30]     6
(50, 100]    3
(30, 50]     3
(10, 20]     2
(0, 10]      2
## 怎麼辦？？
ages["customized_age_grp"].value_counts().sort_index()


# subplot
plt.subplot(row,column,idx)
## plt.subplot 三碼如上所述, 分別表示 row總數, column總數, 本圖示第幾幅(idx)
plt.subplot(321)
plt.plot([0,1],[0,1], label = 'I am subplot1')
plt.legend()

# heatmap
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

# kde 潤飾
# 依不同 EXT_SOURCE 逐項繪製 KDE 圖形
for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
    # 做 subplot
    print(i, source) # 名稱加col_name
    plt.subplot(1, 3, i + 1)
    
    # KDE 圖形
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label = 'target == 0')
    sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label = 'target == 1')
    
    # 加上各式圖形標籤
    plt.title('Distribution of %s by Target Value' % source)
    plt.xlabel('%s' % source); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)

# tight_layout()
tight_layout() can take keyword arguments of pad, w_pad and h_pad. These control the extra padding around the figure border and between subplots. The pads are specified in fraction of fontsize.

# df.drop()
plot_data.drop(['DAYS_BIRTH'],axis=1, inplace=True)
* axis 一定要加 否則會報錯

# df.sample(n)
>>> df['num_legs'].sample(n=3, random_state=1) # 對df num_legs欄位抽三組

# df.dropna()
我想問的是如果下這個指令 如果一個欄位是空值就是全部刪掉嗎？


# 把 NaN 數值刪去, 並限制資料上限為 100000 : 因為要畫點圖, 如果點太多，會畫很久!
N_sample = 100000
plot_data = plot_data.dropna().sample(n = N_sample)

# N_sample = 100000
# 把 NaN 數值刪去, 並限制資料上限為 100000 : 因為要畫點圖, 如果點太多，會畫很久!
plot_data = plot_data.dropna().sample(n = N_sample))


# 建立 pairgrid 物件
grid = sns.PairGrid(data = plot_data, size = 3, diag_sharey=False,
                    hue = 'TARGET', vars = [x for x in list(plot_data.columns) if x != 'TARGET'])
## 上半部為 scatter
grid.map_upper(plt.scatter, alpha = 0.2)
## 對角線畫 histogram
grid.map_diag(sns.kdeplot)
## 下半部放 density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r)
plt.suptitle('Ext Source and Age Features Pairs Plot', size = 32, y = 1.05) # 大標
plt.show()

# np.random.random((row, col))
區間：[0.0, 1.0)
## 那如果要取 -1.0 ~ 1.0 呢?
可以利用簡單數學
2 * np.random.random() -1 

# 將train_data & test_data 欄位改成一致
##調整欄位數, 移除出現在 training data 而沒有出現 testing data 中的欄位
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# 填補器設定缺失值補中位數
imputer = Imputer(strategy = 'median')

# 縮放器設定特徵縮放到 0~1 區間
scaler = MinMaxScaler(feature_range = (0, 1))
#    x - min
# ------------
# max - min

# 填補器載入各欄中位數
imputer.fit(train)
# 將中位數回填 train, test 資料中的空缺值
train = imputer.transform(train)
test = imputer.transform(app_test)
# 縮放器載入 train 的上下限, 對 train, test 進行縮放轉換
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# df.to_csv()
submit.to_csv("submission.csv", sep='\t')

# 特徵工程
從事實到對應分數的轉換，我們稱為特徵⼯程

# np.log1p() 數據平滑處理
train_Y = np.log1p(df_train['SalePrice'])
## pred = np.expm1(pred) # log1p()的反函數


# 特徵工程簡化版
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for col in df.columns:
    if df[col] == 'object': # 如果是文字型 / 類別型欄位, 就先補缺 'None' 後, 再做標籤編碼
        df[col] = df[col].fillna('None')
        df[col] = LEncoder.fit_transform(df[col])
    else: # 其他狀況(本例其他都是數值), 就補缺 -1
        df[col] = df[col].fillna(-1)
    df[col] = MMEncoder.fit_transform(df[col].values.reshape(1, 1))

# 將ids跟pred合併 (為了產出csv)
sub = pd.DataFrame({'Id':ids, 'SalePrice': pred})
sub.to_csv('house_baseline.csv', index=False)

# how to do logistic regression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)

# 秀出資料欄位的類型, 與對應的數量
# df.dtypes : 轉成以欄位為 index, 類別(type)為 value 的 DataFrame
# .reset_index() : 預設是將原本的 index 轉成一個新的欄位, 如果不須保留 index, 則通常會寫成 .reset_index(drop=True)
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"] # 給col名稱（最上面）
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index() # groupby後是一個物件,需要用一個aggregate方法來聚集總數,在重置索引值
dtype_df

# unique vs nunique
df[int_features].unique()
>>df[int_features].unique()
好像有一種unique方法是會返回唯一值
df[int_features].nunique()

# 檢查是否缺值
## 檢查欄位缺值數量 (去掉.head()可以顯示全部)
df.isnull().sum() // .sort_values(ascending=False).head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]

# df補空值
df = df.fillna(df.mean())

# 做線性迴歸
estimator = LinearRegression() // estimator = LogisticRegression() 改作羅吉斯回歸
cross_val_score(estimator, train_X, train_Y, cv=5).mean() # cv=k-fold

# df做最大最小化
df_temp = MinMaxScaler().fit_transform(df)

# df搭配標準化
df_temp = StandardScaler().fit_transform(df)

# 散佈圖 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=df['GrLiveArea'], y=train_Y)
plt.show()

# 對離群值做壞壞的事
## 將 GrLivArea 限制在 800 到 2500 以內, 調整離群值
df['GrLivArea'] = df['GrLivArea'].clip(800, 2500)
## 將 GrLivArea 限制在 800 到 2500 以內, 捨棄離群值
keep_indexs = (df['GrLivArea']> 800) & (df['GrLivArea']< 2500) # 這裡一定要用括號
df = df[keep_indexs]
train_Y = train_Y[keep_indexs]

# 對數去偏
對數去偏就是使用自然對數去除偏態，常見於計數／價格這類非負且可能為0的欄位
因為需要將0對應到0,所以先加一在取對數
還原時使用expm1也就是先取指數後再減一

# 方根去偏
就是將數值減去最小值後開根號,最大值有限時使用(例：成績轉換)

# 分布去偏 (boxbox)
函數的 lambda(λ) 參數為 0 時等於 log 函數，lambda(λ) 為 0.5 時等於開根號 (即sqrt)，因此可藉由參數的調整更靈活地轉換數值，但要特別注意Y的輸入數值必須要為正 (不可為0)

# 直方圖(含kde)
sns.distplot(df['LotArea'][:train_num])
plt.show()

# 標籤編碼
df_temp = pd.DataFrame() # initialize
for c in df.columns: # 每一行都要變
    df_temp[c] = LabelEncoder().fit_transform(df[c])
-----------------------------------------------------
# 獨熱編碼
df_temp = pd.get_dummies(df)

train_X = df_temp[:train_num]

# 看時間
print(f'shape : {train_X.shape}')
print(f'score : {cross_val_score(estimator, train_X, train_Y, cv=5).mean()}')
print(f'time : {time.time() - start} sec')


# 標籤編碼 vs 獨熱編碼 in 線性迴歸 & 梯度提升樹
線性迴歸時, 獨熱編碼不僅時間花費很多, 準確率也大幅下降
梯度提升樹時, 分數有小幅提升, 執行時間則約為兩倍
可見獨熱不僅計算時間會增加不少, 也不太適合用在線性迴歸上
##使用時機
綜合建議非深度學習時，類別型特徵建議預設採標籤編碼;深度學習時，預設採獨熱編碼因非深度學習時主要是樹狀模型 (隨機森林 / 梯度提升樹等基於決策樹的模型)，⽤兩次門檻就能分隔關鍵類別;但深度學習主要依賴倒傳遞，標籤編碼會不易收斂
* 當特徵重要性⾼，且可能值較少時，才應該考慮獨熱編碼

# ========類別型特徵預設編碼方式=======
## 均值編碼 (Mean Encoding)  : 使⽤目標值的平均值，取代原本的類別型特徵
###上面的問題
如果交易樣本非常少, 且剛好抽到極端值, 平均結果可能會有誤差很⼤ => 平滑化 ( Smoothing )
均值編碼平滑化公式:
新類別均值 = (原類別平均*類別樣本數+全部的總平均*調整因子)/類別樣本數+調整因子

* 調整因⼦子⽤用來來調整平滑化的程度，依總樣本數調整
小提醒：均值編碼容易overfitting(可利用cross_val_score確認前後分數 來驗證是否合適)

# 均值編碼範例
# 均值編碼 + 線性迴歸
data = pd.concat([df[:train_num], train_Y], axis=1)
for c in df.columns:
    mean_df = data.groupby([c])['SalePrice'].mean().reset_index()
    mean_df.columns = [c, f'{c}\_mean']
    data = pd.merge(data, mean_df, on=c, how='left')
    data = data.drop([c] , axis=1)
print(mean_df)
data = data.drop(['SalePrice'] , axis=1)
estimator = LinearRegression()

# 計數編碼
計數編碼是計算類別在資料中的出現次數，當⽬標平均值與類別筆數呈正/負相關時，可以考慮使用當相異類數量相當⼤時，其他編碼⽅式效果更差，可以考慮雜湊編碼以節省時間
註 : 雜湊編碼效果也不佳，這類問題更好的解法是嵌入式編碼(Embedding)，但是需要深度學習並有其前提，因此這裡暫時不排入課程

# 加上 'Ticket' 欄位的計數編碼
# 第一行 : df.groupby(['Ticket']) 會輸出 df 以 'Ticket' 群聚後的結果, 但因為群聚一類只會有一個值, 因此必須要定義運算
# 例如 df.groupby(['Ticket']).size(), 但欄位名稱會變成 size, 要取別名就需要用語法 df.groupby(['Ticket']).agg({'Ticket_Count':'size'})
# 這樣出來的計數欄位名稱會叫做 'Ticket_Count', 因為這樣群聚起來的 'Ticket' 是 index, 所以需要 reset_index() 轉成一欄
# 因此第一行的欄位, 在第三行按照 'Ticket_Count' 排序後, 最後的 DataFrame 輸出如 Out[5]
count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# count_df = df.groupby(['Ticket'])['Name'].agg({'Ticket_Count':'size'}).reset_index()
# # 但是上面資料表結果只是 'Ticket' 名稱對應的次數, 要做計數編碼還需要第二行 : 將上表結果與原表格 merge, 合併於 'Ticket' 欄位
# # 使用 how='left' 是完全保留原資料表的所有 index 與順序
df = pd.merge(df, count_df, on=['Ticket'], how='left')
count_df.sort_values(by=['Ticket_Count'], ascending=False).head(10)

## 特徵雜湊
# 這邊的雜湊編碼, 是直接將 'Ticket' 的名稱放入雜湊函數的輸出數值, 為了要確定是緊密(dense)特徵, 因此除以10後看餘數
# 這邊的 10 是隨機選擇, 不一定要用 10
df_temp['Ticket_Hash'] = df['Ticket'].map(lambda x:hash(x) % 10)

# 跟雜湊差這行程式碼
df_temp['Ticket_Count'] = df['Ticket_Count'] // 與上述比較

# 找none replace 0
df_train['Resolution'].str.replace('NONE', '0')

# 時間型特徵
最常用的是特徵分解-拆解成年/月／日/時/分/秒的分類值
週期循環特徵是將時間"循環"特性改成特徵⽅式, 設計關鍵在於⾸尾相接, 因此我們需要使用 sin /cos 等週期函數轉換
常見的週期循環特徵有 - 年週期(季節) / 周期(例假日) / 日週期(日夜與生活作息), 要注意的是最⾼與最低點的設置
## 使用方法：
1.大概就是先用apply對欄位中每一個值先做parse...
df['pickup_datetime'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC')
這時候得到的會是string -> datetime format
2.這時候可以用上面得到的東西,分別get parse後的結果
df['pickup_year'] = df['pickup_datetime'].apply(lambda x: datetime.datetime.strftime(x, '%Y')).astype('int64')

# 群聚編碼(Group by Encoding)
1.類似均值編碼的概念，可以取類別平均值 (Mean) 取代險種作為編碼但因為比較像性質描寫，因此還可以取其他統計值，如中位數 (Median)，眾數(Mode)，最大值(Max)，最⼩值(Min)，次數(Count)...等
2.數值型特徵對文字型特徵最重要的特徵組合方式 常見的有 mean, median, mode, max, min, count 等
## 所以什麼時候需要群聚編碼呢？
->與數值特徵組合相同的時候
先以領域知識或特徵重要性挑選強⼒特徵後, 再將特徵組成更強的特徵兩個特徵都是數值就⽤特徵組合, 其中之⼀是類別型就用聚類編碼

# 群聚編碼
# 生活總面積(GrLivArea)對販售條件(SaleCondition)做群聚編碼
# 寫法類似均值編碼, 只是對另一個特徵, 而非目標值
df['SaleCondition'] = df['SaleCondition'].fillna('None')
mean_df = df.groupby(['SaleCondition'])['GrLivArea'].mean().reset_index()
mode_df = df.groupby(['SaleCondition'])['GrLivArea'].apply(lambda x: x.mode()[0]).reset_index()
median_df = df.groupby(['SaleCondition'])['GrLivArea'].median().reset_index()
max_df = df.groupby(['SaleCondition'])['GrLivArea'].max().reset_index()
## 下行是用mean_df&mode_df 先合併而且是用pd.merge
temp = pd.merge(mean_df, mode_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, median_df, how='left', on=['SaleCondition'])
temp = pd.merge(temp, max_df, how='left', on=['SaleCondition'])
temp.columns = ['SaleCondition', 'Area_Sale_Mean', 'Area_Sale_Mode', 'Area_Sale_Median', 'Area_Sale_Max']
temp
## 繼續跟主df合併
df = pd.merge(df, temp, how='left', on=['SaleCondition'])
df = df.drop(['SaleCondition'] , axis=1)
df.head()

=== 下面可以過濾一些string ===
# 只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()

# 特徵選擇概念
特徵需要適當的增加與減少
增加特徵：特徵組合，群聚編碼
減少特徵：特徵選擇
## 特徵選擇的三個方法
過濾法 (Filter) : 選定統計數值與設定⾨門檻，刪除低於⾨門檻的特徵 ex:相關係數過濾法
包裝法 (Wrapper) : 根據⽬目標函數，逐步加入特徵或刪除特徵
嵌入法 (Embedded) : 使⽤用機器學習模型，根據擬合後的係數，刪除係數低於⾨檻的特徵本⽇內容將會介紹三種較常⽤的特徵選擇法 ex:L1(Lasso)嵌入法，GDBT(梯度提升樹)嵌入法

# 相關係數過濾法
power by heatmap..
找到⽬標值 (房價預估目標為SalePrice)之後，觀察其他特徵與⽬標值相關係數
預設顏⾊越紅表⽰越正相關，越藍負相關因此要刪除紅框中顏色較淺的特徵 : 訂出相關係數門檻值，特徵相關係數絕對值低於門檻者刪除SalePrice

# Lasso(L1) 嵌入法
使⽤Lasso Regression 時，調整不同的正規化程度，就會⾃然使得⼀部分的特徵係數為０，因此刪除的是係數為０的特徵，不須額外指定⾨門檻，但需調整正規化程度

# GDBT梯度提升樹 嵌入法
使用梯度提升樹擬合後，以特徵在節點出現的頻率當作特徵重要性，以此刪除重要性低於⾨檻的特徵，這種作法也稱為 GDBT 嵌入法由於特徵重要性不只可以刪除特徵，也是增加特徵的關鍵參考

# 相關係數法實作
## 計算df整體相關係數, 並繪製成熱圖 計算整體的 只會留數值欄位
import seaborn as sns
import matplotlib.pyplot as plt
corr = df.corr()
print(corr)
sns.heatmap(corr)
plt.show()
## 篩選相關係數大於 0.1 或小於 -0.1 的特徵 (要有.index 否則會挑不到)
high_list = list(corr[(corr['SalePrice']>0.1) | (corr['SalePrice']<-0.1)].index)
print(high_list)
------ 要記得pop target !!!

# Lasso(L1) 實作
## step 1
from sklearn.linear_model import Lasso
L1_Reg = Lasso(alpha=0.001)
train_X = MMEncoder.fit_transform(df)
L1_Reg.fit(train_X, train_Y)
L1_Reg.coef_
## step 2
from itertools import compress
L1_mask = list((L1_Reg.coef_>0) | (L1_Reg.coef_<0))
L1_list = list(compress(list(df), list(L1_mask)))
L1_list
## step 3
### L1_Embedding 特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df[L1_list])
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 這邊筆記一下compress的用法
compress

compress 的使用形式如下：

compress(data, selectors)
compress 可用於對數據進行篩選，當 selectors 的某個元素為 true 時，則保留 data 對應位置的元素，否則去除：
>>> from itertools import compress
>>>
>>> list(compress('ABCDEF', [1, 1, 0, 1, 0, 1]))
['A', 'B', 'D', 'F']
>>> list(compress('ABCDEF', [1, 1, 0, 1]))
['A', 'B', 'D']
>>> list(compress('ABCDEF', [True, False, True]))
['A', 'C']

＊我覺得有點像是快速做遮罩的概念

# 特徵的重要性
## 用決策樹來說明
1. 特徵重要性預設⽅式是取特徵決定分支的次數
2. 但分⽀次數以外，還有兩種更直覺的特徵重要性 : 特徵覆蓋度、損失函數降低量本例的特徵覆蓋度(假定八個結果樣本數量量⼀樣多)

* sklearn 當中的樹狀模型，都有特徵重要性這項⽅方法     (.feature_importances_)，⽽而實際上都是分⽀次數


# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()

# 找出你的重要性(by randomForest)
# 隨機森林擬合後, 將結果依照重要性由高到低排序
estimator = RandomForestRegressor()
estimator.fit(df.values, train_Y)
# estimator.feature_importances_ 就是模型的特徵重要性, 這邊先與欄位名稱結合起來, 才能看到重要性與欄位名稱的對照表
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
feats >>> 再利用feat去挑出前50% 的index

## 製作四特徵 : 加, 乘, 互除(分母加1避免除0) 看效果 (Note: 數值原本已經最大最小化介於 [0,1] 區間, 這四種新特徵也會落在 [0,1] 區間)
df['Add_char'] = (df['GrLivArea'] + df['OverallQual']) / 2
df['Multi_char'] = df['GrLivArea'] * df['OverallQual']
df['GO_div1p'] = df['GrLivArea'] / (df['OverallQual']+1) * 2
df['OG_div1p'] = df['OverallQual'] / (df['GrLivArea']+1) * 2
train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()

# 分類預測的集成
已知來自法國的旅客⽣生存機率是 0.8，且年齡 40 到 50 區間的生存機率也是 0.8那麼同時符合兩種條件的旅客，⽣存機率應該是多少呢?
解法 : 邏輯斯迴歸(logistic regression)與其重組我們可以將邏輯斯迴歸理解成「線性迴歸 + Sigmoid 函數」⽽ sigmoid 函數理解成「成功可能性與機率的互換」這裡的成功可能性正表示更可能，負表⽰較不可能

# 葉編碼原理
樹狀模型作出預測時,模型預測時就會將資料分成好幾個區塊,也就是決策樹的葉點,每個葉點資料性質接近,可視為資料的一種分組
雖然不適合直接沿用樹狀模型機率，但分組⽅式有代表性，因此按照葉點將資料離散化，比之前提過的離散化⽅式更更精確，這樣的編碼我們就稱為葉編碼的結果，是⼀組模型產⽣的新特徵，我們可以使用邏輯斯回歸，重新賦予機率 (如下葉圖)，也可以與其他算法結合 (例例如 : 分解機Factorization Machine )使資料獲得新⽣
## 目的
葉編碼的⽬的是重新標記資料，以擬合後的樹狀狀模型分歧條件，將資料離散化，這樣比⼈為寫作的判斷條件更精準，更符合資料的分布情形

step:
每棵樹視為一個新特徵(葉點就是特徵有幾個值)
每個新特徵均為分類型特徵,決策樹的葉點與該特徵一一對應
最後再以邏輯斯回歸合併

# 有點不太明白 先記錄
# 因為訓練邏輯斯迴歸時也要資料, 因此將訓練及切成三部分 train / val / test, 採用 test 驗證而非 k-fold 交叉驗證
# train 用來訓練梯度提升樹, val 用來訓練邏輯斯迴歸, test 驗證效果
train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.5)
train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.5)

# 梯度提升樹調整參數並擬合後, 再將葉編碼 (＊.apply) 結果做獨熱 / 邏輯斯迴歸
# 調整參數的方式採用 RandomSearchCV 或 GridSearchCV, 以後的進度會再教給大家, 本次先直接使用調參結果
gdbt = GradientBoostingClassifier(subsample=0.93, n_estimators=320, min_samples_split=0.1, min_samples_leaf=0.3, 
                                  max_features=4, max_depth=4, learning_rate=0.16)
onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
gdbt.fit(train_X, train_Y)
onehot.fit(gdbt.apply(train_X)[:, :, 0])
lr.fit(onehot.transform(gdbt.apply(val_X)[:, :, 0]), val_Y)
# 將梯度提升樹+葉編碼+邏輯斯迴歸結果輸出
pred_gdbt_lr = lr.predict_proba(onehot.transform(gdbt.apply(test_X)[:, :, 0]))[:, 1]
fpr_gdbt_lr, tpr_gdbt_lr, _ = roc_curve(test_Y, pred_gdbt_lr)
# 將梯度提升樹結果輸出
pred_gdbt = gdbt.predict_proba(test_X)[:, 1]
fpr_gdbt, tpr_gdbt, _ = roc_curve(test_Y, pred_gdbt)
# 畫roc_curve 
plt.plot([0, 1], [0, 1], 'k--') # 不太懂
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lr, tpr_rf_lr, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# 如何解決過擬合或欠擬合
##過擬合
•增加資料量•降低模型複雜度•使用正規化 (Regularization)
##⽋擬合
•增加模型複雜度•減輕或不使⽤正規化

# train_test_split 函數切分(train, test row數一樣)
ex:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# K-fold Cross-validation 切分資料
kf = KFold(n_splits=5) ＃最少要兩切 預設為３ 這是建立一個物件
i = 0
for train_index, test_index in kf.split(X):
    i +=1 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("FOLD {}: ".format(i))
    print("X_test: ", X_test)
    print("Y_test: ", y_test)
    print("-"＊30)

# np.where用法
np.where(y==1)[0] #ｙ是一個array裡的數字 出來的是索引
np.where(x > 0.5, 1, 0)

# 回歸 vs. 分類
機器學習的監督式學習中主要分為回歸問題與分類問題。
回歸代表預測的目標值為實數 (-∞⾄至∞) -> 回歸問題是可以轉化為分類問題
分類代表預測的目標值為類別 (0 或 1)

# 二元分類 (binary-class) vs. 多元分類 (Multi-class)
二元分類，顧名思義就是⽬標的類別僅有兩個。像是詐騙分析 (詐騙⽤戶 vs. 正常⽤戶)、瑕疵偵測 (瑕疵 vs. 正常)多元分類則是⽬標類別有兩種以上。
如⼿寫數字辨識有 10 個類別(0~9),影像競賽 ImageNet 更是有⾼達 1,000 個類別需要分類

# Multi-class vs. Multi-label
當每個樣本都只能歸在⼀個類別，我們稱之為多分類 (Multi-class) 問題；⽽一個樣本如果可以同時有多個類別，則稱為多標籤 (Multi-label)。了解專案的⽬標是甚麼樣的分類問題並選⽤適當的模型訓練。

# from sklearn import datasets
X, y = datasets.make_regression(n_features=1, random_state=42, noise=4) # 生成資料集
model = LinearRrgression()
model.fit(X, y)
prediction = model.predict(X)

# 評估方法
mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估
auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。 **這邊特別注意 y_pred 必須要放機率值進去!**

# 資料二元化後評估
f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
##資料二元化
threshold = 0.5 
y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0

# np.newaxis()
X =  np.array([[1,2,3],[4,5,6], [7,8,9]])
print(X[:, np.newaxis, 2])
>> 3
6
9

# 用sklearn建立線性迴歸模型
## 建立一個線性回歸模型
regr = linear_model.LinearRegression()
## 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)
## 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)
## 可以看回歸模型的參數值
print('Coefficients: ', regr.coef_)
## 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
## 畫出回歸模型與實際資料的分佈
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

# 用sklearn建立羅吉斯迴歸模型
## 讀取鳶尾花資料集
iris = datasets.load_iris()
## 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=4)
## 建立模型
logreg = linear_model.LogisticRegression()
## 訓練模型
logreg.fit(x_train, y_train)
## 預測測試集
y_pred = logreg.predict(x_test)
## 精準度
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# 機器學習模型中的⽬標函數
•損失函數 (Loss function)損失函數衡量預測值與實際值的差異異，讓模型能往正確的⽅向學習
•正則化 (Regularization)正則化則是避免模型變得過於複雜，造成過擬合 (Over-fitting)
前面使用的損失函數是MSE,MAE
## 為了避免 Over-fitting，我們可以把正則化加入⽬標函數中，此時目標函數 = 損失函數 + 正則化
## 因為正則化可以懲罰模型的複雜度，當模型越複雜時其值就會越⼤

# 正則化
正則化函數是⽤來衡量模型的複雜度
有 L1 與 L2 兩種函數
L1：αΣ|weights|    # Lasso = Linear Regression 加上 L1 （可以把某些特徵變為０達到特徵篩選）
L2：αΣ(weights)^2  # Ridge = Linear Regression 加上 L2 （可以處理共線性,解決高度相關的原因是，能夠縮減 X 的高相關特徵)
其中有個超參數α可以調整正則化的強度，LASSO 與 Ridge 就是回歸模型加上不同的正則化函數
這兩種都是希望模型的參數值不要太⼤，原因是參數的數值變⼩，噪音對最終輸出的結果影響越小，提升模型的泛化能力，但也讓模型的擬合能⼒下降

# how to lasso ＆ ridge
## 建模的時候
lasso = linear_model.Lasso(alpha=1.0) // 不用加LinearRegression
ridge = linear_model.Ridge(alpha=1.0)

# 決策樹 (Decision Tree)
從訓練資料中找出規則，讓每⼀次決策能使訊息增益 (Information Gain) 最大化訊息
增益越⼤代表切分後的兩群資料，群內相似程度越⾼
## 建模
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=0, min_samples_split=2, min_samples_leaf=1)
Criterion: 衡量資料相似程度的 metric
Max_depth: 樹能⽣長的最深限制
Min_samples_split: ⾄少要多少樣本以上才進⾏切分
Min_samples_leaf: 最終的葉⼦ (節點) 上⾄少要有多少樣本
clf.features_importances_ 

# 訊息增益
決策樹模型會用 features 切分資料，該選用哪個 feature 來切分則是由訊息增益的⼤大⼩小決定的。希望切分後的資料相似程度很⾼，通常使⽤吉尼係數來來衡量相似程度。
## 決策樹的特徵重要性（Feature importance)
1.我們可以從構建樹的過程中，透過 feature 被⽤來切分的次數，來得知哪些features 是相對有用的
2.所有 feature importance 的總和為 1
3.實務上可以使⽤ feature importance 來了解模型如何進行分類
## how to get feature importance
print(iris.feature_names)
## feature importance numeric
print("Feature importance: ", clf.feature_importances_)
## classfier or regressor
There is a huge difference between classifiers and regressors. Classifiers predict one class from a predetermined list or probabilities of belonging to a class. Regressors predict some value, which could be almost anything.
Differeng metrics are used for classification and regression. 
So it isn't a good idea to use classifier for regression problem and vice versa.

# Gini vs Entropy
Gini impurity and Information Gain Entropy are pretty much the same. And people do use the values interchangeably. Below are the formulae of both:
ref :: https://datascience.stackexchange.com/questions/10228/when-should-i-use-gini-impurity-as-opposed-to-information-gain

# sklearn 建立決策樹模型
根據回歸/分類問題分別建立不同的 Classifier
from sklearn.tree_model import DecisionTreeRegressor
from sklearn.tree_model import DecisionTreeClassifier
clf = DecisionTreeClassifier()

## 建立模型四步驟

在 Scikit-learn 中，建立一個機器學習的模型其實非常簡單，流程大略是以下四個步驟

1. 讀進資料，並檢查資料的 shape (有多少 samples (rows), 多少 features (columns)，label 的型態是什麼？)
    - 讀取資料的方法：
        - **使用 pandas 讀取 .csv 檔：**pd.read_csv
        - **使用 numpy 讀取 .txt 檔：**np.loadtxt 
        - **使用 Scikit-learn 內建的資料集：**sklearn.datasets.load_xxx
    - **檢查資料數量：**data.shape (data should be np.array or dataframe)
2. 將資料切為訓練 (train) / 測試 (test)
    - train_test_split(data)
3. 建立模型，將資料 fit 進模型開始訓練
    - clf = DecisionTreeClassifier()
    - clf.fit(x_train, y_train)
4. 將測試資料 (features) 放進訓練好的模型中，得到 prediction，與測試資料的 label (y_test) 做評估
    - clf.predict(x_test)
    - accuracy_score(y_test, y_pred)
    - f1_score(y_test, y_pred)

# DT vs. CART
原本 DT 是根據 information gain(IG) 來決定要怎麼切割
CART 是找個 impurity function(IF) 來決定要怎麼切割

# 決策樹的缺點
- 若不對決策樹進行限制 (樹深度、葉⼦上至少要有多少樣本等)，決策樹非常容易Over-fitting 
- 為了解決策樹的缺點，後續發展出了隨機森林的概念，以決策樹為基底延伸出的模型

# 集成模型 - 隨機森林 (Random Forest)
集成 (Ensemble) 是將多個模型的結果組合在⼀起，透過投票或是加權的⽅方式得到最終結果

# Where is random?
- 決策樹⽣成時，是考慮所有資料與特徵來做切分的
- ⽽隨機森林的每⼀棵樹在⽣成過程中，都是隨機使用⼀部份的訓練資料與特徵代表每棵樹都是⽤隨機的資料訓練⽽成的

# Random Forest
在 training data 中, 從中取出一些 feature & 部份 data 產生出 Tree (通常是CART)
並且重複這步驟多次, 會產生出多棵 Tree 來
最後利用 Ensemble (Majority Vote) 的方法, 結合所有 Tree, 就完成了 Random Forest

1. 準備 training data 
## Bootstrap
為了讓每棵有所不同, 主要就是 training data 的採樣結果都會不太一樣
## Bagging
一種採樣方式, 假設全體 training data 有N筆, 你要採集部分資料, 
但是又不想要採集到全體的資料 (那就不叫採集了), 要如何做?
一般常見的方式為: 從 N 筆 data 挑資料, 一次挑一筆, 挑出的會再放回去, 最後計算的時候重複的會不算(with replacement), 假設最後為y, N > y

因為是用 bagging on data, 所以每棵 Tree 在建立的時候, 都會是用不一樣的 data 去建立的
- ** Random Forest 所建立的每棵 Tree, 在 data 跟 feature 上, 都有一定程度上的不同
- ** 設定最少要 bagging 出 (k / 2) + 1 的 feature, 才比較有顯著結果, K 為原本的 feature 數量,或者另外一個常見設定是 square(k)

2. Build Tree
這邊, 就沒什麼好說的了, 只要將前述的 data & feature 準備好, 餵入 CART 就可以了
唯一要注意的事情, Random Forest 不須要讓任何的 Tree 做 prune

3. Ensemble
簡單來說就是合體
給我一筆 data, 我會讓這 50 棵 Tree 分別去預估可能的 class, 最後全體投票, 多數決決定
如果今天是用 Regression 的 RF, 則是加總起來除以總棵數, 就是預估的答案

4. Out Of Bag (bagging沒用到的data)
衡量可能的錯誤率
** 因為重複採樣的關係, 平均來講, 每棵大約會有 1/3 training data 採樣不到
所以收集這些 data, 最後等到 Forest 建立完成之後, 將這些 data 餵進去判斷, 最後得出錯誤率
這方式稱為 Out-Of-Bag (OOB)

## Notes
1. 若隨機森林中樹的數量太少，造成嚴重的Overfit，是有可能會比決策樹差。但如果都是⽤預設的參數，實務上不太會有隨機森林比決策樹差的情形，要特別注意程式碼是否有誤
2. 隨機森林中的每一棵樹，是希望能夠沒有任何限制，讓樹可以持續生長 (讓樹生成很深，讓模型變得複雜) 不要過度生長，避免 Overfitting
隨機森林: 希望每棵樹都能夠盡量複雜，然後再通過投票的方式，處理過擬合的問題。因此希望每棵樹都能夠盡量的生長
0.632 bootstrap: 這是傳統的統計問題，採用取後放回的方式，抽取與資料量同樣大小的 N 筆資料，約會使用 63.2 % 的原生資料。

# Random Forest 建模
from sklearn.ensemble import RandomForestClassifier // 代表隨機森林是個集成模型
## 讀取鳶尾花資料集
iris = datasets.load_iris()
## 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)
## 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=20, max_depth=4)
## 訓練模型
clf.fit(x_train, y_train)
## 預測測試集
y_pred = clf.predict(x_test)

# note:
同樣是樹的模型，所以像是 max_depth, min_samples_split 都與決策樹相同可決定要⽣成數的數量，越多越不容易過擬和，但是運算時間會變長
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
n_estimators=10, #決策樹的數量
criterion="gini",
max_features="auto", #如何選取 features         
max_depth=10,
min_samples_split=2,
min_samples_leaf=1)

# 梯度提升機 Gradient Boosting Machine
隨機森林使⽤的集成⽅法稱為 Bagging (Bootstrap aggregating)，用抽樣的資料與 features ⽣成每⼀棵樹，最後再取平均
Boosting 則是另一種集成方法，希望能夠由後⾯生成的樹，來修正前⾯樹學不好的地方要怎麼修正前面學錯的地⽅方呢？計算 Gradient!
每次生成樹都是要修正前⾯樹預測的錯誤，並乘上 learning rate 讓後面的樹能有更多學習的空間

## Bagging vs. Boosting
Bagging 是透過抽樣 (sampling) 的⽅式來生成每⼀棵樹，樹與樹之間是獨立生成的
Boosting 是透過序列 (additive)的⽅式來生成每一顆樹，每棵樹都會與前⾯的樹關聯，因為後⾯的樹要能夠修正

# 使用Sklearn中的梯度提升機
梯度提升機同樣是個集成模型，透過多棵決策樹依序⽣生成來來得到結果，緩解原本決策樹容易易過擬和的問題，實務上的結果通常也會比決策樹來來得好
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingClassifier()

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(
loss="deviance", #Loss 的選擇，若若改為 exponential 則會變成Adaboosting 演算法，概念相同但實作稍微不同
learning_rate=0.1, #每棵樹對最終結果的影響，應與 n_estimators 成反比
n_estimators=100 #決策樹的數量量)

# Random Forest vs. Gradient boosting
決策樹計算特徵重要性的概念是，觀察某⼀特徵被⽤來切分的次數⽽定。
假設有兩個⼀模一樣的特徵，在隨機森林中每棵樹皆為獨立，因此兩個特徵皆有可能被使用，最終統計出來的次數會被均分。
在梯度提升機中，每棵樹皆有關連，因此模型僅會使⽤其中⼀個特徵，另⼀個相同特徵的重要性則會消失

# coding
clf = GradientBoostingClassifier()

# 超參數調整
之前接觸到的所有模型都有超參數需要設置
•LASSO，Ridge: α的⼤⼩
•決策樹：樹的深度、節點最⼩樣本數
•隨機森林：樹的數量
這些超參數都會影響模型訓練的結果，建議先使用預設值，再慢慢進⾏調整超參數會影響結果，
但提升的效果有限，資料清理與特徵工程才能最有效的提升準確率，調整參數只是⼀個加分的⼯具。
## how to 調整
窮舉法 (Grid Search)：直接指定超參數的組合範圍，每⼀組參數都訓練完成，再根據驗證集 (validation) 的結果選擇最佳數
隨機搜尋 (Random Search)：指定超參數的範圍，⽤均勻分布進⾏參數抽樣，用抽到的參數進行訓練，再根據驗證集的結果選擇最佳參數，隨機搜尋通常都能獲得更佳的結果
## step by step
若持續使⽤同⼀份驗證集 (validation) 來調參，可能讓模型的參數過於擬合該驗證集，正確的步驟是使用 Cross-validation確保模型泛化性
1. 先將資料切分為訓練/測試集，測試集保留不使⽤
2. 將剛切分好的訓練集，再使⽤Cross-validation 切分 K 份訓練/驗證集
3. 用 grid/random search 的超參數進行訓練與評估
4. 選出最佳的參數，⽤該參數與全部訓練集建模
5. 最後使用測試集評估結果
** 超參數調整對最終結果影響很⼤嗎？
超參數調整通常都是機器學習專案的最後步驟，因為這對於最終的結果影響不會太多，多半是近⼀步提升 3-5 % 的準確率，但是好的特徵工程與資料清理是能夠一口氣提升 10-20 ％的準確率！

## coding
# 設定要訓練的超參數組合
n_estimators = [50, 100, 150]
max_depth = [1, 3, 5]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
# 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
grid_search = GridSearchCV(reg, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1) # reg是上面建立的模型
# 開始搜尋最佳參數
grid_result = grid_search.fit(x_train, y_train)
# 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型 

# 印出最佳結果與最佳參數
print("Best Accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))