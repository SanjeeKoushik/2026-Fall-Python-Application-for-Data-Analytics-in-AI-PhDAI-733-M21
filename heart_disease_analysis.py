# Run in Google Colab: upload heart.csv first (from google.colab import files; files.upload())
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
df = pd.read_csv('heart.csv', encoding='utf-8-sig')
print(df.shape); print(df.isna().sum().sum(), 'missing'); print('dups', df.duplicated().sum())
print(df.dtypes.value_counts()); print(df.target.value_counts(), df.target.value_counts(normalize=True))
print(df.describe().T[['mean','min','max']])
df = df.drop_duplicates().reset_index(drop=True); print('after', df.shape)
print(df.target.value_counts())
print(df.corr()['target'].sort_values())
print(df.groupby('sex').target.mean())
sns.set_style('whitegrid')
fig,ax=plt.subplots(1,2,figsize=(10,3.6))
sns.countplot(data=df,x='target',ax=ax[0],palette=['#4C78A8','#E45756']); ax[0].set_title('Target distribution (0 = no disease, 1 = disease)')
sns.heatmap(df.corr(),cmap='coolwarm',center=0,ax=ax[1],cbar=False); ax[1].set_title('Correlation heatmap')
plt.tight_layout(); plt.savefig('eda1.png',dpi=160); plt.close()
fig,ax=plt.subplots(1,3,figsize=(11,3.2))
for a,c in zip(ax,['age','thalach','oldpeak']):
    sns.boxplot(data=df,x='target',y=c,ax=a,palette=['#4C78A8','#E45756']); a.set_title(c+' by target')
plt.tight_layout(); plt.savefig('eda2.png',dpi=160); plt.close()
X=df.drop(columns='target'); y=df.target
num=['age','trestbps','chol','thalach','oldpeak']; cat=['sex','cp','fbs','restecg','exang','slope','ca','thal']
pre=ColumnTransformer([('n',StandardScaler(),num),('c',OneHotEncoder(handle_unknown='ignore'),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=42)
print(len(Xtr),len(Xte))
res={}
models={'Logistic Regression':LogisticRegression(max_iter=1000),'Random Forest':RandomForestClassifier(n_estimators=300,random_state=42)}
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
for a,(n,m) in zip(ax,models.items()):
    p=Pipeline([('pre',pre),('clf',m)])
    cv=cross_val_score(p,Xtr,ytr,cv=5,scoring='roc_auc').mean()
    p.fit(Xtr,ytr); pr=p.predict(Xte); pb=p.predict_proba(Xte)[:,1]
    res[n]=[accuracy_score(yte,pr),precision_score(yte,pr),recall_score(yte,pr),f1_score(yte,pr),roc_auc_score(yte,pb),cv]
    cm=confusion_matrix(yte,pr); print(n,cm.tolist())
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=a); a.set_title(n); a.set_xlabel('Predicted'); a.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('cm.png',dpi=160); plt.close()
rf=Pipeline([('pre',pre),('clf',RandomForestClassifier(random_state=42))])
g=GridSearchCV(rf,{'clf__n_estimators':[100,200],'clf__max_depth':[10,20,None]},cv=3,scoring='accuracy').fit(Xtr,ytr)
print(g.best_params_); pr=g.predict(Xte); pb=g.predict_proba(Xte)[:,1]
res['Tuned Random Forest']=[accuracy_score(yte,pr),precision_score(yte,pr),recall_score(yte,pr),f1_score(yte,pr),roc_auc_score(yte,pb),g.best_score_]
print(confusion_matrix(yte,pr).tolist())
print(pd.DataFrame(res,index=['acc','prec','rec','f1','auc','cvscore']).T.round(3))
# importances
lr=Pipeline([('pre',pre),('clf',LogisticRegression(max_iter=1000))]).fit(Xtr,ytr)
names=lr.named_steps['pre'].get_feature_names_out()
co=pd.Series(lr.named_steps['clf'].coef_[0],index=names).sort_values()
print(co.head(5)); print(co.tail(5))
rfm=Pipeline([('pre',pre),('clf',RandomForestClassifier(n_estimators=300,random_state=42))]).fit(Xtr,ytr)
im=pd.Series(rfm.named_steps['clf'].feature_importances_,index=names).sort_values(ascending=False); print(im.head(6))
