from sklearn.externals import joblib
import pandas as pd
import xgboost as xgb

def feature_engineering(raw_data):
    for index,col in raw_data.iterrows():
        if (col['Temp']>38.5 or col['Temp']<36) and (col['HR']>90) :
            raw_data.at[index,'feature1']=1
        else:
            raw_data.at[index,'feature1']=0
        if col['Platelets']<100:
            raw_data.at[index,'feature2']=1
        else:
            raw_data.at[index,'feature2']=0
    sel_columns=['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp', 'Age', 'Gender', 'Unit1',
       'Unit2', 'HospAdmTime', 'ICULOS','feature1','feature2']
    sel_data=raw_data[sel_columns]
    data=xgb.DMatrix(sel_data,feature_names=sel_columns)
    return data

def load_sepsis_model():
	loaded_model = joblib.load('xgboost_v2.model')
	return loaded_model

def get_sepsis_score(data,model):
    
    org_feature = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
               'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
               'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
               'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
               'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
               'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
               'HospAdmTime', 'ICULOS']
    raw_cur_test = pd.DataFrame(data,columns=org_feature) 
    cur_test=feature_engineering(raw_cur_test)
    probs=model.predict(cur_test)
    y_pred = (probs >= 0.5)*1
    return probs[-1],y_pred[-1]