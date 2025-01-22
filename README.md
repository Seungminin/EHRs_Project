
python mimic.py TS_ELECTRA Transfer pre_train_heg_sample Test_data.csv
(emb, task, pre_train_data, fine_tuning_data)


### HyperParameter Tuning 이용시
python mimic.py TS_ELECTRA Transfer pre_train_heg_sample heg_sample_data_without_id.csv

### Final Function 변경 시
python mimic_final.py TS_ELECTRA Transfer pre_train_heg_sample heg_sample_data_without_id.csv

### SMOTE dataset 증강 이용시
python mimic_final.py TS_ELECTRA Transfer pre_train_heg_sample2 SMOTE_pre_train.csv

### TVAE dataset 증강 이용시
python mimic_final.py TS_ELECTRA Transfer pre_train_heg_sample2 TVAE_data.csv

### XGBoost Undersampling 사용시
python mimic_final.py TS_ELECTRA Transfer pre_train_heg_sample2 XGBoost_undersampling.csv

### K-Fold 사용시
python mimic_apply_kfold.py TS_ELECTRA Transfer pre_train_heg_sample2 XGBoost_undersampling.csv
