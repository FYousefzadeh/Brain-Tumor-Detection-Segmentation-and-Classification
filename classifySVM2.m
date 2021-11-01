function label=classifySVM2(DWT_feat)
svm2=loadCompactModel('SVM2');
label=predict(svm2,DWT_feat);
end