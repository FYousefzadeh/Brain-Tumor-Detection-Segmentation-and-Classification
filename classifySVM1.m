function label=classifySVM1(DWT_feat)
svm1=loadCompactModel('SVM1');
label=predict(svm1,DWT_feat);
end
