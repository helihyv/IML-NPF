import pandas as pd
import numpy as np
import matplotlib as pt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

random_state = 555

npf = pd.read_csv("npf_train.csv")
npf_test_hidden = pd.read_csv("npf_test_hidden.csv")

print (npf)
print (npf.describe(include="all"))

print("Hidden data")
print (npf_test_hidden.describe(include="all"))


npf["class4"] = npf ["class4"].astype("category")
class2 = np.array(["event"]*npf.shape[0],dtype="object")
class2[npf["class4"]=="nonevent"] = "nonevent"
npf["class2"] = class2
npf["class2"] = npf["class2"].astype("category")

npf = npf.drop("id",axis=1)
npf = npf.drop("partlybad", axis=1)
npf = npf.drop("date", axis=1)

print (npf)
print (npf.describe(include="all"))

npf_test_hidden = npf_test_hidden.drop("id",axis=1)
npf_test_hidden = npf_test_hidden.drop("partlybad", axis=1)
npf_test_hidden = npf_test_hidden.drop("date", axis=1)
npf_test_hidden = npf_test_hidden.drop("class4", axis=1)

print("Hidden test data")
print (npf_test_hidden.describe(include="all"))


sample_size = 240

npf_train = npf.sample(n=sample_size, random_state=random_state)
npf_test = npf.drop(npf_train.index)

print (npf_train.describe(include="all"))
print (npf_test.describe(include="all"))

npf_train_2class = npf_train.drop("class4", axis=1)
npf_test_2class = npf_test.drop("class4",axis=1)
npf_all_2class = npf.drop("class4",axis=1)

npf_train_4class = npf_train.drop("class2", axis=1)
npf_test_4class = npf_test.drop("class2", axis=1)
npf_all_4class = npf.drop("class2",axis=1)

## Division to buckets fo 10-fold cross-validation

buckets = []
buckets_4class = []
remaining_set = npf_train
for i in range(9):
    bucket = remaining_set.sample(24, random_state=random_state)
    buckets.append(bucket.drop("class4",axis=1))
    buckets_4class.append(bucket.drop("class2", axis=1))
    remaining_set = remaining_set.drop(buckets[i].index)
    print(buckets[i].describe(include="all"))
buckets.append(remaining_set.drop("class4",axis=1))
buckets_4class.append(remaining_set.drop("class2",axis=1))
print(buckets[9].describe(include="all"))

##Function for 10-fold cross-validation


def cross_validate (all, buckets, model):
    corrects_ratios = np.empty((10,1))
    for i in range(10):
        train_now = all.drop(buckets[i].index)
        X = train_now.drop("class2",axis=1)
        y = train_now["class2"]
        model.fit(X,y)
        validateX = buckets[i].drop("class2",axis=1)

        predictions = model.predict(validateX)
        corrects = buckets[i]["class2"] == predictions
        corrects_ratios[i] = np.count_nonzero(corrects) / len(corrects)
        print (corrects_ratios[i])
    
    return np.mean(corrects_ratios)

## The classifiers

print("Random forest with 100 trees")
clf_rt100 = RandomForestClassifier(random_state=random_state)
clf_rt100_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt100)
print(clf_rt100_mean_correct_ratio)

print("200 trees")
clf_rt200 = RandomForestClassifier(n_estimators=200, random_state=random_state)
clf_rt200_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt200)
print(clf_rt200_mean_correct_ratio)

print("300 trees")
clf_rt300 = RandomForestClassifier(n_estimators=300, random_state=random_state)
clf_rt300_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt300)
print(clf_rt300_mean_correct_ratio)

print("Random forest with 100 trees, max_features=None" )
clf_rt100N = RandomForestClassifier(random_state=random_state, max_features=None)
clf_rt100N_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt100N)
print(clf_rt100N_mean_correct_ratio)

print("200 trees, max_features=Log2")
clf_rt200L = RandomForestClassifier(n_estimators=200, random_state=random_state, max_features=None)
clf_rt200L_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt200L)
print(clf_rt200L_mean_correct_ratio)

print("Random forest with 100 trees, max_features=Log2" )
clf_rt100L = RandomForestClassifier(random_state=random_state, max_features="log2")
clf_rt100L_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt100L)
print(clf_rt100L_mean_correct_ratio)


print("200 trees, max_features=Log2")
clf_rt200L = RandomForestClassifier(n_estimators=200, random_state=random_state)
clf_rt200L_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt200L)
print(clf_rt200L_mean_correct_ratio)

print("300 trees, max_featutes=log2")
clf_rt300L = RandomForestClassifier(n_estimators=300, random_state=random_state, max_features="log2")
clf_rt300L_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_rt300L)
print(clf_rt300L_mean_correct_ratio)


print("Logistic regression, saga solver, l1 regialrization, C=1")
clf_lr_l2_c1 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=5000, random_state = random_state))
clf_lr_l2_c1_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c1)
print(clf_lr_l2_c1_mean_correct_ratio)


print("Logistic regression, saga solver, l1 regialrization, C=0.5")
clf_lr_l2_c01 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.5, max_iter=5000, random_state = random_state))
clf_lr_l2_c01_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c01)
print(clf_lr_l2_c01_mean_correct_ratio)


print("Logistic regression, saga solver, l1 regialrization, C=5")
clf_lr_l2_c10= make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1",C=5, max_iter=5000, random_state = random_state))
clf_lr_l2_c10_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c10)
print(clf_lr_l2_c10_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=0.9")
clf_lr_l2_c09 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.9, max_iter=5000, random_state = random_state))
clf_lr_l2_c09_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c09)
print(clf_lr_l2_c09_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=0.7")
clf_lr_l2_c07 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.7, max_iter=5000, random_state = random_state))
clf_lr_l2_c07_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c07)
print(clf_lr_l2_c07_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=1.5")
clf_lr_l2_c15 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=1.5, max_iter=5000, random_state = random_state))
clf_lr_l2_c15_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c15)
print(clf_lr_l2_c15_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=1.3")
clf_lr_l2_c13 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=1.3, max_iter=5000, random_state = random_state))
clf_lr_l2_c13_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l2_c13)
print(clf_lr_l2_c13_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=1")
clf_lr_l22_c1 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", max_iter=5000, random_state = random_state))
clf_lr_l22_c1_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c1)
print(clf_lr_l22_c1_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=0.9")
clf_lr_l22_c09 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=0.9, max_iter=5000, random_state = random_state))
clf_lr_l22_c09_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c09)
print(clf_lr_l22_c09_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=1.5")
clf_lr_l22_c15 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=1.5, max_iter=5000, random_state = random_state))
clf_lr_l22_c15_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c15)
print(clf_lr_l22_c15_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=0.5")
clf_lr_l22_c05 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=0.5, max_iter=5000, random_state = random_state))
clf_lr_l22_c05_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c05)
print(clf_lr_l22_c05_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=0.1")
clf_lr_l22_c05 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=0.1, max_iter=5000, random_state = random_state))
clf_lr_l22_c05_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c05)
print(clf_lr_l22_c05_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=10")
clf_lr_l22_c10 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=10, max_iter=5000, random_state = random_state))
clf_lr_l22_c10_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c10)
print(clf_lr_l22_c10_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=0.01")
clf_lr_l22_c01 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=0.01, max_iter=5000, random_state = random_state))
clf_lr_l22_c01_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c01)
print(clf_lr_l22_c01_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=0.01")
clf_lr_l1_c01 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.01, max_iter=5000, random_state = random_state))
clf_lr_l1_c01_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l1_c01)
print(clf_lr_l1_c01_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=0.05")
clf_lr_l22_c05 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", C=0.05, max_iter=5000, random_state = random_state))
clf_lr_l22_c05_mean_correct_ratio = cross_validate(npf_train_2class, buckets, clf_lr_l22_c05)
print(clf_lr_l22_c05_mean_correct_ratio)

## Logistic regression with l1 regularisation and c=1 performed best! Now train it with full training data.

clf_class2 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=5000, random_state = random_state))
X_class2 = npf_train_2class.drop("class2", axis=1)
y_class2 = npf_train_2class["class2"]
clf_class2.fit(X_class2,y_class2)

## Calculate success on training data

predictions_test_class2 = clf_class2.predict(npf_test_2class.drop("class2", axis=1))
corrects_test_class2 = predictions_test_class2 == npf_test_2class["class2"]
correct_ratio_class2 = np.count_nonzero(corrects_test_class2) / len(predictions_test_class2)
print("Classification success for class2 on test data was ")
print(correct_ratio_class2)

## ratio was 0.8348623853211009


## Now train with all available data to improve acuracy

clf_class2_final = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=5000, random_state = random_state))
X_class2_final = npf_all_2class.drop("class2", axis=1)
y_class2_final = npf_all_2class["class2"]
clf_class2_final.fit(X_class2_final,y_class2_final)

# Predict on the hidden test data

class2_predictions_hidden = clf_class2_final.predict(npf_test_hidden)
print("Hidden predictions")
print(class2_predictions_hidden)

class2_probabilities_hidden = clf_class2_final.predict_proba(npf_test_hidden)

print("Hidden probabilities")
print(class2_probabilities_hidden)

print("Intercept and coefficienst")
class2_final_coef =clf_class2_final[1].coef_
class2_final_intercept = clf_class2_final[1].intercept_
print("Amount of non-zero co-efficients")
print (np.count_nonzero(class2_final_coef))
print("Coefficients)")
print(class2_final_coef)
print("Intercept")
print(class2_final_intercept)



#################################
#Classification to four classes #
##################################


##Function for 10-fold cross-validation, multiclass


def cross_validate_multiclass (all, buckets, model):
    corrects_ratios = np.empty((10,1))
    for i in range(10):
        train_now = all.drop(buckets[i].index)
        X = train_now.drop("class4",axis=1)
        y = train_now["class4"]
        model.fit(X,y)
        validateX = buckets[i].drop("class4",axis=1)

        predictions = model.predict(validateX)
        corrects = buckets[i]["class4"] == predictions
        corrects_ratios[i] = np.count_nonzero(corrects) / len(corrects)
        print (corrects_ratios[i])
    
    
    return np.mean(corrects_ratios)

print("MULTICLASS classification")

print("Logistic regression, saga solver, l1 regialrization, C=1, multinomial")
clf_lr_l2_c1_multi = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=6000, random_state = random_state))
clf_lr_l2_c1_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l2_c1_multi)
print(clf_lr_l2_c1_multi_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=0.1, multinomial")
clf_lr_l2_c01_multi = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.1, max_iter=6000, random_state = random_state))
clf_lr_l2_c01_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l2_c01_multi)
print(clf_lr_l2_c01_multi_mean_correct_ratio)

print("Logistic regression, saga solver, l1 regialrization, C=0.5, multinomial")
clf_lr_l2_c05_multi = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=0.5, max_iter=6000, random_state = random_state))
clf_lr_l2_c05_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l2_c05_multi)
print(clf_lr_l2_c05_multi_mean_correct_ratio)


print("Logistic regression, saga solver, l1 regialrization, C=10, multinomial")
clf_lr_l2_c10_multi = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", C=10, max_iter=6000, random_state = random_state))
clf_lr_l2_c10_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l2_c10_multi)
print(clf_lr_l2_c10_multi_mean_correct_ratio)


print("Logistic regression, saga solver, l1 regialrization, C=1, ovr")
clf_lr_l2_c1_ovr = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=6000, random_state = random_state, multi_class="ovr"))
clf_lr_l2_c1_ovr_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l2_c1_ovr)
print(clf_lr_l2_c1_ovr_mean_correct_ratio)

print("Logistic regression, saga solver, l2 regialrization, C=1, multinomial")
clf_lr_l22_c1_multi = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l2", max_iter=6000, random_state = random_state))
clf_lr_l22_c1_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_lr_l22_c1_multi)
print(clf_lr_l22_c1_multi_mean_correct_ratio)


print("Random forest with 100 trees")
clf_rt100_multi = RandomForestClassifier(random_state=random_state)
clf_rt100_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_rt100_multi)
print(clf_rt100_multi_mean_correct_ratio)

print("Random forest with 200 trees")
clf_rt200_multi = RandomForestClassifier(random_state=random_state, n_estimators=200)
clf_rt200_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_rt200_multi)
print(clf_rt200_multi_mean_correct_ratio)

print("Random forest with 300 trees")
clf_rt300_multi = RandomForestClassifier(random_state=random_state, n_estimators=300)
clf_rt300_multi_mean_correct_ratio = cross_validate_multiclass(npf_train_4class, buckets_4class, clf_rt300_multi)
print(clf_rt300_multi_mean_correct_ratio)

## Best stays the same, multinomial multiclass better

print("Training the best performin classifier weith ther whole training data")
print("Logistic regression, saga solver, l1 regialrization, C=1, multinomial")
clf_class4 = make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=6000, random_state = random_state))
clf_class4.fit(npf_train_4class.drop("class4", axis=1), npf_train_4class["class4"])
class4_predictions =  clf_class4.predict(npf_test_4class.drop("class4",axis=1))
class4_corrects = class4_predictions == npf_test_4class["class4"]
correct_ratio_class4 = np.count_nonzero(class4_corrects) / len(class4_predictions)  

print("Correct ratio on test data")
print(correct_ratio_class4)
#0.6330275229357798


events_predicted = class4_predictions != "nonevent"
events_occurred = npf_test_4class["class4"] != "nonevent"
binary_corrects = events_predicted == events_occurred
correct_ratio_binary = np.count_nonzero(binary_corrects) / len(binary_corrects)


print("Binary classifdication correct ratio:")
print(correct_ratio_binary)
#0.8532110091743119


## Finally train on the whole data

clf_class4_final =make_pipeline(StandardScaler(),LogisticRegression(solver="saga", penalty="l1", max_iter=6000, random_state = random_state))
clf_class4_final.fit(npf_train_4class.drop("class4", axis=1), npf_train_4class["class4"])


## Predict on hidden test data, propability required is of the three event classes combined 

npf_test_hidden.describe()

class4_predictions_hidden = clf_class4_final.predict(npf_test_hidden)
print("Hidden predictions")


class4_probabilities_hidden = clf_class4_final.predict_proba(npf_test_hidden)

print("Hidden probabilities")



print("Class order in probabilities")
print(clf_class4_final.classes_)

event_probabilities_hidden = np.sum (class4_probabilities_hidden[:,0:3], axis=1)
print(event_probabilities_hidden)
print(class4_predictions_hidden)
results = pd.DataFrame({"class4": class4_predictions_hidden, "p": event_probabilities_hidden})

results.describe(include="all")
results.to_csv("results.csv", index=None)

print("Intercept and coefficienst")
class4_final_coef =clf_class4_final[1].coef_
class4_final_intercept = clf_class4_final[1].intercept_
print("Amount of nonzro coefficients by class")
print (np.count_nonzero(class4_final_coef, axis=1))
print("Coefficients")
print(class4_final_coef)
print("Intercepts")
print(class4_final_intercept)

print (np.count_nonzero(np.sum(class4_final_coef, axis=0)))

class4_final_coef_asdf =pd.DataFrame(class4_final_coef)
print(class4_final_coef_asdf.describe())

