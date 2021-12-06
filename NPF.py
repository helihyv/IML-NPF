import pandas as pd
import numpy as np
import matplotlib as pt
import seaborn as sns

random_state = 555

npf = pd.read_csv("npf_train.csv")
npf_test_hidden = pd.read_csv("npf_test_hidden.csv")

print (npf)
print (npf.describe(include="all"))

npf["class4"] = npf ["class4"].astype("category")
class2 = np.array(["event"]*npf.shape[0],dtype="object")
class2[npf["class4"]=="nonevent"] = "nonevent"
npf["class2"] = class2
npf["class2"] = npf["class2"].astype("category")

npf = npf.drop("id",axis=1)
npf = npf.drop("partlybad", axis=1)

print (npf)
print (npf.describe(include="all"))

sample_size = 240

npf_train = npf.sample(n=sample_size, random_state=random_state)
npf_test = npf.drop(npf_train.index)

print (npf_train.describe(include="all"))
print (npf_test.describe(include="all"))

## Division to buckets fo 10-fold cross-validation

buckets = []
remaining_set = npf_train
for i in range(9):
    buckets.append(remaining_set.sample(24))
    remaining_set = remaining_set.drop(buckets[i].index)
    print(buckets[i].describe(include="all"))
buckets.append(remaining_set)
print(buckets[9].describe(include="all"))

