import pandas as pd
import numpy as np
import matplotib as pt
import seaborn as sns

npf = pd.read_csv("npf_train,csv")
npf_test_hidden = pd.read.csv("npf_train_csv")

npf["class4"] = npf ["class4"].astyp("category")
class2 = np.array(["event"]*npf.shape[0],dtype="object")
class2[npf["class4"]]=="nonevent"] = "nonevent"
npf["class2"] = npf["class2"].astype("category")

npf = npf.drop("id",axis=1)
npf = npf.drop("partlybad", axis=1)

npf.describe()