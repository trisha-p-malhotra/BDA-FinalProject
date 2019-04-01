import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sklearn.linear_model as lm
import sklearn.metrics as mt
import sklearn.preprocessing as preprocess
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB




from sklearn.cross_validation import train_test_split

#reading the data from the file
def read_store_file(filename):
    dataframe_rep = pd.read_csv(filename,sep=";")
    #print(dataframe_rep.columns)
    attributes_names = dataframe_rep.columns
    #print(dataframe_rep.info())
    #print(dataframe_rep.corr())
    array_rep = np.array(dataframe_rep)
    return array_rep,attributes_names

# (['age', 'job', 'marital', 'education', 'default', 'housing', 'loan','contact', 'month', 'day_of_week', 'duration',
# 'campaign', 'pdays','previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
# 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'],


# performing basic exploratory data analysis on given array

def distribution_data(array_data,attribute_names):
    # Age distribution  - could be changed to categorical for avoiding a sparse matrix

    rec,attr=array_data.shape
    for i in range(attr-1):
        plt.hist(array_data[:,[i]])
        plt.xlabel(attribute_names[i])
        plt.ylabel("Frequency")
        plt.savefig(attribute_names[i]+".PNG")
        plt.close()


    # Call duration distribution -



def normalize(attributes):
    attr = preprocess.normalize(attributes)
    return attr


def correlation(attributes):

    print(np.corrcoef(attributes))







def preprocessing_1(array_data):


    # age
    
    for i in range(0, len(array_data[:,0])):
        j=array_data[:,0][i]
        if (j <= 19):
            j =1
        elif (j <= 29):
            j =2
        elif (j <= 39):
            j =3
        elif (j <= 49):
            j =4
        elif (j <= 59):
            j =5
        elif (j <= 69):
            j =6
        else:
            j =7
        array_data[:,0][i] = j


    # job
    for i in range(0, len(array_data[:,1])):
        j = array_data[:, 1][i]
        if j == "housemaid":
            j =1
        elif i=="services":
            j =2
        elif i=="admin.":
            j =3
        elif i=="blue-collar":
            j =4
        elif i=="technician":
            j =5
        elif i=="retired":
            j =6
        elif i=="management":
            j =7
        elif i=="unemployed":
            j =8
        elif i=="self-employed":
            j =9
        elif i=="entrepreneur":
            j =10
        elif i=="student":
            j =11
        else:
            j =0
        array_data[:,1][i] = j

    #Marital status

    for i in range(0, len(array_data[:, 2])):
        j = array_data[:, 2][i]
        if j == "married":
            j =1
        elif j == "single":
            j =2
        elif j == "divorced":
            j =3
        else:
            j=0
        array_data[:, 2][i] = j


    # education
    for i in range(0, len(array_data[:,3])):
        j = array_data[:, 3][i]
        if j == "basic.4y":
            j =1
        elif i=="high.school":
            j =2
        elif i=="basic.6y":
            j =3
        elif i=="basic.9y":
            j =4
        elif i=="professional.course":
            j =5
        elif i=="university.degree":
            j =6
        elif i=="illiterate":
            j =7
        else:
            j =0
        array_data[:,3][i] = j


    # default
    for i in range(0, len(array_data[:, 4])):
        j = array_data[:, 4][i]
        if j == "no":
            j =1
        elif j == "yes":
            j =2
        else:
            j =0
        array_data[:, 4][i] = j


    # Housing
    for i in range(0, len(array_data[:, 5])):
        j = array_data[:, 5][i]
        if j == "no":
            j =1
        elif j == "yes":
            j =2
        else:
            j =0
        array_data[:, 5][i] = j

    #loan
    for i in range(0, len(array_data[:, 6])):
        j = array_data[:, 6][i]
        if j == "no":
            j =1
        elif j == "yes":
            j =2
        else:
            j =0
        array_data[:, 6][i] = j

    #Communications
    for i in range(0, len(array_data[:, 7])):
        j = array_data[:, 7][i]
        if j == "telephone":
            j =1
        elif j == "cellular":
            j =2
        else:
            j =0
        array_data[:, 7][i] = j

    #Month
    for i in range(0, len(array_data[:, 8])):
        j = array_data[:, 8][i]
        if j == "jan":
            j =1
        elif j == "feb":
            j =2
        elif j == "mar":
            j =3
        elif j == "apr":
            j =4
        elif j == "may":
            j =5
        elif j == "jun":
            j =6
        elif j == "jul":
            j =7
        elif j == "aug":
            j =8
        elif j == "sep":
            j =9
        elif j == "oct":
            j =10
        elif j == "nov":
            j =11
        elif j == "dec":
            j =12

        else:
            j =0
        array_data[:, 8][i] = j


    #Day of week
    for i in range(0, len(array_data[:, 9])):
        j = array_data[:, 9][i]
        if j == "mon":
            j =1
        elif j == "tue":
            j =2
        elif j == "wed":
            j =3
        elif j == "thu":
            j =4
        elif j == "fri":
            j =5
        else:
            j =0
        array_data[:, 9][i] = j

     #Call duration
    for i in range(0, len(array_data[:,10])):
        j = array_data[:, 10][i]
        if (j <= 50):
            j =1
        elif (j <= 100):
            j =2
        elif (j <= 150):
            j =3
        elif (j <= 200):
            j =4
        elif (j <= 250):
            j =5
        elif (j <= 300):
            j =6
        elif (j <= 400):
            j =7
        elif (j <= 600):
            j =8
        elif (j <= 1000):
            j =9
        else:
            j =10
        array_data[:,10][i] = j

    #Campaign days
    for i in range(0, len(array_data[:,11])):
        j = array_data[:, 11][i]
        if (j <= 2):
            j =1
        elif (j <= 4):
            j =2
        elif (j <= 6):
            j =3
        elif (j <= 8):
            j =4
        elif (j <= 12):
            j =5
        elif (j <= 16):
            j =6
        elif (j <= 20):
            j =7
        elif (j <= 30):
            j =8
        elif (j <= 40):
            j =9
        else:
            j =10
        array_data[:,11][i] = j

    #pdays
    for i in range(0, len(array_data[:,12])):
        j = array_data[:, 12][i]
        if (j <= 1):
            j =1
        elif (j <= 3):
            j =2
        elif (j <= 6):
            j =3
        elif (j <= 9):
            j =4
        elif (j <= 12):
            j =5
        elif (j <= 16):
            j =6
        elif (j <= 20):
            j =7
        elif (j <= 30):
            j =8
        elif (j <= 40):
            j =9
        elif (j==999):
            j=0
        else:
            j =10
        array_data[:,12][i] = j

    #Previous
    for i in range(0, len(array_data[:,13])):
        j = array_data[:, 13][i]
        if (j <= 5):
            pass
        else:
            j =100
        array_data[:,13][i] = j

    #Poutcome
    for i in range(0, len(array_data[:, 14])):
        j = array_data[:, 14][i]

        if j == "failure":
            j =1
        elif j == "success":
            j =2
        else:
            j =0
        array_data[:, 14][i] = j


    # # emp.var.rate  preprocessing not required
    # values = [1.1, 1.4, -0.1, -0.2, -1.8, -2.9, -3.4, -3, -1.7, -1.1]
    # for i in range(0, len(array_data[:, 16])):
    #     j = array_data[:, 15][i]
    #     j = values.index(j)
    #     array_data[:, 15][i] = j

    # # cons.price.idx
    # for i in range(0, len(array_data[:, 16])):
    #     j = array_data[:, 16][i]
    #
    #     if j>= 92.01 and j < 92.50:
    #         j= 0
    #     elif j>=92.50 and j<93.00:
    #         j=1
    #     elif j>= 93.00 and j < 93.50:
    #         j= 2
    #     elif j>=93.50 and j<94.00:
    #         j=3
    #     elif j >= 94.00 and j < 94.50:
    #         j = 4
    #     elif j >= 94.50 and j < 95.00:
    #         j = 5
    #
    #     array_data[:, 16][i] = j

    #cons.conf.idx
    for i in range(0, len(array_data[:,17])):
        j = array_data[:, 17][i]
        if (j <= -50):
            j =1
        elif (j <= -47):
            j =2
        elif (j <= -44):
            j =3
        elif (j <= -41):
            j =4
        elif (j <= -38):
            j =5
        elif (j <= -35):
            j =6
        elif (j <= -30):
            j =7
        elif (j <= -25):
            j =8
        elif (i>-25):
            i=9
        else:
            j =0
        array_data[:,17][i] = j

    #euribor3m
    for i in range(0, len(array_data[:,18])):
        j = array_data[:, 18][i]
        if (j <= 0.3):
            j =1
        elif (j <= 0.6):
            j =2
        elif (j <= 0.9):
            j =3
        elif (j <= 1.2):
            j =4
        elif (j <= 1.5):
            j =5
        elif (j <= 1.8):
            j =6
        elif (j <= 2.1):
            j =7
        elif (j <= 2.4):
            j =8
        elif (j <= 2.7):
            j =9
        elif (j <= 3.0):
            j =10
        elif (j <= 3.3):
            j =11
        elif (j <= 3.6):
            j =12
        elif (j <= 3.9):
            j =13
        elif (j <= 4.2):
            j =14
        elif (j <= 4.5):
            j =15
        elif (j <= 4.8):
            j =16
        else:
            j =20
        array_data[:,18][i] = j

    # #nr.employed does not require categorization
    # values2 = [5191,5228.1,5195.8,5176.3,5099.1,5076.2,5017.5,5023.5,5008.7,4991.6,4963.6]
    # for i in range(0, len(array_data[:, 19])):
    #     j = array_data[:, 19][i]
    #     j = values2.index(j)
    #     array_data[:, 19][i] = j


    #Y
    for i in range(0, len(array_data[:, 20])):
        j = array_data[:, 20][i]
        if j == "no":
            j =0
        elif j == "yes":
            j =1
        else:
            j =2
        array_data[:, 20][i] = j

    return array_data


def build_model(attributes,target):
    X_train, X_test, y_train, y_test = train_test_split(attributes,target,test_size=0.20)
    # could try with multiple models
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    prediction = NB.predict(X_test)
    # accuracy = mt.accuracy_score(y_test, prediction)

    fpr_LgR, tpr_Lgr, t_Lgr = roc_curve(y_test, prediction)
    AUC_LgR = mt.auc(fpr_LgR, tpr_Lgr)
    print("-----------------------Accuracy score of Pt-----------------------------")
    print(mt.accuracy_score(y_test, prediction))
    print("-----------------------Precision score of Pt-----------------------------")
    print(mt.precision_score(y_test, prediction))
    print("-----------------------Recall score of Pt-----------------------------")
    print(mt.recall_score(y_test, prediction))
    print("-----------------------F1 score of Pt-----------------------------")
    print(mt.f1_score(y_test, prediction))
    print("-----------------------AUC of Pt-----------------------------")
    print(AUC_LgR)
    # print(len(y_train))
    # model = lm.LogisticRegression()
    # model.fit(X_train,y_train.astype(int))
    # predicted = model.predict(X_test)
    # print(mt.accuracy_score(predicted,y_test.astype(int)))

def removeUnknowns(data):
    new_data = []
    for row in data:
        if not 'unknown' in row:
            new_data.append(row)
    print(len(new_data))
    return np.array(new_data)


def main():
    #filename= input(" Enter the name of the file along with the extension : ")
    array_data,attribute_names=read_store_file("bank-additional-full.csv")

    #distribution_data(array_data,attribute_names)
    #build_model(attributes,target)


    data = removeUnknowns(array_data)
    data = preprocessing_1(data)
    print(data[0])
    target = data[:, 20]
    build_model(data, target)
    # print(data)
    # attributes = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]]

    # normalized_data = normalize(attributes)




    #print(normalized_data)
    #distribution_data(attributes,attribute_names)
    # build_model(normalized_data,target)   # we could try with other classification models


if __name__ == '__main__':
    main()