import csv
import numpy as np
import matplotlib.pyplot as plot
from copy import deepcopy
import datetime
import pickle
from LinearReg import Regress

toLoad = False
toKMean = True
toPlot = False
toRegModel = True

# Number of clusters - Fixed
k = 100

if toLoad == True:
    # csv file name
    filename = "../Data/NCEDC_EarthQuake.csv"

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting field names through first row
        fields = next(csvreader)

        # extracting each data row one by one
        for row in csvreader:
            rows.append(row)

    npRows = np.array(rows)
    np.random.shuffle(npRows)
    listRows = []
    for i in range(len(npRows)):
        try:
            npRows[i,0] = datetime.datetime.strptime(npRows[i,0], "%Y/%m/%d %H:%M:%S.%f")
            npRows[i, 1] = float(npRows[i, 1])
            npRows[i, 2] = float(npRows[i, 2])
            npRows[i, 4] = float(npRows[i, 4])
            tempRow = [npRows[i,0], npRows[i,1], npRows[i,2], npRows[i,4]]
            listRows.append(tempRow)
        except Exception:
            print("Error at " + str(i))

    npRows = np.array(listRows)


    with open('dummpedData.obj', 'wb') as fp:
        pickle.dump(npRows, fp)

else:
    with open('dummpedData.obj', 'rb') as fp:
        npRows = pickle.load(fp)

x = npRows[1:1900429,1].astype(np.float)
y = npRows[1:1900429,2].astype(np.float)

X = np.array(list(zip(x, y)))

if toKMean == True:
    # X coordinates of random centroids
    C_x = np.random.randint(np.min(x)+20, np.max(x)-20, size=k)
    # Y coordinates of random centroids
    C_y = np.random.randint(np.min(y)+20, np.max(y)-20, size=k)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = np.linalg.norm(C - C_old, axis=1)
    # Loop will run till the error becomes zero
    count = 0
    while ((count != 200) and (error != 0)).all():
        print('In iteration of ' + str(count))
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = np.linalg.norm(X[i] - C, axis=1)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = np.linalg.norm(C - C_old, axis=1)
        count += 1

        # if count % 20 == 0 :
        #     fig = plot.figure()
        #     plot.scatter(x[100:20000],y[100:20000], c='#050505', s=7)
        #     plot.axis([min(x) - 5, max(x) + 5, -180, 180])
        #     plot.scatter(C[0], C[1], marker='*', s=200, c='g')
        #     plot.savefig('test' + str(count) + '.png')
        #     plot.close(fig)

    with open('kMeans.obj', 'wb') as fp:
        pickle.dump(C, fp)
else:
    with open('kMeans.obj', 'rb') as fp:
        C = pickle.load(fp)

if toPlot == True:
    fig = plot.figure()
    plot.scatter(x[100:20000],y[100:2000], c='#050505', s=7)
    plot.axis([min(x) - 5,max(x) + 5,-180,180])
    plot.scatter(C_x, C_y, marker='*', s=200, c='g')
    plot.savefig('test0.png')
    plot.close(fig)