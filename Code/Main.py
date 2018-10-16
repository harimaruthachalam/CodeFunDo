import csv
import numpy as np
import matplotlib.pyplot as plot
from copy import deepcopy

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

        # get total number of rows
    print("Total no. of rows: %d" % (csvreader.line_num))

# printing the field names
print('Field names are:' + ', '.join(field for field in fields))

#  printing first 5 rows
print('\nFirst 5 rows are:\n')
for row in rows[:5]:
    # parsing each column of a row
    for col in row:
        print("%10s" % col, end= " "),
    print('\n')

npRows = np.array(rows)
np.random.shuffle(npRows)

x = npRows[1:2000,1].astype(np.float)
y = npRows[1:2000,2].astype(np.float)


#plot.show()


X = np.array(list(zip(x, y)))
# Number of clusters
k = 15
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
while count != 200:
    print('Count is ' + str(count))
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

    if count % 20 == 0 :
        fig = plot.figure()
        plot.scatter(x[100:2000],y[100:2000], c='#050505', s=7)
        plot.axis([min(x) - 5, max(x) + 5, -180, 180])
        plot.scatter(C[0], C[1], marker='*', s=200, c='g')
        plot.savefig('test' + str(count) + '.png')
        plot.close(fig)



fig = plot.figure()
plot.scatter(x[100:2000],y[100:2000], c='#050505', s=7)
plot.axis([min(x) - 5,max(x) + 5,-180,180])
plot.scatter(C_x, C_y, marker='*', s=200, c='g')
plot.savefig('test0.png')
plot.close(fig)