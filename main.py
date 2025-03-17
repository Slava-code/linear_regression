import sys
import matplotlib.pyplot as plt
import csv
import numpy as np

# get the arguments from input line
filename = sys.argv[1]
learning_rate = float(sys.argv[2])
iterations = int(sys.argv[3])

years = []
days = []

# read the file and get the year and days values
with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        years.append(float(row['year']))
        days.append(float(row['days']))

def visualize_data():
    plt.figure(figsize=(10, 6))
    plt.plot(years, days, linestyle='-', color='b')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.savefig("data_plot.jpg")

visualize_data()

def normalize():
    x = np.array(years)
    m = np.min(x)
    M = np.max(x)

    x_norm = (x-m) / (M-m)
    X_normalized = np.column_stack((x_norm, np.ones(len(x_norm))))
    return X_normalized

X_normalized = normalize()
print("Q3:")
print(X_normalized)

def q4():
    # turn array of days into numpy array
    Y = np.array(days)
    # take transpose of X
    transpose = X_normalized.T
    # calculate w and b
    weights = np.linalg.inv(transpose.dot(X_normalized)).dot(transpose.dot(Y))
    return weights

closed_form = q4()
print("Q4:")
print(closed_form)

def gradient_descent():
    weight = np.zeros((2, 1))
    y_days = np.array(days)
    # make a matrix out of x values
    x = X_normalized
 #   x = np.column_stack((x, np.ones(len(x))))
    n = len(years)

    # store the values
    weights = []
    weights.append(weight)

    for t in range(iterations):
        # every 10th iteration print the current weights
        if t % 10 == 0:
            print(weight.T[0])
        # summation notation
        sum = np.zeros((2,1))
        for i in range(n):
            y = weight.T.dot(x[i])
            y = y - y_days[i]
            sum += (y * x[i]).reshape((2, 1))
        sum = sum / n

        weight = weight - learning_rate * sum
        weights.append(weight)
    
    return weights

def graph_descent(weights):
    # turn array of days into numpy array
    Y = np.array(days)
    n = len(years)
    # calculate losses
    C = []
    for t in range(iterations):
        loss = 1/(2*n) * np.sum((X_normalized.dot(weights[t]) - Y) ** 2)
        C.append(loss)
    print(range(iterations))
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), C, linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.savefig("loss_plot.jpg")

print("Q5a:")
weights = gradient_descent()
print("Q5b: 0.45")
print("Q5c: 400")

graph_descent(weights)

def prediction(year):
    x = np.array(years)
    m = np.min(x)
    M = np.max(x)
    x = (year-m) / (M-m)

    y_hat = closed_form[0] * x + closed_form[1]
    return y_hat

y_hat = prediction(2023)
print("Q6: " + str(y_hat))

def q7():
    symbol = '<' if weights[-1][0] < 0 else ('>' if weights[-1][0] > 0 else '=')
    print("Q7a: " + symbol)
    print("Q7b: w > 0 indicated that the number of ice days increases from year to year, while w < 0 means it decreases. " + 
          "w = 0 indicates that the number of ice days stays the same")
    
q7()

def q8():
    x = np.array(years)
    m = np.min(x)
    M = np.max(x)

    x_star = - (closed_form[1] / closed_form[0])
    x_star = x_star * (M - m) + m

    return x_star

x_star = q8()
print("Q8a: " + str(x_star))
print("Q8b: It is a reasonable prediction if the trends during the analyzed time frame were linear. "+
      "However, the industrial activities of humans have amplified the trend of the freezing days decreasing, "+
      "likely leading to an exponential relationship. Therefore, human activiy might be the cause of the trend being non-linear, "+
      "which means the year predicted by the model could be significantly higher than the real value.")