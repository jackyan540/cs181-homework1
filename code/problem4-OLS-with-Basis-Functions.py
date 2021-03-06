#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input varible is sunspots, is_years
# should be false
def make_basis(xx,part='a',is_years=True):
#DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        basis = np.vstack([xx**i for i in range(6)]).T
    if part == "a" and not is_years:
        xx = xx/20
        basis = np.vstack([xx**i for i in range(6)]).T
    if part == "b":
        basis = np.vstack([np.ones(xx.shape)]+[np.exp((-1*(xx-y)**2)/25) for y in range(1960,2011,5)]).T
    if part == "c":
        basis = np.vstack([np.ones(xx.shape)]+[np.cos(xx/j) for j in range(1,6)]).T
    if part == "d":
        basis = np.vstack([np.ones(xx.shape)]+[np.cos(xx/j) for j in range(1,26)]).T
    return basis

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
for i,p in enumerate(['a','b','c','d']):
    grid_X = make_basis(grid_years, part=p, is_years=True)
    x_basis = make_basis(years, part=p, is_years=True)
    w = find_weights(x_basis,Y)
    grid_Yhat = np.dot(w, grid_X.T)
    loss_Yhat = np.dot(w, make_basis(years, part=p).T)
    print(f"part 1{p} loss:",sum((Y-loss_Yhat)**2))
    plt.figure(4+i)
    plt.title("Data & Regression Plot for part 1"+p)
    plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()

# TODO: plot and report sum of squared error for each basis

# Plot the data and the regression line.
#plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
#plt.xlabel("Year")
#plt.ylabel("Number of Republicans in Congress")
#plt.show()

grid_ss = np.linspace(min(sunspot_counts), max(sunspot_counts), 200)
ss_counts = sunspot_counts[:13]
for i,p in enumerate(['a','c','d']):
    grid_X = make_basis(grid_ss, part=p, is_years=False)
    x_basis = make_basis(ss_counts, part=p, is_years=False)
    w = find_weights(x_basis, Y[:13])
    grid_Yhat = np.dot(w, grid_X.T)
    loss_Yhat = np.dot(w, make_basis(ss_counts, part=p, is_years=False).T)
    print(f"part 2{p} loss:",sum((Y[:13]-loss_Yhat)**2))
    plt.figure(8+i)
    plt.title("Data & Regression Plot for part 2"+p)
    plt.plot(ss_counts, republican_counts[:13], 'o', grid_ss, grid_Yhat, '-')
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.show()
