# Install the rpart library
install.packages("rpart")

# Load a previously installed library, namely rpart
library(rpart)

install.packages("rpart.plot")
library(rpart.plot)

library(cluster)

install.packages("clValid")
library(clValid)


# Read the dataset with non demographic attributes only, stored in week1-non_demographic_data.csv file, into a data structure named 'week1' in R
week1 = read.csv("week1-non_demographic_data.csv", header = TRUE)

# Discard attributes: 'Year', 'Week', 'DF', 'DHF'
week1$Year = NULL
week1$week = NULL
week1$df = NULL
week1$dhf = NULL

# Divide the dataset in training and test sets
smp_size <- floor(2/3 * nrow(week1))
set.seed(123)
train_ind <- sample(seq_len(nrow(week1)), size = smp_size)
train <- week1[train_ind, ]
test <- week1[-train_ind, ]

# Grow the classification tree by defining the formula and constructing the tree
formula = outbreak ~ temp_avg+humidity+rainfall
tree <- rpart(formula, data=train, method="class")

### Draw an rpart plot with the number and percentage of observations
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###   N    Y 
###  44   78

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5491803

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
###  pred  N  Y
###     N 23 21
###     Y 34 44

# Grow the classification tree using the previously defined formula and constructing the tree by tuning the pre-pruning parameters
tree <- rpart(formula, data=train, method="class", control=rpart.control(cp=0.025))

# Draw the rpart plot of the pre-pruned tree, containing the number and percentage of observations
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###   N    Y 
###  56   66

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5655738

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
###  pred  N  Y
###     N 30 26
###     Y 27 39


# Create a copy of the dataset to be used with Clustering
week1_cl = week1

# Remove the attribute that is to be predicted
week1_cl$outbreak <- NULL

### The attributes are all numeric, so there's no need to convert them

# Rescale variables to assure compatibility
week1_cl = scale(week1_cl)

# Compute internal validation measures about the clustering models obtained with K-means and PAM algorithms
intern <- clValid(week1_cl, 2, clMethods=c("kmeans","pam"), validation="internal")

# View the computed results, that show that PAM is the best algorithm
summary(intern)

### Output
### Clustering Methods:
###  kmeans pam 
### 
### Cluster sizes:
###  2 
### 
### Validation Measures:
###                            2
###                             
### kmeans Connectivity  37.7488
###        Dunn           0.0278
###        Silhouette     0.3722
### pam    Connectivity  32.3329
###        Dunn           0.0487
###        Silhouette     0.3707
### 
### Optimal Scores:
### 
###              Score   Method Clusters
### Connectivity 32.3329 pam    2       
### Dunn          0.0487 pam    2       
### Silhouette    0.3722 kmeans 2

# Apply the pam clustering algorithm
pam.result <- pam(week1_cl, k=2, metric="manhattan")

# Check how many observations are in each cluster, and which are correct and incorrect
table(week1$outbreak, pam.result$clustering)

### Output, which gives that N should correspond to cluster 1 and Y to cluster 2 (189/364 correct observations)
###    1  2
### N 91 88
### Y 87 98

# Plot several graphs, including the silhouette plot, that provides a valuable insight into the accuracy of the clusters
plot(pam.result)
