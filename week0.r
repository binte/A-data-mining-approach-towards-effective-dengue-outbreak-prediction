# Install the rpart library
install.packages("rpart")

# Load a previously installed library, namely rpart
library(rpart)

install.packages("rpart.plot")
library(rpart.plot)

library(cluster)

install.packages("clValid")
library(clValid)


# Read the dataset with non demographic attributes only, stored in week0-non_demographic_data.csv file, into a data structure named 'week0' in R
week0 = read.csv("week0-non_demographic_data.csv", header = TRUE)

# Discard attributes: 'Year', 'Week', 'DF', 'DHF'
week0$Year = NULL
week0$week = NULL
week0$df = NULL
week0$dhf = NULL

# Divide the dataset in training and test sets
smp_size <- floor(2/3 * nrow(week0))
set.seed(123)
train_ind <- sample(seq_len(nrow(week0)), size = smp_size)
train <- week0[train_ind, ]
test <- week0[-train_ind, ]

# Grow the classification tree by defining the formula and constructing the tree
formula = outbreak ~ temp_avg+humidity+rainfall
tree <- rpart(formula, data=train, method="class")

# Draw an rpart plot with the number and percentage of observations
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###  N  Y 
### 48 74 

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)
    
### Output
### pred  N  Y
###    N 24 24
###    Y 37 37

# Grow the classification tree using the previously defined formula and constructing the tree by tuning the pre-pruning parameters
tree <- rpart(formula, data=train, method="class", control=rpart.control(cp=0.03, minbucket=3, minsplit=8))

# Draw the rpart plot of the pre-pruned tree, containing the number and percentage of observations
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###   N    Y 
###  21   101 

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5491803

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
###  pred  N  Y
###     N 12 9
###     Y 49 52


# Create a copy of the dataset to be used with Clustering
week0_cl = week0

# Remove the attribute that is to be predicted
week0_cl$outbreak <- NULL

### The attributes are all numeric, so there's no need to convert them

# Rescale variables to assure compatibility
week0_cl = scale(week0_cl)

# Compute internal validation measures about the clustering models obtained with K-means and PAM algorithms
intern <- clValid(week0_cl, 2, clMethods=c("kmeans","pam"), validation="internal")

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
###        Silhouette     0.3725
### pam    Connectivity  32.3329
###        Dunn           0.0486
###        Silhouette     0.3712
### 
### Optimal Scores:
### 
###              Score   Method Clusters
### Connectivity 32.3329 pam    2       
### Dunn          0.0486 pam    2       
### Silhouette    0.3725 kmeans 2

# Apply the pam clustering algorithm
pam.result <- pam(week0_cl, k=2)

# Check how many observations are in each cluster, and which are correct and incorrect
table(week0$outbreak, pam.result$clustering)

### Output, which gives that N should correspond to cluster 2 and Y to cluster 1 (191/365 correct observations	)
###    1  2
### N 87 92
### Y 99 87

# Plot several graphs, including the silhouette plot, that provides a valuable insight into the accuracy of the clusters
plot(pam.result)
