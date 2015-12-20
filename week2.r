# Install the rpart library
install.packages("rpart")

# Load a previously installed library, namely rpart
library(rpart)

# Install the rpart.plot library
install.packages("rpart.plot")

# Load a previously installed library, namely rpart.plot
library(rpart.plot)

install.packages("clValid")
library(clValid)



# Read the dataset with non demographic attributes only, stored in week2-non_demographic_data.csv file, into a data structure named 'week2' in R
week2 = read.csv("week2-non_demographic_data.csv", header = TRUE)

# Discard attributes: 'Year', 'Week', 'DF', 'DHF'
week2$Year = NULL
week2$week = NULL
week2$df = NULL
week2$dhf = NULL

# Divide the dataset in training and test sets
smp_size <- floor(2/3 * nrow(week2))
set.seed(123)
train_ind <- sample(seq_len(nrow(week2)), size = smp_size)
train <- week2[train_ind, ]
test <- week2[-train_ind, ]

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
###  60   61

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5867769

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
###  pred  N  Y
###     N 34 26
###     Y 24 37

# Grow the classification tree using the previously defined formula and constructing the tree by tuning the pre-pruning parameters
tree <- rpart(formula, data=train, method="class", control=rpart.control(cp=0.024793))

# Draw the rpart plot of the pre-pruned tree, containing the number and percentage of observations
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###   N    Y 
###  47   74

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.5950413

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
###  pred  N  Y
###     N 28 19
###     Y 30 44


####################
# Create a copy of the dataset to be used with Clustering
week2_cl = week2

# Remove the attribute that is to be predicted
week2_cl$outbreak <- NULL

### The attributes are all numeric, so there's no need to convert them

# Rescale variables to assure compatibility
week2_cl = scale(week2_cl)

# Compute internal validation measures about the clustering models obtained with K-means and PAM algorithms
intern <- clValid(week2_cl, 2, clMethods=c("kmeans","pam"), validation="internal")

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
### kmeans Connectivity  35.3556
###        Dunn           0.0482
###        Silhouette     0.3732
### pam    Connectivity  31.7802
###        Dunn           0.0487
###        Silhouette     0.3723
### 
### Optimal Scores:
### 
###              Score   Method Clusters
### Connectivity 31.7802 pam    2       
### Dunn          0.0487 pam    2       
### Silhouette    0.3732 kmeans 2

# Apply the pam clustering algorithm
pam.result <- pam(week2_cl, k=2, metric="manhattan")

# Check how many observations are in each cluster, and which are correct and incorrect
table(week2$outbreak, pam.result$clustering)

### Output, which gives that N should correspond to cluster 2 and Y to cluster 1 (199/363 correct observations)
###     1   2
### N  79 100
### Y  99  85

# Plot several graphs, including the silhouette plot, that provides a valuable insight into the accuracy of the clusters
plot(pam.result)
