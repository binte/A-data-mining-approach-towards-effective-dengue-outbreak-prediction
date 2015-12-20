# Install the rpart library
install.packages("rpart")

# Load a previously installed library, namely rpart
library(rpart)

install.packages("rpart.plot")
library(rpart.plot)

library(cluster)

install.packages("clValid")
library(clValid)


# Read the main dataset, stored in dengue.csv file, into a data structure named 'dat_demographic' in R
dat_demographic = read.csv("dengue.csv", header = TRUE)

# Discard attributes: 'Year', 'Week', 'DF', 'DHF', 'Town', 'District', 'Epidemic'
dat_demographic$year = NULL
dat_demographic$week = NULL
dat_demographic$df = NULL
dat_demographic$dhf = NULL
dat_demographic$temp_avg = NULL
dat_demographic$humidity = NULL
dat_demographic$rainfall = NULL
dat_demographic$town = NULL
dat_demographic$district = NULL
dat_demographic$epidemic = NULL

# Divide the dataset in training and test sets
smp_size <- floor(2/3 * nrow(dat_demographic))
set.seed(123)
train_ind <- sample(seq_len(nrow(dat_demographic)), size = smp_size)
train <- dat_demographic[train_ind, ]
test <- dat_demographic[-train_ind, ]

# Grow the classification tree by defining the formula and constructing the tree
formula = outbreak ~ age+gender+race+job
tree <- rpart(formula, data=train, method="class")

# 	Draw an rpart plot with the number and percentage of observations, 
# with the leaves aligned at the bottom and without abbreviating values
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE, faclen=0)

### The tree that has been grown has only one node. It is necessary to tune the parameters

# After a few attempts, the command below has grown a promising tree
tree <- rpart(formula, data=train, method="class", control=rpart.control(cp=0.0005, minsplit=50))

# 	Draw an rpart plot with the number and percentage of observations, 
# with the leaves aligned at the bottom and without abbreviating values
prp(tree, extra=101, type=3, digits=4, fallen.leaves=TRUE, faclen=0)

# Based on the tree that has been grown against the training set, predict the outcome variable in the testing set
pred = predict(tree, test, type="class")

summary(pred)

### Output
###   N    Y 
###  66  1960

# Calculate the percentage of accurate predictions
mean(pred == test$outbreak)

### Output
### 0.6031589

# Print a table with the true negatives, false negatives, true positives and false positives
table(pred, test$outbreak)

### Output
### pred    N    Y
###    N   37   29
###    Y  775 1185


# Create a copy of the dataset to be used with Clustering
dat_demographic_cl = dat_demographic

# Remove the attribute that is to be predicted
dat_demographic_cl$outbreak <- NULL

# Convert the non-numeric attributes in numeric attributes
dat_demographic_cl$age <- as.numeric(dat_demographic_cl$age)
dat_demographic_cl$gender <- as.numeric(dat_demographic_cl$gender)
dat_demographic_cl$race <- as.numeric(dat_demographic_cl$race)
dat_demographic_cl$job <- as.numeric(dat_demographic_cl$job)

# Rescale variables to assure compatibility
dat_demographic_cl = scale(dat_demographic_cl)

# Compute internal validation measures about the clustering models obtained with K-means and PAM algorithms
intern <- clValid(dat_demographic_cl, 2, clMethods=c("kmeans","pam"), validation="internal")

# View the computed results, that show that K-means is the best algorithm
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
### kmeans Connectivity   0.0000
###        Dunn           0.0787
###        Silhouette     0.2518
### pam    Connectivity  12.3933
###        Dunn           0.0724
###        Silhouette     0.2114
### 
### Optimal Scores:
### 
###              Score  Method Clusters
### Connectivity 0.0000 kmeans 2       
### Dunn         0.0787 kmeans 2       
### Silhouette   0.2518 kmeans 2

# Set the seed for the random number generators, as an assurance that the results will be reproducible
set.seed(1)

# Apply the kmeans clustering algorithm
kmeans.result <- kmeans(dat_demographic_cl, centers=2, nstart=25)

# Check how many observations are in each cluster, and which are correct and incorrect
table(dat_demographic$outbreak, kmeans.result$cluster)

### Output, which gives that N should correspond to cluster 2 and Y to cluster 1 (3227/6076 correct observations	)
###      1    2
### N 1395  949
### Y 2278 1454

# print some info about the output of kmeans
print(kmeans.result)

### Output
### K-means clustering with 2 clusters of sizes 178, 186
### 
### Cluster means:
###     temp_avg   humidity   rainfall
### 1  0.6501040 -0.7022270 -0.6913675
### 2 -0.6221425  0.6720236  0.6616313
### 
### Clustering vector:
###   [1] 2 2 1 2 1 1 1 2 2 1 1 1 2 2 2 2 2 2 2 1 1 1 1 1 2 2 2 2 2 2 1 1 2 1 2 2 2 2 2 1 2 2 1 2 2 2 2 2 2 2 2 1 1 1
###  [55] 1 1 1 1 1 1 1 1 1 2 2 2 1 1 2 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1 1 1 1 2 2 2 1 1 2 2 2 2 1 2 1 1 1 1 2 1 1 1 1
### [109] 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 2 1 2 2 1 2 2 2 2 2 2 2 2 2 1 2 1 1 1
### [163] 2 1 1 1 1 2 2 2 2 2 2 2 2 1 2 1 1 2 2 1 1 1 1 2 2 2 1 2 2 2 1 1 2 1 1 2 2 2 2 2 2 2 2 2 2 1 2 2 2 1 1 1 1 2
### [217] 2 1 2 2 2 2 1 2 2 1 1 2 1 2 2 2 1 1 1 1 2 2 2 1 2 2 2 2 2 1 2 1 2 2 2 2 2 2 2 1 2 2 2 2 1 2 2 1 2 1 1 1 2 1
### [271] 2 2 2 2 2 2 2 1 1 1 2 2 2 1 2 2 2 2 2 2 1 2 1 2 2 2 1 1 1 1 2 2 2 1 2 2 2 2 1 2 1 1 1 1 1 1 1 1 1 1 2 2 2 1
### [325] 2 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 2 2 2 1 2 1 1 1 2 2 2 2 2 2 2 2 1 2 1
### 
### Within cluster sum of squares by cluster:
### [1] 304.3566 299.1399
###  (between_SS / total_SS =  44.6 %)
