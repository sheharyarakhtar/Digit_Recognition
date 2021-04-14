library(keras)

#Import dataset mnist for training and testing
mnist <- dataset_mnist()

#2 lists pf 2. First list first and second part contains training and label and second for test
str(mnist)

#Seperate them to variable
trainx <- mnist$train$x
trainy <- mnist$train$y

testx <- mnist$test$x
testy <- mnist$test$y

table(trainy)
table(testy)

#plot images
par(mfrow = c(3,3))
for(i in 1:9) plot(as.raster(trainx[i,,], max=255))
par(mfrow = c(1,1))
trainx[5,,]
hist(trainx[5,,])
trainy

#FIVE
a <- c(1,12,36,48,66,101,133,139,146)
par(mfrow = c(3,3))
for(i in a) plot(as.raster(trainx[i,,], max = 255))
par(mfrow = c(1,1))

#reshape and rescale
##reshape linearise the matrix into one long row of 784 variables. 784 =28x28pixels
trainx <- array_reshape(trainx, c(nrow(trainx), 784))
str(trainx)
testx <- array_reshape(testx, c(nrow(testx), 784))
str(testx)

#Normalise the values by dividing them by 255, which the maximum value of each variable
trainx <- trainx/255
testx <- testx/255

#One hot encoding
##Forms a matrix array from a given array
trainy <- to_categorical(trainy, 10)
testy <- to_categorical(testy, 10)

head(trainy)

#Model
model <- keras_model_sequential()

model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 10, activation = 'softmax')
summary(model)

#Compile
model %>% 
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = 'accuracy')

#Fit model
history <- model %>% 
  fit(trainx,
      trainy,
      epochs = 12,
      batch_size = 32,
      validation_split = 0.2)

plot(history)
##Evaluation of model
model %>%  evaluate(testx, testy)
pred <- model %>%  predict_classes(testx)
table(Predicted = pred, Actual = mnist$test$y)

prob <- model %>% predict_proba(testx)
cbind(prob, Predicted_class = pred, Actual = mnist$test$y)[1:5,]




library(EBImage)
temp = list.files(pattern = '*.png')
mypic <- list()
for(i in 1:length(temp))mypic[[i]] <- readImage(temp[[i]])

par(mfrow = c(3,2))
for(i in 1:length(temp))plot(mypic[[i]])
par(mfrow = c(1,1))

#527680
#convert to greyscale
for(i in 1:length(temp))colorMode(mypic[[1]]) <- Grayscale

#make them white/blackbackground
for(i in 1:length(temp))mypic[[i]] <- 1-mypic[[i]]

#Resize the image
for(i in 1:length(temp))mypic[[i]] <- resize(mypic[[i]], 28, 28)
#Reshape the pictures
for(i in 1:length(temp))mypic[[i]] <- array_reshape(mypic[[i]], c(28,28,3))
str(mypic)

new <- NULL
for (i in 1:length(temp))new <- rbind(new, mypic[[i]])


newx <- new[,1:784]
newy <- c(5,2,7,6,8,0)

##model prediction
pred <- model %>%  predict_classes(newx)

pred
