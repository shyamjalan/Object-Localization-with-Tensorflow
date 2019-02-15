# Object-Localization-with-Tensorflow
This project was completed under the Flipkart Grid Challenge on Dare2Compete.com. Our team was able to gain 87% accuracy and an All India Rank 97 and Rank 1 in our college, BIT Mesra

# Tools AND Libraries Used: 
1. os
2. numpy
3. tensorflow
4. pandas
5. PIL - Image
6. matplotlib.pyplot

We have divide the project into three parts : 
1. Creating dataset i.e. Preprocessing the input images. (create_dataset.ipynb)
2. Training the model. (training.ipynb)
3. Testing Phase. (testing.ipynb)

Please run the source files in this order, and put a folder named 'images' with all the training and testing images, and the two files 'training.csv' and 'testing.csv' in the folder having the source files.

# 1. Creating Dataset :

First, we are reading the CSV file given for training into a pandas DataFrame. Then, we open the Images via PIL and resize them into the default input size of our CNN i.e. (227*227) in RGB format. These images are stored in a numpy array. X_train is obtained. Now, for Y_train, we scaled the bounding boxes as per our new size and stored them into a numpy array. We, are then saving these files locally to avoid running this code redundantly for training/testing. Same preprocessing is done for the input images (X_test).
Finally, these data get saved as follows :
1. X_train - 'object_localization.npy'
2. Y_train - 'object_localization_y.npy'
3. X_test - 'object_localization_x_test.npy'

# 2. Training the model :

We are using AlexNet CNN Architecture in order to train our model. AlexNet is used as architecture. 5 convolution layers and 3 Fully Connected Layers with 0.5 Dropout Ratio (To avoid overfitting). 60 million Parameters. The training data is divided into two parts - Training data (80%) and Cross-Validation Data (20%). We used mini-batch to train our model with batch-size 8 (taken after testing accuracy). We used TensorFlow to define all the layer of our CNN Model. We used Mean Squared Error as loss and ran AdamOptimizer to minimize the same. Now, we randomly initialized the variables and started the iterations (epochs=150). In each epoch, we are calculating the cost and using the parameters to cross-validate. After completion, the trained model is saved as 'CNN_final.ckpt'.

# 3. Testing Phase :

We use the 'X_test' from Step 1 and 'CNN_final.ckpt' from Step 2 in this step. We use the same CNN Model as in training to run forward propagation. We are storing the images in a list and running testing for 1000 images at a time due the the memory limitations of our computer. All the predicted Co-ordinates of bounding boxes are then recieved in the list. Now, are using matplotlib.pyplot in order to draw the bounding box predctions on few test-images. Now, the list is converted to a Pandas DataFrame with respective labels. The data is rescaled back to (640*480) and concatenated the names column from test.csv and then saved the final predictions into a CSV file 'y_pred.csv'. 

In order To improve the accuracy /predictions we tuned the hyperparameters and finally our Model Gave satisfactory results with the values Mentioned below:

Hyperparameters:
Number of Epochs: 150
Mini-Batch Size :8
Learning rate:0.00001
validation size:20



	
