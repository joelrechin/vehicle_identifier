import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
import seaborn as sn

#Defined several variables to be used later.
img_w, img_h = 200, 200
epochs = 1
batch_size = 10
num_classes = 3
input_shape = (img_w,img_h,1)

#Define the image locations for train, val, and test.
train_dir = r"vehicles\train"
test_dir = r"vehicles\test"
val_dir = r"vehicles\validation"

#Create the image generator for the test set.
train_datagen = ImageDataGenerator(
    width_shift_range= 0.25, 
    height_shift_range= 0.25,
    rescale = 1./255,
    zoom_range= 0.3)

#Create the image generator for the validation and test set.
test_val_datagen = ImageDataGenerator(rescale=1./255)

#Apply the image generators.
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_w,img_h),
    batch_size=20,
    color_mode='grayscale',
    class_mode='categorical')

val_gen = test_val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_w,img_h),
    batch_size=20,
    color_mode='grayscale',
    class_mode='categorical')

test_gen = test_val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_w,img_h),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False)

#Define parameters to easily adjust for optimization.
filters_1 = 16
filters_2 = 32
filters_3 = 64
neurons_dense_1 = 256
neurons_dense_2 = 128
drop_out_1 = 0.3
drop_out_2 = 0.3

#Define, compile and fit the CNN model.
model = Sequential()
model.add(Conv2D(filters_1, (3, 3),strides=(1,1),input_shape=input_shape, name='Conv2D_1'))
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Conv2D(filters_2, (3, 3), strides=(1,1),padding='valid', name='Conv2D_2'))
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Conv2D(filters_3, (3, 3), strides=(1,1),padding='valid', name='Conv2D_3'))
model.add(Activation(activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
model.add(Flatten())
model.add(Dense(neurons_dense_1))
model.add(Activation(activation='relu'))
model.add(Dropout(drop_out_2))
model.add(Dense(neurons_dense_2))
model.add(Activation(activation='relu'))
model.add(Dropout(drop_out_2))
model.add(Dense(num_classes, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=epochs, batch_size=batch_size)

#Make predictions on the test set of data.
predicted = model.predict(test_gen,batch_size=batch_size)

#View results using a confusion matrix.
y_pred = np.argmax(predicted,axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_gen.classes, y_pred))
print('Classification Report')
target_names = ['Midsize', 'Tesla', 'Truck']
print(classification_report(test_gen.classes, y_pred, target_names=target_names))

#Plot Heat Map
df_matrix = pd.DataFrame(confusion_matrix(val_gen.classes, y_pred), columns=['Midsize', 'Tesla', 'Truck'], index=['Midsize', 'Tesla', 'Truck'])
plt.figure(figsize= (10,7))
sn.heatmap(df_matrix, annot=True,cmap='Greens', fmt='g')
plt.show()








    



             





