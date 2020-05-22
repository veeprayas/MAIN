import os
import pickle
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from PIL import *
import tensorflow as tf
import numpy as np
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_num_files(path):
  if not os.path.exists(path):
    return 0
  return sum([len(files) for r, d, files in os.walk(path)])

def get_num_subfolders(path):
  if not os.path.exists(path):
    return 0
  return sum([len(d) for r, d, files in os.walk(path)])


def create_img_generator():
  return  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )


Image_width, Image_height = 299, 299
Training_Epochs = 6
Batch_Size = 32
Number_FC_Neurons = 1024

train_dir = (r'C:\Users\Asus\Downloads\keras-deep-learning\06\demos\demos\Transfer_learning\hymenoptera_data/train')
validate_dir = (r'C:\Users\Asus\Downloads\keras-deep-learning\06\demos\demos\Transfer_learning\hymenoptera_data/valid')
num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)
num_epoch = Training_Epochs
batch_size = Batch_Size


train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

#   Connect the image generator to a folder contains the source images the image generator alters.
#   Training image generator
train_generator = train_image_gen.flow_from_directory(
  train_dir,
  target_size=(Image_width, Image_height),
  batch_size=batch_size,
  seed = 42    #set seed for reproducability
)

#   Validation image generator
validation_generator = test_image_gen.flow_from_directory(
  validate_dir,
  target_size=(Image_width, Image_height),
  batch_size=batch_size,
  seed=42       #set seed for reproducability
)


InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
print('Inception v3 base model without last FC loaded')



x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(Number_FC_Neurons, activation='relu')(x)        # new FC layer, random init
predictions = Dense(num_classes, activation='softmax')(x)
#x = GlobalAveragePooling2D(name='avg_pool')(x)
#x = Dropout(0.4)(x)
#predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)
model.summary()

#print (model.summary())

print ('\nPerforming Transfer Learning')
  #   Freeze all layers in the Inception V3 base model
for layer in InceptionV3_base_model.layers:
  layer.trainable = False
#   Define model compile for basic Transfer Learning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#pickle.dump(model, open('model.pkl','wb'))
# Fit the transfer learning model to the data from the generators.
# By using generators we can ask continue to request sample images and the generators will pull images from
# the training or validation folders and alter them slightly
history_transfer_learning = model.fit_generator(
  train_generator,
  epochs=num_epoch,
  steps_per_epoch = num_train_samples // batch_size,
  validation_data=validation_generator,
  validation_steps = num_validate_samples // batch_size,
  class_weight='auto')


pickle.dump(model, open('model.pkl', 'wb'))

# Loading model to compare the results
#model1 = pickle.load(open('model.pkl', 'rb'))



#from keras.models import load_model



#video = cv2.VideoCapture(0)

#while True:
#        _, frame = video.read()

        #Convert the captured frame into RGB
#        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
#        im = im.resize((299,299))
#        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3
 #       img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
 #       prediction = (loaded_model.predict(img_array)[0][0])
 #       pred = int(prediction.argmax())
        
  #      if pred == 0:
   #             print("ant")
    #    else:
     #           print("bee")
      #  cv2.imshow("Capturing", frame)
      #  key=cv2.waitKey(1)
      #  if key == ord('q'):
       #         break
#video.release()
#cv2.destroyAllWindows()