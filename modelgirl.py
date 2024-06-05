import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

# Define the callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') >= 0.70:
            print('\nReached 70% validation accuracy, canceling training!')
            self.model.stop_training = True

def create_and_train_model():
    # Set path to dataset
    train_dir = 'C:/Users/user/Downloads/Data settt/Girl/faceshape-master/published_dataset' 
    validation_dir = 'C:/Users/user/Downloads/Data settt/Girl/faceshape-master/published_dataset' 

    # Using ImageDataGenerator for preprocessing and augmentation
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Loading images from folder
    train_generator = training_datagen.flow_from_directory(
        directory=train_dir,
        batch_size=64,
        class_mode='categorical',
        target_size=(150, 150)
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(
        directory=validation_dir,
        batch_size=16,
        class_mode='categorical',
        target_size=(150, 150)
    )

    # Determine the number of classes
    num_classes = len(train_generator.class_indices)

    # Define the model architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Instantiate the callback
    callbacks = myCallback()

    # Train the model
    history = model.fit(
        train_generator,
        epochs=800,
        validation_data=validation_generator,
        callbacks=[callbacks]
    )

    return model, history

if __name__ == '__main__':
    model, history = create_and_train_model()

    # Save the model
    model.save('model.h5')