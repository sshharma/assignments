import os
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, Input, MaxPooling2D, Dropout, Flatten, Dense, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import argparse
import pandas as pd
import csv
import keras_tuner as kt  # Import Keras Tuner for hyperparameter tuning

def build_model(hp):
    # Hyperparameters to tune
    filter_size = hp.Choice('filter_size', [3, 5])
    num_conv_blocks = hp.Int('num_conv_blocks', min_value=3, max_value=6, step=1)
    initial_filters = hp.Choice('initial_filters', [32, 64])
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    dilation_rate = hp.Int('dilation_rate', min_value=1, max_value=3)
    activation = hp.Choice('activation', ['relu', 'tanh'])
    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    l2_reg = hp.Float('l2_regularization', min_value=0.0, max_value=0.01, step=0.005)
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=64)
    use_dilation = hp.Boolean('use_dilation')

    model = Sequential()
    model.add(Input(shape=(args.img_size, args.img_size, image_channel)))

    filters = initial_filters
    for i in range(num_conv_blocks):
        if use_dilation and i == num_conv_blocks - 1:
            model.add(Conv2D(filters, (filter_size, filter_size), dilation_rate=dilation_rate, kernel_regularizer=l2(l2_reg)))
        else:
            model.add(Conv2D(filters, (filter_size, filter_size), kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rate))
        filters *= 2  # Double the number of filters for the next block

    model.add(Flatten())
    model.add(Dense(dense_units, kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(dropout_rate))

    model.add(Dense(5, activation='softmax'))

    # Optimizer selection
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main()-> None:
    global args, image_channel  # Make args and image_channel accessible in build_model
    # Read user input
    parser = argparse.ArgumentParser(description='Dilated Convolutional Neural Network with Hyperparameter Tuning')
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=30)
    parser.add_argument('--img_size', type=int, help='Image size', default=224)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
    parser.add_argument('--gpu_id', type=int, help='GPU ID', default=0)
    args = parser.parse_args()

    device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'
    with tf.device(device):
        print(f"Running on {device}")

    # Parameters
    date_time = datetime.datetime.now()
    date_str = date_time.strftime('%Y-%m-%d')
    time_str = date_time.strftime('%H%M%S')
    ep = args.epochs
    image_channel = 3

    # Create a directory to save the model
    model_dir = f'models/{date_str}_Dilated_CNN'
    os.makedirs(model_dir, exist_ok=True)

    # Load the data
    data_dir = "data-2/train/"
    categories = ["butterfly", "chicken", "dog", "horse", "spider"]

    # Initialize lists to store filenames and labels
    filenames = []
    labels = []

    # Iterate through the categories
    for category in categories:
        category_folder = os.path.join(data_dir, category)
        category_filenames = os.listdir(category_folder)
        for filename in category_filenames:
            filenames.append(os.path.join(category, filename))
            labels.append(category)

    assert len(filenames) == len(labels), "Mismatch between filenames and labels length"

    # Create DataFrame
    data = pd.DataFrame({
        'filename': filenames,
        'label': labels
    })

    # Data Preparation
    labels = data['label']
    X_train, X_val = train_test_split(data, test_size=0.2, stratify=labels, random_state=42)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=60,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1,
        shear_range=0.1,
        fill_mode='reflect',
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generators
    train_generator = train_datagen.flow_from_dataframe(
        X_train,
        directory=data_dir,
        x_col='filename',
        y_col='label',
        class_mode='categorical',
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size)
    )

    val_generator = test_datagen.flow_from_dataframe(
        X_val,
        directory=data_dir,
        x_col='filename',
        y_col='label',
        class_mode='categorical',
        batch_size=args.batch_size,
        target_size=(args.img_size, args.img_size),
        shuffle=False
    )

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        factor=0.2,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Hyperparameter Tuning using Keras Tuner
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='hyperparam_dir',
        project_name='dilated_cnn_tuning'
    )

    # Perform the search
    tuner.search(
        train_generator,
        validation_data=val_generator,
        epochs=ep,
        callbacks=[early_stopping, reduce_lr]
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. 
    Best number of convolutional blocks: {best_hps.get('num_conv_blocks')}
    Best initial filters: {best_hps.get('initial_filters')}
    Best filter size: {best_hps.get('filter_size')}
    Best dropout rate: {best_hps.get('dropout_rate')}
    Best dilation rate: {best_hps.get('dilation_rate')}
    Best activation function: {best_hps.get('activation')}
    Best optimizer: {best_hps.get('optimizer')}
    Best learning rate: {best_hps.get('learning_rate')}
    Best L2 regularization: {best_hps.get('l2_regularization')}
    Best dense units: {best_hps.get('dense_units')}
    Use dilation: {best_hps.get('use_dilation')}
    """)

    # Build the best model and train it
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=ep,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save the model with a unique name
    timestamp = date_time.strftime('%Y%m%d_%H%M%S')
    model_name = f'tuned_cnn_e{ep}_{timestamp}_model.keras'
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)

    # Collect configurations
    config = {
        'date': date_str,
        'time': time_str,
        'model_dir': model_dir,
        'model_name': model_name,
        'num_conv_blocks': best_hps.get('num_conv_blocks'),
        'initial_filters': best_hps.get('initial_filters'),
        'filter_size': best_hps.get('filter_size'),
        'dropout_rate': best_hps.get('dropout_rate'),
        'dilation_rate': best_hps.get('dilation_rate'),
        'activation': best_hps.get('activation'),
        'optimizer': best_hps.get('optimizer'),
        'learning_rate': best_hps.get('learning_rate'),
        'l2_regularization': best_hps.get('l2_regularization'),
        'dense_units': best_hps.get('dense_units'),
        'use_dilation': best_hps.get('use_dilation'),
        'epochs': ep,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'image_channel': image_channel,
        'train_accuracy': history.history['accuracy'][-1],
        'val_accuracy': history.history['val_accuracy'][-1],
        'train_loss': history.history['loss'][-1],
        'val_loss': history.history['val_loss'][-1],
        'total_parameters': model.count_params(),
    }

    # Write configurations to CSV
    config_file = os.path.join('configurations', 'configuration_hyperparameter_cnn_only.csv')
    os.makedirs('configurations', exist_ok=True)
    file_exists = os.path.isfile(config_file)

    with open(config_file, mode='a', newline='') as csvfile:
        fieldnames = list(config.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(config)

if __name__ == '__main__':
    main()
