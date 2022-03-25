
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Input
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import cv2
import tempfile


def format_image(image, label):
  image = tf.image.resize(image, IMAGE_SIZE) / 255.0
  return  image, label

def remove_badlyEncoded_images():
    num_skipped = 0
    for folder_name in ("cats_dataset/", "dogs_dataset/"):
        folder_path = os.path.join('/home/progforce/tensorflow_datasets/', folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10) # https://keras.io/examples/vision/image_classification_from_scratch/
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

def transformationTFDS_in_ImageDirs():
    ds = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
    good_count = bad_count = cats = dogs = 0
    cats_fls = '/home/progforce/tensorflow_datasets/createDataSet1/cats_dataset/'
    dogs_fls = '/home/progforce/tensorflow_datasets/createDataSet1/dogs_dataset/'

    for image, label in ds:
        try:
            plt.imshow(image)
            # plt.axis('off')
            # plt.title("label: " + str(label))
            # plt.show()
            if label == 0:
                cats += 1
                plt.savefig(cats_fls + "cats_" + str(cats) + ".jpg")  # save the figure to file
                # with open(cats_fls + "cats_" + str(cats) + ".jpg", "wb") as cat_ds:
                #     cat_ds.write(image)
            else:
                dogs += 1
                plt.savefig(dogs_fls + "dogs_" + str(cats) + ".jpg")  # save the figure to file
                # with open(dogs_fls + "dogs_" + str(dogs) +".jpg", "wb") as dog_ds:
                #     dog_ds.write(image)

            good_count += 1
            print('good count: ' + str(good_count))
        except:
            bad_count += 1
            print('bad count' + str(bad_count))
        plt.clf()
    print('cats =: ' + str(cats))
    print('dogs =: ' + str(dogs))
    print('bad images = ' + str(bad_count))

def generate_dataset():
    image_size = (pixels, pixels)
    batch_size = BATCH_SIZE

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/home/progforce/tensorflow_datasets/createDataSet1/",
        validation_split=0.2,
        subset="training",
        seed=13,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "/home/progforce/tensorflow_datasets/createDataSet1/",
        validation_split=0.2,
        subset="validation",
        seed=13,
        image_size=image_size,
        batch_size=batch_size,
    )

    # The saved data set occupies about 15 GB, although the original images are only about 500 MB !!!
    # path_train_ds = os.path.join(tempfile.gettempdir(), "saved_trainDS")
    # path_valid_ds = os.path.join(tempfile.gettempdir(), "saved_validDS")
    # tf.data.experimental.save(train_ds, path_train_ds)
    # tf.data.experimental.save(train_ds, path_valid_ds)


if __name__ == '__main__':

    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("Hub version: ", hub.__version__)
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

    module_selection = ("inception_v3", 299, 2048)  # @param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
    handle_base, pixels, FV_SIZE = module_selection
    MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
    IMAGE_SIZE = (pixels, pixels)
    BATCH_SIZE = 32
    print("Using {} with input size {} and output dimension {}".format(MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

    tfds.disable_progress_bar() # it is not necessary

    generate_dataset()

    # new_dataset = tf.data.experimental.load(path)
    # for elem, label in new_dataset:
    #     plt.imshow(elem)
    #     plt.axis('off')
    #     plt.title("label: " + str(label))
    #     plt.show()

    # entire_set, info = tfds.load(
    #     'cats_vs_dogs',
    #     with_info=True,
    #     as_supervised=True,
    # )









    """
    # it'll divide dataset in ratio 80%, 10%, 10% (train, validation, test)
    (train_examples, validation_examples, test_examples), info = tfds.load(
        'cats_vs_dogs',
        split=['train[80%:]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes

    #  print(str(info))  // A large set of images of cats and dogs. There are 1738 corrupted images that are dropped.

    # train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
    # validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
    # test_batches = test_examples.map(format_image).batch(1)

    """








    """
    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]   # y = 2x - 1

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(x, y, epochs=200, verbose=1)

    export_dir = 'saved_TFmodel/1'
    tf.saved_model.save(model, export_dir)

    # Convert the SavedModel to TFLite model
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    tflite_model_file = pathlib.Path('model.tflite')
    tflite_model_file.write_bytes(tflite_model)

    # alternate file saving option
    with open("modelAlternative.tflite", "wb") as f:
        f.write(tflite_model)

#========================= Initialize the TFLite interpreter to try it out ========================>
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the TensorFlow Lite model on random input data
    input_shape = input_details[0]['shape']
    inputs, outputs = [], []
    tflite_res = []
    for _ in range(100):
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        tflite_res.append(tflite_results[0][0])

        # Test the TensorFlow model on random input data.
        tf_results = model(tf.constant(input_data))
        output_data = np.array(tf_results)

        inputs.append(input_data[0][0])
        outputs.append(output_data[0][0])

    plt.figure(figsize=(10, 6))
    plt.title("TensorFlow model")
    plt.plot(inputs, outputs, 'r')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("TFLite model")
    plt.plot(inputs, tflite_res, 'g')
    plt.show()
    """

