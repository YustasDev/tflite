import pdb
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
import pickle
import shelve
from keras.preprocessing.image import ImageDataGenerator
import PIL
import PIL.Image
import sklearn
from sklearn.model_selection import train_test_split
import splitfolders
from tqdm import tqdm



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
    cats_fls = '/home/progforce/tensorflow_datasets/createDataSet1/cat/'
    dogs_fls = '/home/progforce/tensorflow_datasets/createDataSet1/dog/'

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


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate_dataset(path_to_imagesDir):
    image_size = IMAGE_SIZE
    batch_size = BATCH_SIZE

    #train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     path_to_imagesDir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=13,
    #     image_size=image_size,
    #     batch_size = 1
    # )
    #
    # val_ds = tf.keras.utils.image_dataset_from_directory(
    #     path_to_imagesDir,
    #     validation_split=0.2,
    #     subset="validation",
    #     seed=13,
    #     image_size=image_size,
    # )

    entire_ds = tf.keras.utils.image_dataset_from_directory(
        path_to_imagesDir,
        subset="train",
        seed=13,
        image_size=image_size,
        batch_size = 1
    )

    class_names = entire_ds.class_names
    print('class_names: ' + str(class_names))

    # by default batch_size = 32
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(30):
    #         ax = plt.subplot(6, 5, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    #         plt.axis("off")
    #     plt.show()

    # IF batch_size = 1
    # for image, label in train_ds.take(1):
    #     image = np.squeeze(image)
    #     print("Image shape: ", image.shape)
    #     print("Label: ", label.numpy())
    #     plt.imshow(image.astype("uint8"))
    #     plt.axis("off")
    #     plt.show()

    # https://www.tensorflow.org/tutorials/load_data/images
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    #return train_ds, val_ds
    return entire_ds


def image_example(image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape

    feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

    return tf.train.Example(features=tf.train.Features(feature=feature))
    # pdb.set_trace()

def split_folders(input_folder, output_folder):
    splitfolders.ratio(input_folder, output = output_folder, seed = 13, ratio=(0.7, 0.2, 0.1), group_prefix=None)

def representative_data_gen():
    for input_value, _ in test_batches.take(100):
        yield [input_value]


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)




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

    path_to_imagesDir = "/home/progforce/tensorflow_datasets/createDataSet1/"


#=================== It works, but my computer can't handle it ===========================================>
    # Save dataset
    #savingDS_dir = "/home/progforce/tensorflow_datasets/savingDS/"
    #tf.data.experimental.save(train_ds, savingDS_dir, compression='GZIP')
    # with open(savingDS_dir + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
    #     pickle.dump(train_ds.element_spec, out_)

# ========================= Load dataset ========================================================>
#     with open(savingDS_dir + '/element_spec', 'rb') as in_:
#         es = pickle.load(in_)
#     loaded = tf.data.experimental.load(savingDS_dir, es, compression='GZIP')
# ==========================================================================================================>
    input_folder = "/home/progforce/tensorflow_datasets/createDataSet1/"
    output_folder = "/home/progforce/tensorflow_datasets/createDataSet1_Split/"
    #split_folders(input_folder, output_folder)

#===================== create the necessary datasets ==============================>
    train_ds = tf.keras.utils.image_dataset_from_directory(
        output_folder + '/train',
        image_size=IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        seed = 13
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        output_folder + '/val',
        image_size=IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        seed = 13
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        output_folder + '/test',
        image_size=IMAGE_SIZE,
        batch_size = 1,
        seed = 13
    )

    train_batches = train_ds.map(format_image).prefetch(1)
    validation_batches = valid_ds.map(format_image).prefetch(1)
    test_batches = test_ds.map(format_image)
    num_classes = 2

    # Check shape
    for image_batch, label_batch in train_batches.take(1):
        print(str(image_batch.shape))
        print(str(label_batch))

    for image_batch, label_batch in test_batches.take(1):
        print(str(image_batch.shape))

#================== create and fit model ==========================================>
    """
    do_fine_tuning = False  # @param {type:"boolean"}

    feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                       input_shape=IMAGE_SIZE + (3,),
                                       output_shape=[FV_SIZE],
                                       trainable=do_fine_tuning)


    print("Building model with", MODULE_HANDLE)
    model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(num_classes)
    ])
    model.summary()

    # @title (Optional) Unfreeze some layers
    NUM_LAYERS = 7  # @param {type:"slider", min:1, max:50, step:1}
    if do_fine_tuning:
        feature_extractor.trainable = True
        for layer in model.layers[-NUM_LAYERS:]:
            layer.trainable = True
    else:
        feature_extractor.trainable = False

    if do_fine_tuning:
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.002, momentum=0.9),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    else:
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

    EPOCHS = 5
    history = model.fit(train_batches,
                     epochs=EPOCHS,
                     validation_data=validation_batches)
    """
    CATS_VS_DOGS_SAVED_MODEL = "exp_saved_model"
    #tf.saved_model.save(model, CATS_VS_DOGS_SAVED_MODEL)

#=============== convert saved model into TFLite model ===============================>
    converter = tf.lite.TFLiteConverter.from_saved_model(CATS_VS_DOGS_SAVED_MODEL)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    tflite_model_file = 'converted_model.tflite'

    with open(tflite_model_file, "wb") as f:
        f.write(tflite_model)

#============================== test the TFLite model using the Python Interpreter ===========>
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=tflite_model_file)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    #Gather results for the randomly sampled test images
    predictions = []

    test_labels, test_imgs = [], []
    for img, label in tqdm(test_batches.take(10)):
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        predictions.append(interpreter.get_tensor(output_index))

        test_labels.append(label.numpy()[0])
        test_imgs.append(img)

    class_names = ['cat', 'dog']

    plt.figure(figsize=(10, 10))
    for index in range(10):
        z = plt.subplot(2, 5, index+1)
        plot_image(index, predictions, test_labels, test_imgs)
    plt.show()

# ========================= Game Over )) ==============================================>
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

