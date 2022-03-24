
import tensorflow as tf
from keras.models import Model
from keras.layers import Input

import pathlib
import numpy as np
import matplotlib.pyplot as plt






if __name__ == '__main__':
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

