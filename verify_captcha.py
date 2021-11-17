import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import cv2
from pathlib import Path
import os


# Desired image dimensions
img_width = 322
img_height = 100
max_length = 6

characters = ('G', 'b', '5', 'p', 'M', 'J', 'T', '3', '1', 'R', 'N', 'F', 'Y', 'e', 'P', 'K', '2', 'A', 'L', 'Z', 'h', 'I', '8', 't', 'a', 'D', '6', 'd', 'B', '7', 'r', 'E', 'n', 'S', 'l', '4', 'f', 'j', 'k', 'c', 'm', 'Q', 'y', 'H', 'i', '9')

# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

def encode_single_sample(img_path):
    # 1. Read image
    img = tf.io.read_file(img_path)
    # 2. Decode and convert to grayscale
    img = tf.io.decode_png(img, channels=1)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    return img

# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def main():
    # Pre-processing

    # data_dir = Path("../Image Test/")
    data_dir = Path("./Images")
    images = sorted(list(map(str, list(data_dir.glob("*.jfif")))))

    for img_path in images:
        img = cv2.imread(img_path)

        lower =(170, 170, 170) # lower bound for each channel
        upper = (255, 255, 255) # upper bound for each channel

        # create the mask and use it to change the colors
        mask = cv2.inRange(img, lower, upper)
        img[mask != 0] = [255,255,255]

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (thresh, gray) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        height, width = gray.shape[:2]

        blank_image = np.zeros((img_height,img_width), np.uint8)
        blank_image[:,:] = (255)

        l_img = blank_image.copy()

        x_offset = y_offset = 0
        # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
        l_img[y_offset:y_offset+height, x_offset:x_offset+width] = gray.copy()

        thresh = cv2.threshold(l_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            if cv2.contourArea(c) < 10:
                cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

        result = 255 - thresh
        
        file_to_rem = Path(img_path)
        file_to_rem.unlink()
        # cv2.imwrite(img_path.replace(".jfif","2.jpg"), detected_lines)
        cv2.imwrite(img_path.replace(".jfif",".png"), result)


    # model = keras.models.load_model('./Model')
    model = keras.models.load_model('./Model')

    images = sorted(list(map(str, list(data_dir.glob("*.png")))))


    for img_path in images:
        # img = cv2.imread(img_path, 0)
        #   (thresh, img) = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        #   cv2.imwrite(img_path, img)

        processed_img = encode_single_sample(img_path)

        preds = model.predict(tf.expand_dims(processed_img, axis=0))
        pred_texts = decode_batch_predictions(preds)
        
        # os.rename(img_path, "../Image Test/" + "".join(pred_texts[0]).replace("[UNK]","") + '.png')
        os.rename(img_path, "./Images/" + "".join(pred_texts[0]).replace("[UNK]","") + '.png')

        # remove file
        # file_to_rem = Path(img_path)
        # file_to_rem.unlink()

        # return verified captcha
        # return "".join(pred_texts[0]).replace("[UNK]","")

        print("pred_texts")
        print("".join(pred_texts[0]).replace("[UNK]",""))
        print("file name")
        print(img_path.split(os.path.sep)[-1].split(".png")[0])

main()