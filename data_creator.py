import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)


images = np.concatenate((train_images, test_images), axis=0)
labels = np.concatenate((train_labels, test_labels), axis=0)

fashion_df = pd.DataFrame(images)
fashion_df['label'] = labels


fashion_df.to_csv('fashion_mnist.csv', index=False)
print("CSV file created successfully.")
