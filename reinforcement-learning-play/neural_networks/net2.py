from keras import callbacks, metrics, datasets, losses
import matplotlib.pyplot as plt
import numpy as np
from networks import make_model2

((train_x1, train_y1), (test_x1, test_y1)) = datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(test_x1.shape)
print(test_x1.dtype)

# Normalize
train_x1 = train_x1 / 255.0
test_x1 = test_x1 / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_x1[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_y1[i]])
plt.show()

train_x2 = np.array(train_x1)
test_x2 = np.array(test_x1)
train_y2 = np.array(train_y1)
test_y2 = np.array(test_y1)

def is_top_clothing(label):
    return class_names[label] in {
        'T-shirt/top', 'Shirt', 'Pullover', 'Dress', 'Coat'
    }

for i in range(train_y2.shape[0]):
    train_y2[i] = 1 if is_top_clothing(train_y2[i]) else 0
for i in range(test_y2.shape[0]):
    test_y2[i] = 1 if is_top_clothing(test_y2[i]) else 0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_x2[i], cmap=plt.cm.binary)
    plt.xlabel(train_y2[i])
plt.show()

model2 = make_model2()
model2.summary()

model2.compile(
    optimizer="adam",
    # we can see `from_logits=True` as a sigmoid function extra layer
    # the output is transformed from any reals into the space of bernulli probability
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model2.fit(
    train_x2,
    train_y2,
    epochs=12,
    # tensorboard --logdir logs/run2
    callbacks=callbacks.TensorBoard(log_dir="logs/run2")
)

test_logits_y2 = model2.predict(test_x2)
pred_y2 = np.where(test_logits_y2.reshape(-1) >= 0, 1.0, 0.0)
m = metrics.BinaryAccuracy()
m.update_state(test_y2, pred_y2)
accuracy = m.result()

print("Accuracy: ", accuracy)