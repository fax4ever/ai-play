from networks import make_model3
from keras import callbacks, metrics, datasets, losses

((train_x1, train_y1), (test_x1, test_y1)) = datasets.fashion_mnist.load_data()

model3 = make_model3()
model3.summary()

model3.compile(
    optimizer="adam",
    # we can see `from_logits=True` as a softmax function extra layer
    # the output is transformed from any reals 
    # into the space of a categorical vector of probabilities
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model3.fit(
    train_x1,
    train_y1,
    epochs=12,
    # tensorboard --logdir logs/run3
    callbacks=callbacks.TensorBoard(log_dir="logs/run3")
)

pred_logits_y1 = model3.predict(test_x1)
m = metrics.SparseCategoricalAccuracy()
m.update_state(test_y1, pred_logits_y1)
accuracy = m.result()

print("Accuracy: ", accuracy)