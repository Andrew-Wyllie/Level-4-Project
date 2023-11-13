import matplotlib.pyplot as plt

epochs = [1,2,3,4,5,6,7,8,9,10]
loss_list = [
0.2,
0.24,
0.32,
0.53,
0.4,
0.54,
0.56,
0.59,
0.57,
0.42,
]
accuracy_list = [
0.2,
0.19,
0.355,
0.4175,
0.425,
0.5425,
0.71,
0.72,
0.8125,
0.58,
]

fig, axes = plt.subplots(1, 1, figsize=(12, 5))

plt.subplot(121)
plt.title("Loss")
plt.plot(epochs, loss_list)
plt.grid()
plt.ylabel('Loss')
plt.xlabel('Epochs')

plt.subplot(122)
plt.title("Accuracy")
plt.plot(epochs, accuracy_list)
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.axis((0.8,len(epochs)+0.2,0,1))

plt.show()

print("Done")