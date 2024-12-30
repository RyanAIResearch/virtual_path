import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load('training_metrics.npz')
epochs = data['epochs']
ppl = data['ppl']
accuracy = data['accuracy']

# Plot PPL
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, ppl, label='PPL')
plt.xlabel('Epoch')
plt.ylabel('PPL')
plt.title('PPL over Epochs')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()