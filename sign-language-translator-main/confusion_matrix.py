# confusion_matrix.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sample data (replace with your actual predictions)
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Example true labels
y_pred = [0, 1, 1, 0, 2, 1, 0, 1, 2, 0]  # Example predicted labels

# ASL signs (replace with your 50 classes)
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
               'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
               'U', 'V', 'W', 'X', 'Y', 'Z', '1', '2', '3', '4', 
               '5', '6', '7', '8', '9', '10', '11', '12', '13', 
               '14', '15', '16', '17', '18', '19', '20', '21', 
               '22', '23', '24', '25', '26', '27', '28', '29', 
               '30', '31', '32', '33', '34', '35', '36', '37', 
               '38', '39', '40', '41', '42', '43', '44', '45', 
               '46', '47', '48', '49', '50']

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualization
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save high-quality image
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as confusion_matrix.png")
# confusion_matrix.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Sample data (replace with your actual predictions)
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
y_pred = [0, 1, 1, 0, 2, 1, 0, 1, 2, 0]

# ASL signs (replace with your 50 classes)
class_names = ['A', 'B', 'C']  # Example for 3 classes

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualization
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save high-quality image
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as confusion_matrix.png")
