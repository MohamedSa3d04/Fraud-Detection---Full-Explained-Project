from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np

# Print Basic Metrics (Accuracy, balanced-Accuracy, Confusion Matrix)
def basic_metrics(y_true, y_predicted):
    tn, fp, fn, tp = confusion_matrix(y_true, y_predicted).ravel()
    p = tp + fn
    n = tn + fp
    print(f'tn = {tn} - fn = {fn} - tp = {tp} - fp = {fp}')

    bAccuracy = balanced_accuracy_score(y_true, y_predicted)
    normal_Accuracy = (tn + tp) / (p + n)
    print(f'Balanced Accuracy = {bAccuracy} - Accuracy = {normal_Accuracy}')

    f1 = f1_score(y_true, y_predicted)
    recall = recall_score(y_true, y_predicted)
    precision = precision_score(y_true, y_predicted)
    print(f'F1 = {f1} - Recall = {recall} - Precision = {precision}')
    print(classification_report(y_true, y_predicted))

# Get Best Thershold for the currect values
def get_best_thershold(y_true, probas_predicted):
    precisions, recalls, thrsholds = precision_recall_curve(y_true, probas_predicted)
    precisions, recalls = precisions[:-1] , recalls[:-1]
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    value, index = np.max(f1_scores), np.argmax(f1_scores)

    best_threshold = thrsholds[index]

    print(f'Best threshold is {best_threshold} With F1 = {value} Precision = {precisions[index]} Recall =  {recalls[index]}\n')
    return best_threshold

# Evaluating the model with a specific thereshold (reutrning the best one!)
def evaluate_model(model, xVal, yVal, threshold = 0.5):
    probas = model.predict_proba(xVal)[:, 1]
    predicted = (probas >= threshold).astype(int)
    print("Metrices For Validation: ")
    basic_metrics(yVal, predicted)
    best_thershold = get_best_thershold(yVal, probas)
    return best_thershold


# Let's evaluate the model useing Curves
def evaluate_model_with_curves(model, xVal, yVal):
    probas = model.predict_proba(xVal)[:, 1]
    precisions, recalls, _ = precision_recall_curve(yVal, probas)
    fpr, tpr, _ = roc_curve(yVal, probas)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC = {roc_auc}')

    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.grid()

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, label='Precision-Recall curve (area = {:.2f})'.format(auc(recalls, precisions)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()
  


    
