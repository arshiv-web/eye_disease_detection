import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from helper.eye_dataset import load as eyedataload
from helper.eye_dataset import loadSingle as eyedataloadSingle

def onboardModel(model_path):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 5)
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.eval()
    return model

def predict(model, dataloader):
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
    return true_labels, predicted_labels

def predictSingle(model_path, image_path):
    model = onboardModel(model_path)
    dataset = eyedataloadSingle(image_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs)
            probabilities = nn.functional.softmax(outputs, dim=1)
            all_class_probabilities = probabilities[0].tolist()
            conf, predicted = torch.max(probabilities, 1)
            predicted_class_idx = predicted.item()
            confidence = conf.item()
    
    with open('class2label.pk1', 'rb') as f:
        class2label = pickle.load(f)
    label2class = {label: class_name for class_name, label in class2label.items()}
    
    predictions = [
        (label2class[idx], all_class_probabilities[idx])
        for idx in range(len(all_class_probabilities))
    ]

    predictions.sort(key=lambda item: item[1], reverse=True)
    
    top_class_name = predictions[0][0]
    top_confidence = predictions[0][1]
    print(f'\nPredicted Class: {top_class_name} with probability {top_confidence*100:.2f}%')
    
    return predictions


def predictStudyDataset(model_path, study_dataset_path):
    model = onboardModel(model_path)
    test_dataset = eyedataload(study_dataset_path, 'study')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    true_labels, predicted_labels = predict(model, test_loader)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    # Print the metrics
    print(f'\nAccuracy: {100*accuracy:.4f} %')
    print(f'Precision: {100*precision:.4f} %')
    print(f'Recall: {100*recall:.4f} %')
    print(f'F1 Score: {100*f1:.4f} %')

    print(confusion_matrix(true_labels, predicted_labels))

if __name__ == '__main__':
    predictStudyDataset('models/arshiv_2.pth/state_dict.pth', 'splitDataset/study')
    # predictSingle('models/arshiv_2.pth/state_dict.pth', 'stage/55d275e7a3eac62a54ea405ec61a703a.jpg')