import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import accuracy_score
import sys
import os

import mlflow

from helper.eye_dataset import load as eyedataload


os.makedirs("models", exist_ok=True)
model_name = input('Enter model name')

#Loading
print('\n Loading Training Dataset & DataLoaader')
train_dataset = eyedataload('splitDataset/train', 'train')
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print('\n Loading Val Dataset & DataLoaader')
val_dataset = eyedataload('splitDataset/val', 'val')
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

#Training
if(len(sys.argv) > 1):
    model = resnet18(weights=None)
    class_set_size = 5
    model.fc = nn.Linear(model.fc.in_features, class_set_size)
    model.load_state_dict(torch.load(sys.argv[1]))
else:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    class_set_size = 5
    model.fc = nn.Linear(model.fc.in_features, class_set_size)

criterion = nn.CrossEntropyLoss()

learning_rate, weight_decay, num_epochs = 1e-5, 5e-4, 20
quarter_epoch = num_epochs // 4
control = 5

params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
optimizer = optim.Adam([{'params':params_1x}, {'params': model.fc.parameters(), 'lr': learning_rate*10}], lr=learning_rate, weight_decay=weight_decay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prev_valid_loss = float('inf')
convergence_counter = 0

mlflow.start_run()
mlflow.log_param('model_name', model_name)
mlflow.log_param("learning rate", learning_rate)
mlflow.log_param("num epochs", num_epochs)
mlflow.log_param("weight decay", weight_decay)

print(f'Entering training loop (device => {device}, epochs => {num_epochs})...\n')

try:
    for current_epoch in range(num_epochs):
        model.train()
        run_loss = 0.0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            run_loss = run_loss + loss.item()

        print(f"Epoch {current_epoch+1} of {num_epochs}, Loss: {run_loss/len(train_dataloader)}")
        mlflow.log_metric("train_loss", run_loss/len(train_dataloader), current_epoch)

        model.eval()
        valid_loss = 0.0

        all_preds = []
        all_labels = []
        correct = total = 0

        with torch.no_grad():
            for (inputs, labels) in enumerate(val_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss = valid_loss + loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        print(f"Validation Loss: {valid_loss/len(val_dataloader)}")
        mlflow.log_metric("valid_loss", valid_loss/len(val_dataloader), current_epoch)

        if (current_epoch + 1) % quarter_epoch == 0:
            accuracy = accuracy_score(all_labels, all_preds)
            print(f"Validation Accuracy: {accuracy}")
            mlflow.log_metric("validation accuracy", accuracy)
        print()

        if(valid_loss >= prev_valid_loss):
            convergence_counter+=1
            if(convergence_counter >= control):
                print("Converged. Stopping training.")
                break
        else:
            convergence_counter = 0
        prev_valid_loss = valid_loss

    mlflow.pytorch.save_state_dict(model.state_dict(), f"models/{model_name}.pth")
    mlflow.pytorch.save_model(torch.jit.script(model), f"models/{model_name}_traced.pth")
    print("Model saved successfully.")

    mlflow.pytorch.log_model(model, "model")
    mlflow.pytorch.log_model(torch.jit.script(model), "scripted_model")
    mlflow.end_run()

except Exception as e:
    print(f'An error occured: {str(e)}')
    mlflow.pytorch.save_state_dict(model.state_dict(), f'{model_name}_error.pth')
    mlflow.pytorch.save_model(torch.jit.script(model), f'{model_name}_traced_error.pth')
    print('Model saved due to error.')

torch.cuda.empty_cache()