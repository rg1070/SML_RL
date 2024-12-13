import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the updated SimpleNN_3L with appropriate input size
class SimpleNN_3L(nn.Module):
    def __init__(self, input_size, L1, L2, L3):
        super(SimpleNN_3L, self).__init__()
        self.fc1 = nn.Linear(input_size, L1)
        self.fc2 = nn.Linear(L1, L2)
        self.fc3 = nn.Linear(L2, L3)
        self.fc4 = nn.Linear(L3, 1)  # Output layer for regression

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Training function for 3-layer model
def train_3L(model, criterion, optimizer, X_train, y_train, batch_size, num_epochs, device):
    model.train()
    losses = []
    best_loss = float('inf')
    best_model_state = None
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size].to(device)
            targets_batch = y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / (len(X_train) // batch_size)
        losses.append(avg_epoch_loss)

        # Checkpointing
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()

    return losses, best_model_state

# Function to train the model on your data
def RL_NN_3L(X, y, num_epochs=100, batch_size=3, L1=124, L2=256, L3=512, weight_decay=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = SimpleNN_3L(X_tensor.shape[1], L1, L2, L3).to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    # Train the model
    losses, best_model_state = train_3L(model, criterion, optimizer, X_train, y_train, batch_size, num_epochs, device)

    # Restore the best model state
    model.load_state_dict(best_model_state)

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test.to(device))
        test_loss = criterion(test_outputs, y_test.to(device))
        #print(f'Test Loss: {test_loss.item():.4f}')

    return model, losses, test_loss

# Fine-tuning the model with new data
def fine_tune_model(model, criterion, optimizer, X_train, y_train, batch_size, num_epochs, device):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i+batch_size].to(device)
            targets_batch = y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / (len(X_train) // batch_size)
        losses.append(avg_epoch_loss)

    return model, losses

# Fine-tuning process encapsulated into a function
def fine_tune_existing_model(model, new_X_data, new_y_data, batch_size=2, fine_tune_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert new data to tensors
    new_X = torch.tensor(new_X_data, dtype=torch.float32)
    new_y = torch.tensor(new_y_data, dtype=torch.float32).view(-1, 1)

    # Split new dataset into training and test sets
    new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(new_X, new_y, test_size=0.2, random_state=42)

    # Fine-tune the pre-trained model with the new data
    fine_tune_optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)  # Lower learning rate

    # Fine-tune the model using the new dataset
    fine_tuned_model, fine_tune_losses = fine_tune_model(model, nn.MSELoss(), fine_tune_optimizer, new_X_train, new_y_train, batch_size, fine_tune_epochs, device)

    # Evaluate the fine-tuned model on the new test set
    with torch.no_grad():
        fine_tuned_model.eval()
        new_test_outputs = fine_tuned_model(new_X_test.to(device))
        new_test_loss = nn.MSELoss()(new_test_outputs, new_y_test.to(device))
        #print(f'Fine-tuned Test Loss: {new_test_loss.item():.4f}')

    return fine_tuned_model, fine_tune_losses, new_test_loss


def NN_pred(Model_NN, input_data):
    # Recall the trained model
    Model_NN.eval()  # Set the model to evaluation mode
    
    # Convert the input data to a tensor (ensure it's the correct dtype, i.e., float32)
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    # Get the prediction
    with torch.no_grad():  # No need to compute gradients
        prediction = Model_NN(input_tensor)
    
    # Output the prediction
    predicted_value = prediction.item()  # Convert tensor to scalar value
    
    return  predicted_value


if __name__ == '__main__':
    # Data for training (X) and target (y)
    Transposed_Sampled_State = [
        (132479, 0, 0, 77.87, 3992.32, 9212.98, 118.28, 0.024220410029554468, 10480.108252263679, 0.0, 21.6, 4.1, 0.0),
        (132479, 0, 1, 91.75, 2441.89, 2009.29, 225.37, 0.05734170144684216, 7774.173377140024, 0, 19.0, 4.1, 0.000166667),
        (132479, 0, 1, 72.78, 3317.36, 6160.56, 229.55, 0, 8062.627874188739, 0, 19.0, 7.7, 0.0)
    ]
    Value = [0.04734602907769546, 0.06420829056323789, 0.06487175692124406]

    # NN Hyper Parameters
    L1 = 64
    L2 = 128
    L3 = 64
    batch_size = 2  # Smaller batch size since the dataset is small
    num_epochs = 200

    # Train the model
    Model_NN, losses, test_loss = RL_NN_3L(Transposed_Sampled_State, Value, num_epochs=num_epochs, batch_size=batch_size, L1=L1, L2=L2, L3=L3)

    # Plot the losses for training
    plt.plot(range(1, num_epochs+1), losses, 'ro', markersize=3, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'MSE Loss vs Epoch - Training.\n Train Loss: {min(losses):.4f} \n Test Loss: {test_loss.item():.4f}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Introduce new random set of Transposed_Sampled_State and Value
    new_Transposed_Sampled_State = [
        (132480, 0, 1, 80.23, 3700.32, 5800.98, 230.18, 0.045332510129774, 9520.67823267423, 0.0, 18.9, 3.7, 0.000133333),
        (132480, 0, 0, 90.55, 2550.12, 3400.65, 240.45, 0.062425612836547, 7124.3849211254, 0, 17.0, 5.0, 0.000211111),
        (132480, 0, 1, 75.42, 3500.95, 6200.23, 240.85, 0, 8450.882132437864, 0, 19.0, 8.0, 0.0)
    ]
    new_Value = [0.04200134645788794, 0.06521234567814567, 0.06346987634587234]

    # Fine-tune the model with the new random data
    fine_tuned_model, fine_tune_losses, new_test_loss = fine_tune_existing_model(Model_NN, new_Transposed_Sampled_State, new_Value, batch_size=batch_size, fine_tune_epochs=num_epochs)

    # Plot the losses for fine-tuning
    plt.plot(range(1, len(fine_tune_losses) + 1), fine_tune_losses, 'bo', markersize=3, label='Fine-Tuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'MSE Loss vs Epoch - Fine-Tuning.\n Train Loss: {min(fine_tune_losses):.4f} \n Test Loss: {new_test_loss.item():.4f}')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Prediction
    # Input data for prediction (as given in the question)
    input_data = (132450, 0, 1, 92.35, 2966.73, 2233.35, 225.6, 0, 7443.56349378364, 0, 19.0, 5.7, 0.0)
    
    predicted_value = NN_pred(Model_NN, input_data)
    print(predicted_value)
    
    
    
