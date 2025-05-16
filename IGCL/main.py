import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from torch import nn
from model import IGCL
from torch_geometric.loader import DataLoader
from arguments import arg_parse
import datetime
from copy import deepcopy
from dataset import create_dataset
from identification import custom_recognition_by_correlation
from methods import load_fc_matrices
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, \
    confusion_matrix
from classifier import FullyConnectedClassifier

if __name__ == '__main__':
    # Set random number seed
    torch.manual_seed(42)
    args = arg_parse()

    accuracies = {'val': [], 'test': []}
    epochs = 150
    log_interval = 10
    batch_size = 128
    lr = args.lr

    # dataset
    # Folder list
    folders = ["REST1", "GAMBLING", "MOTOR", "TWMEMORY", "REST2", "EMOTION", "LANGUAGE", "RELATIONAL", "SOCIAL"]
    # Dataset base path
    base_path = ""

    # Write information for this train
    with open('.txt', 'a') as file:
        file.write(f'\n')
    for i, folder1 in enumerate(folders):
        if i != -9:
            for folder2 in folders[i + 1:]:  # Only iterate over items after the current index
                if folder1 != folder2:  # Ensure the two folders are different
                    path1 = base_path + folder1
                    path2 = base_path + folder2
                    print(f"Combination path 1: {path1}")
                    print(f"Combination path 2: {path2}")
                    print("----------")

                    # Load REST1 and REST2 data
                    X_1, y_1 = load_fc_matrices(path1)
                    X_2, y_2 = load_fc_matrices(path2)

                    one_indices = np.argsort(y_1)
                    X_1 = X_1[one_indices]
                    y_1 = y_1[one_indices]

                    two_indices = np.argsort(y_2)
                    X_2 = X_2[two_indices]
                    y_2 = y_2[two_indices]

                    # Read Excel file and create ID-gender mapping dictionary
                    y_1_gender = y_1
                    y_2_gender = y_2
                    # Gender file path
                    file_path = r".xlsx"
                    df = pd.read_excel(file_path)

                    # Create mapping from ID to gender
                    subject_to_gender = df.set_index('Subject')['Gender'].to_dict()

                    # Define mapping from gender character to numerical value
                    gender_to_num = {'M': 0, 'F': 1}

                    # Iterate over all y variables and complete double conversion (ID → character → numerical)
                    for i in range(1, 2):
                        y_var = f'y_{i}_gender'
                        if hasattr(y_var, '__iter__'):
                            # Complete double conversion in one step: ID → gender character → numerical label
                            globals()[y_var] = [gender_to_num[subject_to_gender[id]]
                                                for id in globals()[y_var]]
                            print(f"Converted {y_var} labels to numerical classification (M=0, F=1)")

                    # Calculate threshold for each individual, retaining the top 5% of data
                    threshold_1 = np.percentile(X_1, 95, axis=(1, 2), keepdims=True)
                    threshold_2 = np.percentile(X_2, 95, axis=(1, 2), keepdims=True)
                    # Create sparse matrix, processing each individual separately
                    num_elements_to_keep = 360
                    X_1_sparse = []
                    for i in range(X_1.shape[0]):
                        # Apply threshold to each individual to generate sparse matrix
                        threshold = threshold_1[i, 0, 0]
                        sparse_matrix = csr_matrix(X_1[i] * (X_1[i] >= threshold))

                        # If the number of non-zero elements exceeds the limit, remove excess threshold elements
                        if sparse_matrix.nnz > num_elements_to_keep:
                            # Get indices and values of non-zero elements
                            non_zero_indices = sparse_matrix.nonzero()
                            non_zero_values = sparse_matrix.data

                            # Find indices of non-zero elements equal to the threshold
                            threshold_indices = np.where(non_zero_values == threshold)[0]

                            # Calculate the number of elements to remove
                            num_to_remove = sparse_matrix.nnz - num_elements_to_keep

                            # Select indices to remove from threshold elements
                            remove_indices = threshold_indices[:num_to_remove]

                            # Remove specified elements
                            keep_indices = np.setdiff1d(np.arange(sparse_matrix.nnz), remove_indices)
                            sparse_matrix.data = sparse_matrix.data[keep_indices]
                            sparse_matrix.indices = sparse_matrix.indices[keep_indices]
                            sparse_matrix.indptr = np.append([0], np.cumsum(
                                np.bincount(non_zero_indices[0][keep_indices], minlength=sparse_matrix.shape[0])))

                        X_1_sparse.append(sparse_matrix)

                    X_2_sparse = []
                    for i in range(X_2.shape[0]):
                        threshold = threshold_2[i, 0, 0]
                        sparse_matrix = csr_matrix(X_2[i] * (X_2[i] >= threshold))

                        if sparse_matrix.nnz > num_elements_to_keep:
                            # Get indices and values of non-zero elements
                            non_zero_indices = sparse_matrix.nonzero()
                            non_zero_values = sparse_matrix.data

                            # Find indices of non-zero elements equal to the threshold
                            threshold_indices = np.where(non_zero_values == threshold)[0]

                            # Calculate the number of elements to remove
                            num_to_remove = sparse_matrix.nnz - num_elements_to_keep

                            # Select indices to remove from threshold elements
                            remove_indices = threshold_indices[:num_to_remove]

                            # Remove specified elements
                            keep_indices = np.setdiff1d(np.arange(sparse_matrix.nnz), remove_indices)
                            sparse_matrix.data = sparse_matrix.data[keep_indices]
                            sparse_matrix.indices = sparse_matrix.indices[keep_indices]
                            sparse_matrix.indptr = np.append([0], np.cumsum(
                                np.bincount(non_zero_indices[0][keep_indices], minlength=sparse_matrix.shape[0])))

                        X_2_sparse.append(sparse_matrix)

                    # Stack sparse matrices together while maintaining original dimensions
                    X_1_sparse = np.array([x.toarray() for x in X_1_sparse])
                    X_2_sparse = np.array([x.toarray() for x in X_2_sparse])

                    # Split into training and test sets
                    pot = 400
                    start = 500
                    pot_start = 500
                    pot_end = 969

                    X_1_train = X_1[:pot]
                    X_2_train = X_2[:pot]
                    X_1_sparse_train = X_1_sparse[:pot]
                    X_2_sparse_train = X_2_sparse[:pot]
                    y_1_train = y_1[:pot]
                    y_2_train = y_2[:pot]

                    X_1_eval = X_1[pot:start]
                    X_2_eval = X_2[pot:start]
                    X_1_sparse_eval = X_1_sparse[pot:start]
                    X_2_sparse_eval = X_2_sparse[pot:start]
                    y_1_eval = y_1[pot:start]
                    y_2_eval = y_2[pot:start]

                    X_1_test = X_1[pot_start:pot_end]
                    X_2_test = X_2[pot_start:pot_end]
                    X_1_sparse_test = X_1_sparse[pot_start:pot_end]
                    X_2_sparse_test = X_2_sparse[pot_start:pot_end]
                    y_1_test = y_1[pot_start:pot_end]
                    y_2_test = y_2[pot_start:pot_end]

                    dataset = create_dataset(X_1_train, X_1_sparse_train, y_1_train, X_2_train,
                                             X_2_sparse_train, y_2_train)
                    dataset_eval = create_dataset(X_1_eval, X_1_sparse_eval, y_1_eval, X_2_eval,
                                                  X_2_sparse_eval, y_2_eval)
                    dataset_test = create_dataset(X_1_test, X_1_sparse_test, y_1_test, X_2_test,
                                                  X_2_sparse_test, y_2_test)
                    dataset_num_features = 360

                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    copydataloader = DataLoader(dataset, batch_size=batch_size)
                    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)
                    dataloader_test = DataLoader(dataset_test, batch_size=batch_size)

                    max_acc_all = 0
                    best_model_path = 'best_model.pth'  # Path to save the model

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = IGCL(args.hidden_dim, args.num_gc_layers, args.value1, args.value2, args.value3, args.value4).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    print('================')
                    print(f'Number of individuals: {len(y_1)}')
                    print('lr: {}'.format(lr))
                    print('num_features: {}'.format(dataset_num_features))
                    print('hidden_dim: {}'.format(args.hidden_dim))
                    print('================')

                    for epoch in range(1, epochs + 1):
                        loss_all = 0
                        model.train()
                        for data in dataloader:
                            data, data_aug = data
                            data_copy = deepcopy(data)
                            data_copy.to(device)

                            optimizer.zero_grad()

                            node_num, _ = data.x.size()

                            data = data.to(device)
                            x, M, f_e = model(data.x, data.edge_index)
                            data_aug = data_aug.to(device)
                            x_aug, M_aug, f_e_aug = model(data_aug.x, data_aug.edge_index)
                            loss = model.HCNT_Xent(x, x_aug)
                            loss_all += loss.item()
                            loss.backward()
                            optimizer.step()

                        print(
                            '{}: Epoch {}, Loss {}'.format(datetime.datetime.now(), epoch,
                                                           loss_all / len(dataloader)))

                        if epoch % 5 == 0:
                            model.eval()

                            emb_train, y_tra = model.encoder.get_embeddings(dataloader_eval)
                            emb_test, y_tes = model.encoder.get_embeddings1(dataloader_eval)

                            acc_1 = custom_recognition_by_correlation(emb_train, emb_test)

                            acc_2 = custom_recognition_by_correlation(emb_test, emb_train)

                            print(f'Evaluation {folder1}->{folder2} accuracy: {acc_1:.4f}')

                            print(f'Evaluation {folder2}->{folder1} accuracy: {acc_2:.4f}')

                            acc_all = acc_1 + acc_2

                            # If current acc_all is greater than the maximum, update and save the model
                            if acc_all >= max_acc_all:
                                max_acc_all = acc_all
                                max_acc_epoch = epoch
                                # Save the model
                                torch.save(model.state_dict(), best_model_path)
                                print(f'Model saved, current best accuracy: {max_acc_all:.4f} at epoch: {max_acc_epoch}')

                    max_model = IGCL(args.hidden_dim, args.num_gc_layers, args.value1, args.value2, args.value3, args.value4).to(device)
                    # Load the saved model weights
                    model.load_state_dict(torch.load(best_model_path))

                    # Switch to evaluation mode
                    model.eval()
                    # Perform inference
                    e1, y1 = model.encoder.get_embeddings((copydataloader))
                    emb_eval, y_eval = model.encoder.get_embeddings(dataloader_eval)
                    emb_train, y_tra = model.encoder.get_embeddings(dataloader_test)
                    emb_test, y_tes = model.encoder.get_embeddings1(dataloader_test)

                    # # Gender classification
                    # # Check device
                    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    # print(f"Using device: {device}")
                    #
                    # # Assuming input feature dimension is 129600, number of classes is num_classes
                    # input_dim = 129600
                    # num_classes = 2  # Assuming y_train_split is the label of the training set
                    #
                    # # Concatenate training set and test set
                    # X = np.concatenate([e1, emb_eval, emb_train], axis=0)
                    # Y = np.concatenate([y_1_gender[:400], y_1_gender[400:500], y_1_gender[500:]], axis=0)
                    #
                    # # Convert NumPy arrays to PyTorch tensors
                    # X = X.reshape(-1, 129600)
                    # X = torch.tensor(X, dtype=torch.float32).to(device)
                    # Y = torch.from_numpy(Y).long().to(device)
                    #
                    # # Initialize model, loss function, and optimizer
                    # num_epochs = 600
                    # criterion0 = nn.CrossEntropyLoss().to(device)
                    #
                    # # ten-fold cross-validation
                    # kf = KFold(n_splits=10, shuffle=True, random_state=42)
                    # accuracies = []
                    # f1_scores = []
                    # sensitivities = []
                    # specificities = []
                    # aucs = []
                    #
                    # for fold, (train_indices, val_indices) in enumerate(kf.split(X)):
                    #     # Create training set and validation set
                    #     train_X, train_Y = X[train_indices].to(device), Y[train_indices].to(device)
                    #     val_X, val_Y = X[val_indices].to(device), Y[val_indices].to(device)
                    #
                    #     # Initialize model and optimizer
                    #     model0 = FullyConnectedClassifier(input_dim, num_classes).to(device)
                    #     optimizer0 = optim.Adam(model0.parameters(), lr=0.0002)
                    #
                    #     # Train the model
                    #     for epoch in range(num_epochs):
                    #         model0.train()
                    #         optimizer0.zero_grad()
                    #         outputs = model0(train_X)
                    #         loss = criterion0(outputs, train_Y)
                    #         loss.backward()
                    #         optimizer0.step()
                    #         if epoch % 20 == 0:
                    #             print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
                    #
                    #     # Validate the model
                    #     model0.eval()
                    #     with torch.no_grad():
                    #         val_outputs = model0(val_X)
                    #         _, predicted = torch.max(val_outputs, 1)
                    #
                    #         # Calculate metrics
                    #         val_accuracy = accuracy_score(val_Y.cpu(), predicted.cpu())
                    #         f1 = f1_score(val_Y.cpu(), predicted.cpu())
                    #         cm = confusion_matrix(val_Y.cpu(), predicted.cpu())
                    #         sensitivity = recall_score(val_Y.cpu(), predicted.cpu())  # Sensitivity
                    #         specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity
                    #         auc = roc_auc_score(val_Y.cpu(), val_outputs.cpu()[:, 1])
                    #
                    #         accuracies.append(val_accuracy)
                    #         f1_scores.append(f1)
                    #         sensitivities.append(sensitivity)
                    #         specificities.append(specificity)
                    #         aucs.append(auc)
                    #
                    #         print(f'Fold {fold + 1}, Validation Accuracy: {val_accuracy:.4f}')
                    #         print(f'Fold {fold + 1}, F1 Score: {f1:.4f}')
                    #         print(f'Fold {fold + 1}, Sensitivity: {sensitivity:.4f}')
                    #         print(f'Fold {fold + 1}, Specificity: {specificity:.4f}')
                    #         print(f'Fold {fold + 1}, AUC: {auc:.4f}')
                    #
                    # # Calculate average metrics and standard deviations
                    # avg_acc = np.mean(accuracies)
                    # std_acc = np.std(accuracies)
                    # avg_f1 = np.mean(f1_scores)
                    # std_f1 = np.std(f1_scores)
                    # avg_sen = np.mean(sensitivities)
                    # std_sen = np.std(sensitivities)
                    # avg_spe = np.mean(specificities)
                    # std_spe = np.std(specificities)
                    # avg_auc = np.mean(aucs)
                    # std_auc = np.std(aucs)
                    #
                    # print(f'Model Evaluation:')
                    # print(f'Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}')
                    # print(f'Average F1 Score: {avg_f1:.4f} ± {std_f1:.4f}')
                    # print(f'Average Sensitivity: {avg_sen:.4f} ± {std_sen:.4f}')
                    # print(f'Average Specificity: {avg_spe:.4f} ± {std_spe:.4f}')
                    # print(f'Average AUC: {avg_auc:.4f} ± {std_auc:.4f}')

                    acc_1 = custom_recognition_by_correlation(emb_train, emb_test)

                    print(f'Testing {folder1}->{folder2} accuracy: {acc_1:.4f}')

                    acc_2 = custom_recognition_by_correlation(emb_test, emb_train)

                    print(f'Testing {folder2}->{folder1} accuracy: {acc_2:.4f}')

                    # Write information to a file
                    with open('.txt', 'a') as file:
                        file.write(f'Testing {folder1}->{folder2} accuracy: {acc_1:.4f}\n')

                    # Write information to a file
                    with open('.txt', 'a') as file:
                        file.write(f'Testing {folder1}->{folder2} accuracy: {acc_2:.4f}\n')