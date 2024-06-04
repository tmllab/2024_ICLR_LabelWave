import torch
import torch.nn as nn
import torch.optim as optim
import os
import labelwave_cifar10_dataloader as dataloader
import csv
device = torch.device("cuda")

EPOCH = 200
pre_epoch = 0
root_dir='./cifar-10'
BATCH_SIZE = 128
LR = 0.01
r=0.4
noise_mode = 'sym'
argsseed = 1


for iternum in range(0,1):
    file_name = 'Results_labelwave_'+ str(noise_mode) + str(r) +'_cifar10_ResNet18_'+'lr'+ str(LR) +'_bs'+ str(BATCH_SIZE) + '_' + str(iternum)
    file_name = str(file_name)
    print(file_name)

    csvfile = file_name + ".csv"
    with open(csvfile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'LabelWave', 'ValAcc', 'TestAcc'])

    path = os.getcwd()
    file_path = path + '/' + file_name
    folder = os.path.exists(file_path)
    if not folder:
        os.mkdir(file_path)

    loader = dataloader.CifarDataLoader(root_dir, noise_mode, r, BATCH_SIZE, num_workers=8)

    test_loader = loader.run('test')
    train_loader = loader.run('train')
    val_loader = loader.run('val')

    def build_model():
        from resnet import ResNet18
        model = ResNet18(10)
        print('============ use resnet18 ')
        model = model.cuda()
        return model

    print("Start Training!")
    net = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    previous_predictions = {}
    for epoch in range(pre_epoch, EPOCH):
        net.train()
        labelwave = 0
        total_loss = 0
        total_batches = 0
        current_epoch_predictions = {}

        for batch_idx, (inputs, labels, paths) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1
            loss.backward()
            optimizer.step()
            _, current_predictions = torch.max(outputs.data, 1)
            for idx, path_tensor in enumerate(paths):
                path = str(path_tensor.item())
                current_epoch_predictions[path] = current_predictions[idx].item()

        labelwave = 0
        for path, prediction in current_epoch_predictions.items():
            if path in previous_predictions and previous_predictions[path] != prediction:
                labelwave += 1
        previous_predictions = current_epoch_predictions.copy()
        print(f"Epoch {epoch + 1}, LabelWave: {labelwave}")

        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (inputs, labels, _) in enumerate(val_loader):
                net.eval()
                images, labels = inputs, labels
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                val_acc = 100 * correct / total
                val_acc = val_acc.item()

            correct = 0
            total = 0
            for batch_idx, (inputs, labels, _) in enumerate(test_loader):
                net.eval()
                images, labels = inputs, labels
                images, labels = images.cuda(), labels.cuda()
                outputs = net(images)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                test_acc = 100 * correct / total
                test_acc = test_acc.item()
        print('ValAcc：%.3f%%' % val_acc, 'TestAcc：%.3f%%' % test_acc)

        with open(csvfile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, labelwave, val_acc, test_acc])