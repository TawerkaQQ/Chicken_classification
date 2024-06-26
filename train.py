import torch


from tqdm import tqdm


def train_loop(
        model,
        start_epoch,
        end_epoch,
        optimizer,
        device,
        criterion,
        train_summary_writer,
        val_summary_writer,
        train_dataloader,
        val_dataloader,
        model_save_dir,
        best_acc=0.0
):

    current_best_acc = best_acc
    for epoch in range(start_epoch, end_epoch):
        train_loss, train_accuracy = train_step(
            model, train_dataloader, optimizer, criterion, device, epoch
        )
        train_summary_writer.add_scalar('loss', train_loss, epoch)
        train_summary_writer.add_scalar('accuracy', train_accuracy, epoch)

        val_loss, val_accuracy = val_step(
            model, val_dataloader, criterion, device, epoch
        )
        val_summary_writer.add_scalar('loss/val', val_loss, epoch)
        val_summary_writer.add_scalar('accuracy/val', val_accuracy, epoch)

        if val_accuracy > current_best_acc:
            current_best_acc = val_accuracy
            torch.save(
                {
                    'accuracy': val_accuracy,
                    'loss': val_loss,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                model_save_dir / 'best_model.pt'
            )

        torch.save(
            {
                'accuracy': val_accuracy,
                'loss': val_loss,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            model_save_dir / 'latest_model.pt'
        )


def train_step(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    print(model.parameters())
    print(optimizer)  # PRINT
    print(type(optimizer))  # PRINT
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(dataloader, desc=f'Train epoch {epoch}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    return epoch_loss, epoch_acc



def val_step(model, dataloader, criterion, device, epoch):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in tqdm(dataloader, desc=f'Test epoch {epoch}'):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc