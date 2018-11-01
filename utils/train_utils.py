from torch.autograd import Variable
import numpy as np
import os
import torch


def prep_inputs(batch, device=0):
    inputs = (Variable(batch['sentence1']).cuda(device=device),
              Variable(batch['sentence2']).cuda(device=device))
    label = Variable(batch['label']).cuda(device=device)

    return inputs, label


def evaluate(model, dataloader, loss_func, device, n_batches=10):
    loss = 0
    accuracy = 0
    model.train(False)

    for ix, batch in enumerate(dataloader):
        if ix >= n_batches:
            break

        inputs, label = prep_inputs(batch, device=device)

        predicted = model(inputs)

        loss += loss_func(predicted, label).data.cpu().numpy()
        prediction = torch.argmax(predicted, dim=1)

        # TODO: get the confusion here
        accuracy += (torch.sum(prediction == label).data.cpu().numpy() / len(prediction))

    accuracy /= n_batches
    loss /= n_batches

    model.train(True)

    return loss, accuracy


def predict(model, dataloader, device):
    list_predictions = []
    for ix, batch in enumerate(dataloader):
        inputs, label = prep_inputs(batch, device=device)

        predicted = model(inputs)
        prediction = torch.argmax(predicted, dim=1)
        list_predictions += list(prediction.data.cpu().numpy())

    return np.array(list_predictions)


def train(model, dataloader_train, dataloader_val, optimizer, loss_func, device, model_dir: str,
          n_epochs: int = 4, start_epoch=None,
          val_every: int = 1000, save_every: int = 50000,
          writer=None
          ):
    epoch = start_epoch or 0

    while epoch < n_epochs:
        model.train(True)

        acc_val = None
        for iteration, batch in enumerate(dataloader_train):
            if iteration > 0 and iteration % save_every == 0:
                torch.save(model.state_dict(),
                           f=os.path.join(model_dir, '{}_{}.pt'.format(epoch, iteration)))

            # Predict
            inputs, label = prep_inputs(batch, device=device)
            predicted = model(inputs)

            # Backprop, optimize
            optimizer.zero_grad()
            loss_train = loss_func(predicted, label)
            loss_train.backward()
            optimizer.step()

            # Log to tensorboard
            iter_total = (epoch * len(dataloader_train)) + iteration
            if writer:
                writer.add_scalar('train.loss', loss_train.data.cpu().numpy(), iter_total)

            # Calculate validation accuracy
            if iteration > 0 and iteration % val_every == 0:
                loss_val, acc_val = evaluate(model, dataloader_val, loss_func=loss_func, n_batches=200, device=device)

                s = "Epoch: {}, {:.2f}%: train loss: {}, validation loss: {}, validation acc: {}".format(
                    epoch, (iteration / len(dataloader_train)) * 100, loss_train.data.cpu().numpy(), loss_val, acc_val
                )
                print(s)
                if writer:
                    writer.add_scalar('val.loss', loss_val, iter_total)
                    writer.add_scalar('val.acc', acc_val, iter_total)

        print('\n----------------------------------------------------------------------------------------------------')
        print("Epoch:", epoch + 1, "label accuracy:", acc_val)
        print('----------------------------------------------------------------------------------------------------\n')

        epoch += 1

    # return model
