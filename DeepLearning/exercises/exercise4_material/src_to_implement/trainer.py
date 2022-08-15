import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import os
import time


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit

        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._threshold = 0.5

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch, best_model):
        dir_name = '/home/cip/nf2021/ko65beyp/DL_ex4/checkpoints'
        for f in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, f))
        t.save({'state_dict': best_model}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        # -propagate through the network
        outputs = self._model(x).to(t.float32)
        y = y.to(t.float32)

        # -calculate the loss
        loss = self._crit(outputs, y)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()

        # -return the loss
        return float(loss)

    
    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        outputs = self._model(x).to(t.float32)
        y = y.to(t.float32)
        loss = self._crit(outputs, y)
        # return the loss and the predictions
        return loss, outputs
        
    def train_epoch(self):
        # set training mode
        self._model.train()
        running_loss = []

        # iterate through the training set
        for local_batch, local_labels in tqdm(self._train_dl):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            if self._cuda:
                local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
            # perform a training step
            running_loss.append(self.train_step(local_batch, local_labels))
        # calculate the average loss for the epoch and return it
        return sum(running_loss) / len(running_loss)

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        runnning_metrics = []
        running_loss = []
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for local_batch, local_labels in tqdm(self._val_test_dl):
                # transfer the batch to the gpu if given
                if self._cuda:
                    local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
                # perform a validation step
                loss, output = self.val_test_step(local_batch, local_labels)
                output = (output > self._threshold).float()
                # save the predictions and the labels for each batch
                runnning_metrics.append(f1_score(local_labels.cpu(), output.cpu(), average='macro'))
                running_loss.append(float(loss))
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        print(f'f1_score: {sum(runnning_metrics) / len(runnning_metrics)}')
        return sum(running_loss) / len(running_loss)

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch = 0
        best_model = None
        while True:
            epoch += 1
            print(f'epoch: {epoch} of {epochs}')
            # stop by epoch number
            if epoch > epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_losses.append(self.train_epoch())
            val_losses.append(self.val_test())
            # append the losses to the respective list
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if min(val_losses) == val_losses[-1]:
                best_model = self._model.state_dict()

            if epoch - val_losses.index(min(val_losses)) > self._early_stopping_patience:
                break
        self.save_checkpoint(epoch, best_model)
        # return the losses for both training and validation
        return train_losses, val_losses
