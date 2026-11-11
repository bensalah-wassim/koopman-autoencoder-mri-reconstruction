# Import libraries
import torch
from torch.optim import Adam
from tqdm import tqdm
from functools import wraps
from time import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import gc


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r  took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


class Trainer:
    def __init__(self, model, learning_rate, epoch, R):
        """The class for training the model
        model: nn.Module
            A pytorch model
        optimizer_type: 'adam'
        learning_rate: float
        epoch: int
        R: int
        """
        self.model = model.to("cuda")
        self.kernels = []
        # Get the model parameters
        for name, param in self.model.named_parameters():
            if name[0:3] == "CNN":
                self.kernels.append(param.shape[-2:])

        self.learning_rate = learning_rate
        self.R = R
        self.optimizer = Adam(self.model.parameters(), self.learning_rate)
        self.epoch = epoch

    # @timing
    def train(self,
              calib,
              coil,
              R,
              draw_curve=False, early_stop=False, silent=True):


        calib = torch.tensor(calib, dtype=torch.float).to("cuda")
        coils, nx_calib, ny_calib = calib.shape


        [kernel_x_1, kernel_y_1] = [5, 2]
        [kernel_x_2, kernel_y_2] = [1, 1]
        [kernel_x_3, kernel_y_3] = [3, 2]
        target_x_start = np.int32(
            np.ceil(kernel_x_1 / 2)
            + np.floor(kernel_x_2 / 2)
            + np.floor(kernel_x_3 / 2)
            - 1
        )
        target_x_end = np.int32(nx_calib - target_x_start - 1)
        target_y_start = (
            np.int32(
                (np.ceil(kernel_y_1 / 2) - 1)
                + (np.ceil(kernel_y_2 / 2) - 1)
                + (np.ceil(kernel_y_3 / 2) - 1)
            )
            * self.R
        )
        target_y_end = (
            ny_calib
            - np.int32(
                (
                    np.floor(kernel_y_1 / 2)
                    + np.floor(kernel_y_2 / 2)
                    + np.floor(kernel_y_3 / 2)
                )
            )
            * self.R
            - 1
        )

        target_dim_X = target_x_end - target_x_start + 1
        target_dim_Y = target_y_end - target_y_start + 1
        target_dim_Z = self.R - 1

        targets = torch.zeros((target_dim_Z,
                               target_dim_X,
                               target_dim_Y)).to("cuda")
        for t in range(0, R - 1):
            targets[t, :, :] = calib[
                coil,
                target_x_start: target_x_end + 1,
                target_y_start + t + 1: target_y_end + t + 2,
            ]

        losses = []
        torch.cuda.empty_cache()
        weights = self.model.state_dict()

        for n_epoch in tqdm(range(self.epoch), leave=False):
            self.model.train()
            epoch_loss = 0

            predictions = self.model(calib)
            predictions = predictions.view(
                (1, target_dim_Z, target_dim_X, target_dim_Y)
            )
            loss = self.loss_function(recon_x=predictions[0], x=targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.detach().cpu().item()

            if n_epoch % 50 == 0 and not silent:
                print(
                    "Epoch %d/%d - Loss: %.3f " % (n_epoch + 1,
                                                   self.epoch,
                                                   epoch_loss)
                )

            losses.append(epoch_loss)

        if draw_curve:
            plt.figure()
            plt.plot(np.arange(self.epoch) + 1, losses, label="Training loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        if early_stop:
            self.model.load_state_dict(weights)

        del predictions
        del targets

        return {"losses": losses}

    def loss_function(self, recon_x, x):
        diff = recon_x - x
        # Compute the Frobenius norm
        frobenius_norm = torch.norm(diff, p="fro").to("cuda")
        return frobenius_norm


@timing
def train_model(
    model,
    calib,
    Udata,
    epochs,
    R,
    draw_curve=True,
    early_stop=False,
    lr=1e-3,
    device="cuda",
    silent=False,
    save_bool=False,
    save_path=None
):
    torch.cuda.empty_cache()
    # Print model summary
    if not silent:
        summary(model, Udata.shape)

    dim = Udata.shape
    [kernel_x_1, kernel_y_1] = [5, 2]
    [kernel_x_2, kernel_y_2] = [1, 1]
    [kernel_x_3, kernel_y_3] = [3, 2]

    target_x_start = np.int32(
        np.ceil(kernel_x_1 / 2)
        + np.floor(kernel_x_2 / 2)
        + np.floor(kernel_x_3 / 2)
        - 1
    )
    target_x_end = np.int32(dim[1] - target_x_start)
    target_y_start = (
        np.int32(
            (np.ceil(kernel_y_1 / 2) - 1)
            + (np.ceil(kernel_y_2 / 2) - 1)
            + (np.ceil(kernel_y_3 / 2) - 1)
        )
        * R
        + 1
    )
    target_y_end = (
        dim[2]
        - np.int32(
            (
                np.floor(kernel_y_1 / 2)
                + np.floor(kernel_y_2 / 2)
                + np.floor(kernel_y_3 / 2)
            )
        )
        * R
    )
    target_dim_X = target_x_end - target_x_start
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_Z = R - 1
    recon = Udata.copy()
    for coil in range(dim[0]):
        if not silent:
            print(f"Training on the channel: {coil}")
        torch.cuda.empty_cache()
        model.initialize_weights_bias()
        # Define Trainer parameters
        trainer = Trainer(model=model, learning_rate=lr, epoch=epochs, R=R)
        # Train with ACS liens
        log = trainer.train(calib=calib, coil=coil, R=R, early_stop=early_stop)
        if save_bool:
            print("Model saved")
            torch.save(
                model.state_dict(), save_path+f"coil_{coil}.pth"
                )
        # Convert to PyTorch tensor and move to device
        input_tensor = torch.tensor(Udata, dtype=torch.float).to(device)
        with torch.no_grad():

            # Infer missing lines
            predictions = model(input_tensor)
            predictions = predictions.cpu().clone().detach()
            predictions = predictions.view((1,
                                            target_dim_Z,
                                            target_dim_X,
                                            target_dim_Y))
            if not silent:
                print(
                    f"Reconstructing the missing lines on the channel {coil}")
            target_x_end_kspace = dim[1] - target_x_start
            for t in range(0, R - 1):

                target_y_start = (
                    np.int32(
                        (np.ceil(kernel_y_1 / 2) - 1)
                        + np.int32((np.ceil(kernel_y_2 / 2) - 1))
                        + np.int32(np.ceil(kernel_y_3 / 2) - 1)
                    )
                    * R
                    + t
                    + 1
                )
                target_y_end_kspace = (
                    dim[2]
                    - np.int32(
                        (np.floor(kernel_y_1 / 2))
                        + (np.floor(kernel_y_2 / 2))
                        + np.floor(kernel_y_3 / 2)
                    )
                    * R
                    + t
                )
                recon[
                    coil,
                    target_x_start: target_x_end_kspace,
                    target_y_start: target_y_end_kspace + 1: R,
                ] = predictions[0, t, :, ::R]

    # Visualization
    if draw_curve:
        plt.figure()
        plt.plot(log["losses"], label="losses")
        plt.legend()
        plt.title("loss")

    del predictions
    del input_tensor
    gc.collect()

    if save_bool:
        np.save(save_path+"_kspace.npz",
                arr=recon, allow_pickle=False)

    return model, recon
