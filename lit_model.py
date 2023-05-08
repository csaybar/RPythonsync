import torch
import numpy as np
import pytorch_lightning as pl

from utils import create_memmap

# Create a DataLoader
class DataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, type="val"):
        self.data = np.memmap(
            data_path,
            dtype='float32',
            mode='r+',
            shape=(100, 8, 512, 512)
        )
        
        if type == "train":
            self.data = self.data[:80, ...]
        elif type == "val":
            self.data = self.data[80:, ...]
        else:
            raise ValueError("type must be either train or val")
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        batch = self.data[idx, ...]
        X = torch.from_numpy(batch[:4, ...]).float()
        y = torch.from_numpy(batch[4:, ...]).float()
        return X, y

# 1. Create a DataLoader
class AC_DataLoader(pl.LightningDataModule):
    def __init__(self, data_path, batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
    def train_dataloader(self):
        train_data = DataLoader(self.data_path, type="train")
        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
    def val_dataloader(self):
        val_data = DataLoader(self.data_path, type="val")
        return torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False
        )

# Set a model
class SimpleCNN(torch.nn.Module):
    def __init__(self,inch=4, outch=4):
        super(SimpleCNN,self).__init__()
        self.cnn_layers=torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=inch,
                out_channels=outch,
                kernel_size=1,
                stride=1)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        return x

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = torch.nn.MSELoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    ## 1. Create a memmap file
    file = "data.bin"
    shape  = (100, 8, 512, 512)
    #create_memmap(file, shape)

    ## 2. Create a memmap file
    dataset = AC_DataLoader(file, 4)
    
    ## 3. Create a model
    model = SimpleCNN()
    litmodel = LitModel(model)
    
    ## 4. Define a callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=25,
        verbose=False,
        mode='min'
    )
    
    ## 5. Define logger WANDB
    #logger = pl.loggers.WandbLogger("test")
    logger = None
    

    ## 6. Define trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    ## 7. Train
    trainer.fit(litmodel, dataset)
    
    ## 8. Create a torchscript model
    # It is a way to convert your PyTorch models (written in Python)
    # into a format that can be efficiently executed in different
    # environments, such as servers, edge devices, or embedded systems.    
    model = trainer.model.model
    model.eval()
    example = torch.rand(1, 4, 512, 512)
    traced_script_module = torch.jit.trace(model, example) # good for models with static computation graph.
    traced_script_module.save("model.pt")
    
    ## 9. Load a torchscript model
    #import torch
    #model = torch.jit.load("model.pt")