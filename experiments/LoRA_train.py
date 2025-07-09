import os
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_with_lora import Model
from src.dataset_horae import Horae
from experiments.options import opts

import copy

if __name__ == '__main__':
    dataset_transforms = Horae.data_transform(opts)

    train_opts = copy.deepcopy(opts)
    val_opts = copy.deepcopy(opts)

    # DocExplore
    train_dataset = Horae(
        path="/list_horae_train.pkl", opts=opts, transform=dataset_transforms)
    val_dataset = Horae(
        path="/list_horae_val.pkl", opts=opts, transform=dataset_transforms)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=opts.batch_size, num_workers=opts.workers, shuffle=True)

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # clip_loss
        dirpath='saved_models/%s' % opts.exp_name,
        filename="{epoch:02d}-{mAP:.2f}-{clip_loss:.2f}",
        mode='min',  # min
        save_last=True)

    ckpt_path = os.path.join(
        'saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print('resuming training from %s' % ckpt_path)

    trainer = Trainer(
        accelerator='gpu',
        gpus=[opts.gpu],
        min_epochs=1, max_epochs=opts.epochs,
        benchmark=True,
        logger=logger,
        check_val_every_n_epoch=1,
        resume_from_checkpoint=ckpt_path,
        callbacks=[checkpoint_callback]
    )

    if ckpt_path is None:
        model = Model()
    else:
        print('resuming training from %s' % ckpt_path)
        model = Model().load_from_checkpoint(ckpt_path)

    print('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader)
