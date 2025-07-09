import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageOps

# Posibles labels
# De acá se pueden obtener las categorías que se pueden usar 
# Si el positive es de una categoría, el negative no puede ser de la misma.

labels = {'figure',
          'figure person',
          'figure person symbol',
          'figure person symbol.',
          'figure sketch',
          'figure sketch person',
          'figure sketch person symbol',
          'figure sketch person symbol.',
          'figure sketch person.',
          'figure sketch person. symbol',
          'figure sketch person. symbol.',
          'figure sketch symbol',
          'figure sketch symbol.',
          'figure sketch. person symbol',
          'figure sketch. person. symbol',
          'figure sketch. person. symbol.',
          'figure symbol',
          'figure symbol.',
          'figure. person symbol',
          'figure. person symbol.',
          'figure. sketch person symbol',
          'figure. sketch person. symbol',
          'person',
          'person symbol',
          'sketch',
          'sketch person',
          'sketch person symbol',
          'sketch symbol',
          'sketch. person symbol',
          'sketch. symbol',
          'symbol'}

class Horae(Dataset):
    """
    Esta clase tiene la finalidad de entregar los datos de la siguiente manera:
    (imagen original, imagen original con filtros (+), otra imagen (-))

    La idea es entrenar DINOv2 de forma contrastiva.
    """

    def __init__(self, path, transform=None, opts=None):
        try:
            self.df = pd.read_pickle(path)
        except Exception as e:
            print(f"No se pudo cargar el archivo pickle: {e}")
            raise e 

        self.transform = transform
        self.opts = opts

        # Cambiamos el path de las imágenes
        self.change_path()
        print(len(self.df))
        # self.validation()

    def __len__(self):
        return self.opts.len #len(self.df)

    def __getitem__(self, idx):
        df_idx = self.df.iloc[idx]
        label_idx = df_idx['label']

        crop_anchor = self.crop_image(df_idx)
        crop_anchor_tensor = self.transform_image(crop_anchor)

        transform_dino = self.transforms_dino() 
        crop_positive_tensor = transform_dino(crop_anchor)

        return crop_anchor_tensor, crop_positive_tensor
    
    def validation(self):
        for idx in tqdm(range(len(self.df))):
            df_idx = self.df.iloc[idx]

            x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
            if (x2 - x1) > 500 and (y2 - y1) > 500:
                tqdm.write(f"Se descartó el crop {df_idx['x1']},{df_idx['y1']},{df_idx['x2']},{df_idx['y2']} por ser muy grande")
                self.df.drop(idx, inplace=True)

                                 

    def crop_image(self, df_idx):
        path_image = df_idx['filename'].iloc[0] if isinstance(df_idx['filename'], pd.Series) else df_idx['filename']

        if not os.path.exists(path_image):
            raise FileNotFoundError(f"No se encontró el archivo de imagen: {path_image}")

        img = Image.open(path_image).convert('RGB')
        x1, y1, x2, y2 = map(int, [df_idx['x1'], df_idx['y1'], df_idx['x2'], df_idx['y2']])
        return img.crop((x1, y1, x2, y2))

    def transform_image(self, image):
        padded_image = ImageOps.pad(image, size=(self.opts.max_size, self.opts.max_size))
        return self.transform(padded_image)

    def save_sample(self, original, positive, negative):

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(original.permute(1, 2, 0))
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(positive.permute(1, 2, 0))
        ax[1].set_title('Positive')
        ax[1].axis('off')

        ax[2].imshow(negative.permute(1, 2, 0))
        ax[2].set_title('Negative')
        ax[2].axis('off')

        plt.show()

    def change_path(self, new_path='/media/chr/Datasets/HORAE/imgs/'):
        self.df['filename'] = self.df['filename'].apply(lambda x: x.replace('/home/cloyola/datasets/HORAE/data/pages_classification/', new_path))

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

    def transforms_dino(self):
        # Devuelve el objeto de transformación en lugar de aplicarlo directamente
        return transforms.Compose([
            transforms.Resize((self.opts.max_size, self.opts.max_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomResizedCrop(self.opts.max_size, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

