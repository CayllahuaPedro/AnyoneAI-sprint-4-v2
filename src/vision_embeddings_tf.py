import numpy as np
import pandas as pd
import os
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, InceptionV3, ConvNeXtTiny
)
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image.
    
    Args:
    - image_path (str): Path to the image file.
    - target_size (tuple): Desired image size.
    
    Returns:
    - np.array: Preprocessed image.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img


class FoundationalCVModel:

    def __init__(self, backbone, mode='eval', input_shape=(224, 224, 3)):
        self.backbone_name = backbone
        input_layer = Input(shape=input_shape)

        # Modelos Keras (estables y sin dependencias extra de Transformers)
        if backbone == 'resnet50':
            self.base_model = ResNet50(weights='imagenet', include_top=False)
        elif backbone == 'resnet101':
            self.base_model = ResNet101(weights='imagenet', include_top=False)
        elif backbone == 'densenet121':
            self.base_model = DenseNet121(weights='imagenet', include_top=False)
        elif backbone == 'densenet169':
            self.base_model = DenseNet169(weights='imagenet', include_top=False)
        elif backbone == 'inception_v3':
            self.base_model = InceptionV3(weights='imagenet', include_top=False)
        elif backbone in {'convnext_tiny', 'convnextv2_tiny'}:
            self.base_model = ConvNeXtTiny(weights='imagenet', include_top=False)
        else:
            raise ValueError(
                f"Unsupported backbone model: {backbone}. "
                "Usá resnet50, resnet101, densenet121, densenet169, inception_v3 o convnext_tiny."
            )

        if mode == 'eval':
            self.base_model.trainable = False

        # 🔥 Construcción del modelo
        x = self.base_model(input_layer)
        outputs = GlobalAveragePooling2D()(x)

        self.model = Model(inputs=input_layer, outputs=outputs)

    def predict(self, images):
        return self.model.predict(images)


class ImageFolderDataset:

    def __init__(self, folder_path, shape=(224, 224), image_files=None):
        self.folder_path = folder_path
        self.shape = shape

        if image_files:
            self.image_files = image_files
        else:
            self.image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))
            ]

        self.clean_unidentified_images()

    def clean_unidentified_images(self):
        cleaned_files = []
        for img_name in self.image_files:
            img_path = os.path.join(self.folder_path, img_name)
            try:
                Image.open(img_path).convert("RGB")
                cleaned_files.append(img_name)
            except:
                print(f"Skipping {img_name}")
        self.image_files = cleaned_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        img = load_and_preprocess_image(img_path, self.shape)
        return img_name, img


def get_embeddings_df(
    batch_size=32,
    path="data/images",
    dataset_name='',
    backbone="resnet50",
    directory='Embeddings',
    image_files=None
):

    dataset = ImageFolderDataset(folder_path=path, image_files=image_files)
    model = FoundationalCVModel(backbone)

    img_names = []
    features = []

    for i in range(0, len(dataset), batch_size):
        batch_files = dataset.image_files[i:i + batch_size]
        batch_imgs = np.array([
            dataset[j][1] for j in range(i, min(i + batch_size, len(dataset)))
        ])

        batch_features = model.predict(batch_imgs)

        img_names.extend(batch_files)
        features.extend(batch_features)

        print(f"Batch {i // batch_size + 1} done")

    df = pd.DataFrame({
        'ImageName': img_names,
        'Embeddings': features
    })

    df_aux = pd.DataFrame(df['Embeddings'].tolist())
    df = pd.concat([df['ImageName'], df_aux], axis=1)

    os.makedirs(f'{directory}/{dataset_name}', exist_ok=True)

    df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)