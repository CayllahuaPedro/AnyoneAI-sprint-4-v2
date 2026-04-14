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


def _build_tf_dataset(folder_path, image_files, target_size, batch_size):
    paths = [os.path.join(folder_path, f) for f in image_files]
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape((target_size[0], target_size[1], 3))
        return img

    ds = paths_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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

    gpus = tf.config.list_physical_devices('GPU')
    print(f"TF {tf.__version__} | GPUs visible: {len(gpus)}")

    target_size = (224, 224)
    ds = _build_tf_dataset(path, dataset.image_files, target_size, batch_size)

    features = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for step, batch in enumerate(ds, start=1):
        batch_features = model.model(batch, training=False).numpy()
        features.extend(batch_features)
        print(f"Batch {step}/{total_batches} done")

    df = pd.DataFrame({
        'ImageName': dataset.image_files,
        'Embeddings': features
    })

    df_aux = pd.DataFrame(df['Embeddings'].tolist())
    df = pd.concat([df['ImageName'], df_aux], axis=1)

    os.makedirs(f'{directory}/{dataset_name}', exist_ok=True)

    df.to_csv(f'{directory}/{dataset_name}/Embeddings_{backbone}.csv', index=False)