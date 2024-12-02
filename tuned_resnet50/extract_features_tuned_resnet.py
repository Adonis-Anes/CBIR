import os
import logging
import numpy as np
import pandas as pd
from keras.models import load_model
from preprocessor.preprocessing import image_preprocessing
from config import SAVED_FEATURES_DIR, TRAIN_DIR, TUNED_RESNET50_MODEL_PATH


def load_tuned_resnet(model_path=TUNED_RESNET50_MODEL_PATH):
    """
    Carga el modelo ResNet50 ajustado (tuned) desde un archivo.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.

    Returns:
        keras.Model: Modelo cargado listo para extracción de características.
    """
    logging.info(f"Loading tuned ResNet50 model from {model_path}...")
    model = load_model(model_path)
    feature_extraction_model = model  # Usa el modelo tal como está para la extracción
    logging.info("Tuned ResNet50 model loaded")
    return feature_extraction_model


def get_features_from_tuned_resnet(image_input, model):
    """
    Extrae características de una imagen utilizando el modelo ajustado ResNet50.

    Args:
        image_input: Path to the image file or preloaded image as numpy array.
        model (keras.Model): Modelo ajustado para la extracción de características.

    Returns:
        numpy.ndarray: Vector de características de la imagen.
    """
    img = image_preprocessing(image_input)  # Preprocesar la imagen
    img = img.reshape(1, 224, 224, 4)  # Ajustar al tamaño esperado por el modelo
    feature = model.predict(img)
    feature = feature[0]  # Elimina la dimensión batch
    return feature


def construct_features_dict_tuned(model):
    """
    Construye un diccionario de características a partir de un conjunto de imágenes.

    Args:
        image_folder (str): Carpeta que contiene las imágenes.
        model (keras.Model): Modelo ajustado para la extracción de características.

    Returns:
        dict: Diccionario con nombres de imágenes como claves y vectores de características como valores.
    """
    features_dict = {}
    for img_name in os.listdir(TRAIN_DIR):
        img_path = os.path.join(TRAIN_DIR, img_name)
        if os.path.isfile(img_path):
            features = get_features_from_tuned_resnet(img_path, model)
            features_dict[img_name] = features
            logging.info(f"Extracted features for {img_name}")
    return features_dict


def get_features_df(features_dict):
    """
    Convierte el diccionario de características en un DataFrame.

    Args:
        features_dict (dict): Diccionario de características.

    Returns:
        pd.DataFrame: DataFrame con las características.
    """
    df = pd.DataFrame.from_dict(features_dict, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "image"}, inplace=True)
    return df


def save_features(df, filename="features_tuned_resnet.csv"):
    """
    Guarda las características extraídas en un archivo CSV.

    Args:
        df (pd.DataFrame): DataFrame con las características.
        filename (str): Nombre del archivo de salida.
    """
    output_path = os.path.join(SAVED_FEATURES_DIR, filename)
    df.to_csv(output_path, index=False)
    logging.info(f"Features saved to {output_path}")


def extract_tuned_resnet_features():
    """
    Pipeline completo para la extracción de características utilizando el modelo ajustado ResNet50.

    Args:
        model_path (str): Ruta al archivo del modelo guardado.
        image_folder (str): Carpeta que contiene las imágenes.

    Returns:
        None
    """
    model = load_tuned_resnet()
    features_dict = construct_features_dict_tuned(model)
    df = get_features_df(features_dict)
    save_features(df)

if __name__ == "__extract_tuned_resnet_features__":
    extract_tuned_resnet_features()