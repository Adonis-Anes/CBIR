from preprocessor.preprocessing import image_preprocessing
import sys
import os
import pandas as pd
import logging
from feature_extraction.tuned_resnet50.train_resnet50 import train_resnet50
from feature_extraction.tuned_resnet50.extract_features_tuned_resnet import extract_tuned_resnet_features
from feature_extraction.extract_features_hog import extract_hog_features
from feature_extraction.extract_features_vit import extract_vit_features
from feature_extraction.extract_features_wavelet import extract_features_wavelet_main
from faiss_index.build_index  import build_faiss_index, create_db_csv
from ui.app import main as app
import subprocess
from config import TRAIN_DIR


def main():
    logging.basicConfig(level=logging.INFO)
    #train_resnet50()
    #extract_tuned_resnet_features()
    #extract_hog_features()
    #extract_vit_features()
    #build_faiss_index()
    #subprocess.run(["streamlit", "run", "ui/app.py"])
    #extract_features_wavelet_main()
    build_faiss_index('features_wavelet_haar.csv', 'feat_extract_wavelet.index')


if __name__ == "__main__":
    main()