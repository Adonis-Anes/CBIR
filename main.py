import logging
from feature_extraction.tuned_resnet50.train_resnet50 import train_resnet50
from feature_extraction.tuned_resnet50.extract_features_tuned_resnet import extract_tuned_resnet_features
from feature_extraction.extract_features_hog import extract_hog_features
from feature_extraction.extract_features_vit import extract_vit_features
from feature_extraction.extract_features_wavelet import extract_features_wavelet_main
from faiss_index.build_index import build_faiss_index, create_db_csv


def main():
    logging.basicConfig(level=logging.INFO)
    train_resnet50()
    extract_tuned_resnet_features()
    build_faiss_index('features_tuned_resnet.csv', 'feat_extract_tuned_resnet.index')
    extract_hog_features()
    build_faiss_index('features_hog.csv', 'feat_extract_hog.index')
    extract_vit_features()
    build_faiss_index('features_vit.csv', 'feat_extract_vit.index')
    extract_features_wavelet_main()
    build_faiss_index('features_wavelet_haar.csv', 'feat_extract_wavelet.index')
    create_db_csv()

if __name__ == "__main__":
    main()