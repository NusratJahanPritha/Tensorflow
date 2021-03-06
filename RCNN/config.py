import os
ORIG_BASE_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"])

MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

INPUT_DIMS = (224, 224)
# define the path to the output model and label binarizer
MODEL_PATH = "raccoon_detector.h5"
ENCODER_PATH = "label_encoder.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
