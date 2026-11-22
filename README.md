
# SWE015 – Intelligent Handwritten Answer Digitizer

## Project Overview

An AI-powered web application that recognizes handwritten
digits and letters and corrects misspelled words using
Natural Language Processing and a quantized BERT model.

The application supports image uploads and handwritten
drawing through an interactive canvas.

## Main Features

- Handwritten digit recognition
- Handwritten letter and short-word recognition
- Character segmentation using OpenCV
- Minimum Edit Distance spelling candidates
- Smoothed bigram language model
- Context-aware BERT masked-language-model reranking
- Quantized INT8 BERT for word correction
- Quantized INT8 CNN for EMNIST character classification
- CNN Grad-CAM visualization developed in the notebook

## AI Models

| Model | Purpose |
|---|---|
| Dense Neural Network | EMNIST baseline comparison |
| CNN | EMNIST character classification |
| CNN INT8 | Quantized character classification |
| Classical NLP | Candidate generation and n-gram ranking |
| BERT | Context-based candidate reranking |
| BERT INT8 | Efficient contextual word correction |

The current Digit and Letter interface uses the existing
recognition models. The EMNIST INT8 CNN is accessible
through a separate testing endpoint.

The application uses quantized BERT in its letter
correction pipeline.

## Technology Stack

- Python and Flask
- TensorFlow / Keras
- TensorFlow Lite
- PyTorch
- Hugging Face Transformers
- OpenCV
- NLTK
- RapidFuzz
- HTML, CSS and JavaScript

## Setup

Recommended environment: Python 3.13 on Windows.

### 1. Install dependencies

    python -m pip install -r requirements.txt

### 2. Download NLTK data

    python -m nltk.downloader brown

### 3. Download model weights

Model weights are excluded from GitHub due to file size.


Model download link: [Download SWE015 Model Weights](https://drive.google.com/file/d/1kb03-IMr5MeoRTBEFF8pP3X2RxD7juCi/view?usp=sharing)

Place the required files in these locations:

    models/
        digits_model.keras
        letters_model.keras

    quantized_models/
        swe015_cnn_int8.tflite
        bert_int8_weights.pt

The original EMNIST Dense and CNN weights are also
available in the model backup for reproducibility.

### 4. Run the application

    python aqq.py

Open:

    http://127.0.0.1:5001

## How to Use

1. Select Digit or Letter mode.
2. Draw handwritten input or upload an image.
3. For Letter mode, provide sentence context using [MASK].
4. Click Predict.
5. View the raw recognition and corrected word.

Example:

    Input: HELO
    Context: I said [MASK] to my friend.
    Correction: HELLO

## Experiments

Models were trained and evaluated using EMNIST Balanced.

Measured character classification results:

| Model | Test Accuracy |
|---|---|
| Dense | 74.34% |
| CNN | 87.45% |
| CNN INT8 | 87.43% |

These are character-level test results, not word-level
correction accuracy.

Additional benchmarks and training experiments are
documented in SWE015_Model_Training.ipynb.

## Limitations

- Handwriting recognition depends on segmentation
  and image preprocessing.
- The original web recognition models and the EMNIST
  models are different.
- Contextual correction depends on the supplied sentence.
- The application currently supports short English words.

## AI Usage Statement

ChatGPT was used for coding assistance, debugging,
integration guidance, and documentation drafting.

Model training, execution, testing and reported results
were performed in the project environment.

## References

- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- NLTK: https://www.nltk.org/
- BERT: https://huggingface.co/google-bert/bert-base-uncased
- Flask: https://flask.palletsprojects.com/
- PyTorch: https://pytorch.org/
