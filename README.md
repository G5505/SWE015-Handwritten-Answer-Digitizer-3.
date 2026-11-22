# SWE015 – Intelligent Handwritten Answer Digitizer

## Project Overview

An AI-powered web application for recognizing handwritten digits and short English words and correcting recognition errors using classical Natural Language Processing (NLP) and a quantized BERT model.

Users can upload images or draw directly on an interactive canvas. The system segments handwritten characters, recognizes them, and uses sentence context to improve word correction.

## Main Features

- Handwritten digit and letter recognition
- Drawing canvas and image upload
- Character segmentation using OpenCV
- Minimum Edit Distance candidate generation
- Smoothed bigram language model
- BERT masked-language-model candidate reranking
- Dynamic INT8 quantization of BERT
- INT8 CNN trained on EMNIST Balanced
- Grad-CAM visualization in the training notebook

## AI Models and Architecture

| Component | Purpose |
|---|---|
| Dense Neural Network | EMNIST character classification baseline |
| CNN | Improved EMNIST character classification |
| CNN INT8 | Quantized CNN available through a testing endpoint |
| Minimum Edit Distance | Spelling candidate generation |
| N-gram Language Model | Classical candidate ranking |
| BERT | Context-aware masked-language-model reranking |
| BERT INT8 | Quantized contextual correction in the Flask application |

**Current application pipeline:**

Handwritten Input → Image Preprocessing → Character Segmentation → Handwriting Recognition → Minimum Edit Distance → N-gram Candidates → Quantized BERT Reranking → Corrected Word

The Digit and Letter interface currently uses the existing recognition models. The separately trained EMNIST INT8 CNN is accessible through `/predict-emnist-test`. The application uses quantized BERT for contextual correction in Letter mode.

## Technologies

Python, Flask, TensorFlow, Keras, TensorFlow Lite, PyTorch, Hugging Face Transformers, NLTK, RapidFuzz, OpenCV, HTML, CSS, and JavaScript.

## Installation and Setup

**Tested development environment:** Windows, Python 3.13, CPU.

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SWE015-Handwritten-Answer-Digitizer
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Download the Brown Corpus

```bash
python -m nltk.downloader brown
```

### Run the application

python run.py

The launcher automatically downloads missing model weights
from Google Drive and installs them in the required folders.
An internet connection is required for the first setup.

**Model weights backup:**

[Download SWE015 Model Weights](https://drive.google.com/file/d/1kb03-IMr5MeoRTBEFF8pP3X2RxD7juCi/view?usp=sharing)

If automatic downloading fails, download the ZIP manually and extract:

- `digits_model.keras` and `letters_model.keras` into `models/`
- `swe015_cnn_int8.tflite` and `bert_int8_weights.pt` into `quantized_models/`

The BERT tokenizer and configuration may also require an internet connection on the first run.

Open the local application:

http://127.0.0.1:5001

## Using the Application

1. Select Digit or Letter mode.
2. Upload an image or draw a handwritten input.
3. In Letter mode, enter a sentence containing `[MASK]`.
4. Click **Predict**.
5. View the recognized characters, corrected word, confidence, and correction method.

**Example from a local test:**

Raw recognition: `HELO`

Sentence context: `I said [MASK] to my friend.`

Corrected word: `HELLO`

## Experimental Results

Models were trained and evaluated on EMNIST Balanced, containing 47 character classes.

| Model | Character Accuracy | Size | Average Inference Latency |
|---|---:|---:|---:|
| Dense Neural Network | 74.34% | 0.94 MB | 10.89 ms |
| CNN | 87.45% | 2.67 MB | 10.04 ms |
| CNN INT8 | 87.43% | 0.23 MB | 0.20 ms |

Latency values were obtained in Google Colab over 100 runs. The original Keras and INT8 TensorFlow Lite models used different inference paths, so their latency measurements are not a controlled like-for-like comparison.

### BERT Quantization

| Metric | BERT FP32 | BERT INT8 |
|---|---:|---:|
| Model Size | 417.84 MB | 195.57 MB |
| Average Latency | 196.87 ms | 109.64 ms |

BERT latency was measured in Google Colab on CPU over 100 runs using the same masked sentence.

The quantized model retained the same top candidate, `hello`, in the tested example. This single test does not establish equal overall word-correction accuracy.

## Evaluation Limitations

- Character recognition results are not equivalent to word-correction accuracy.
- The web interface's existing recognizers and the EMNIST-trained CNN are separate models.
- Handwriting accuracy depends on image quality, character segmentation, and preprocessing.
- BERT corrections depend on the sentence context.
- The current system is focused on short English words.
- Grad-CAM was generated in the training notebook but is not yet displayed in the web interface.

## Reproducibility

Training and experimental code are documented in `SWE015_Model_Training.ipynb`.

The original trained Dense and CNN models are separate from the models required to run the Flask application.

## AI Usage Statement

ChatGPT was used for coding assistance, troubleshooting, model integration guidance, and documentation drafting.

The experiments and measurements reported above were executed in the project environment. The project author is responsible for understanding, verifying, and explaining the submitted code and results.

## References

- [NIST – EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- [Hugging Face – BERT Base Uncased](https://huggingface.co/google-bert/bert-base-uncased)
- [NLTK Documentation](https://www.nltk.org/)
- [TensorFlow Lite Quantization](https://ai.google.dev/edge/litert/models/post_training_integer_quant)
- [PyTorch Quantization](https://docs.pytorch.org/docs/stable/quantization.html)
- [Flask Documentation](https://flask.palletsprojects.com/)