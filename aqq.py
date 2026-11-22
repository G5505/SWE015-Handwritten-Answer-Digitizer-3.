from flask import Flask, request, jsonify, render_template
import os
import base64
import numpy as np
import cv2
import tensorflow as tf
import string
from nlp_correction import correct_word
from bert_reranker import bert_correct_word

from keras.layers import Dense

old_from_config = Dense.from_config

@classmethod
def custom_from_config(cls, config):
    config.pop("quantization_config", None)
    return old_from_config(config)

Dense.from_config = custom_from_config

from utils.preprocess_digits import preprocess_digit_letter
from utils.preprocess_letters import preprocess_letter


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# SWE015 - Quantized CNN model

QUANTIZED_DIR = os.path.join(BASE_DIR, "quantized_models")

cnn_int8_path = os.path.join(
    QUANTIZED_DIR,
    "swe015_cnn_int8.tflite"
)

# Load INT8 model from bytes to avoid path encoding issues
with open(cnn_int8_path, "rb") as f:
    cnn_int8_bytes = f.read()

cnn_int8 = tf.lite.Interpreter(
    model_content=cnn_int8_bytes
)

cnn_int8.allocate_tensors()

cnn_input_details = cnn_int8.get_input_details()[0]
cnn_output_details = cnn_int8.get_output_details()[0]

print("SWE015 INT8 CNN loaded successfully!")

# Test actual INT8 CNN inference
test_image = np.zeros((1, 28, 28, 1), dtype=np.float32)

scale, zero_point = cnn_input_details["quantization"]

test_image_int8 = np.clip(
    np.round(test_image / scale + zero_point),
    -128, 127
).astype(np.int8)

cnn_int8.set_tensor(
    cnn_input_details["index"],
    test_image_int8
)

cnn_int8.invoke()

test_output = cnn_int8.get_tensor(
    cnn_output_details["index"]
)

print("INT8 CNN inference successful!")

# Predict one EMNIST character using INT8 CNN
def predict_emnist_int8(image_28x28):
    image = np.asarray(image_28x28, dtype=np.float32)
    image = image.reshape(1, 28, 28, 1)

    scale, zero_point = cnn_input_details["quantization"]

    quantized = np.clip(
        np.round(image / scale + zero_point),
        -128, 127
    ).astype(np.int8)

    cnn_int8.set_tensor(
        cnn_input_details["index"],
        quantized
    )

    cnn_int8.invoke()

    output = cnn_int8.get_tensor(
        cnn_output_details["index"]
    )[0]

    out_scale, out_zero = cnn_output_details["quantization"]

    probabilities = (
        output.astype(np.float32) - out_zero
    ) * out_scale

    predicted_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities)) * 100

    return predicted_class, confidence

print("Output shape:", test_output.shape)



digit_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "digits_model.keras")
)

letter_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "letters_model.keras")
)

#signature_encoder = tf.keras.models.load_model(
#    os.path.join(MODELS_DIR, "signature_encoder.keras")
#)

letters = string.ascii_uppercase

def split_characters(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) > 127:
        gray = 255 - gray

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if w > 5 and h > 10:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])

    chars = []

    for x, y, w, h in boxes:
        char_img = img[y:y+h, x:x+w]
        chars.append(char_img)

    return chars

def read_uploaded_image(field_name="image"):
    if field_name not in request.files:
        return None, jsonify({"error": f"No file provided for {field_name}"}), 400

    file = request.files[field_name]

    if file.filename == "":
        return None, jsonify({"error": f"No file selected for {field_name}"}), 400

    file_bytes = file.read()

    if not file_bytes:
        return None, jsonify({"error": f"Uploaded file for {field_name} is empty"}), 400

    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return None, jsonify({"error": f"Invalid image format in {field_name}"}), 400

    return img, None, None


def get_confidence(prediction):
    return float(np.max(prediction)) * 100.0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict-digit", methods=["POST"])
def predict_digit():
    img, error_response, status_code = read_uploaded_image("image")
    if error_response:
        return error_response, status_code

    chars = split_characters(img)

    if len(chars) == 0:
        chars = [img]

    result_digits = []
    confidences = []

    for char_img in chars:
        processed = preprocess_digit_letter(char_img)
        prediction = digit_model.predict(processed, verbose=0)[0]

        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        result_digits.append(str(predicted_digit))
        confidences.append(confidence)

    final_result = "".join(result_digits)
    avg_confidence = sum(confidences) / len(confidences)

    analysis = (
        f"The system detected {len(result_digits)} digit(s) and recognized them as {final_result}."
    )

    return jsonify({
        "prediction": final_result,
        "confidence": f"{avg_confidence:.2f}%",
        "analysis": analysis
    })


@app.route("/predict-letter", methods=["POST"])
def predict_letter():
    img, error_response, status_code = read_uploaded_image("image")
    if error_response:
        return error_response, status_code

    chars = split_characters(img)

    if len(chars) == 0:
        chars = [img]

    result_letters = []
    confidences = []

    for char_img in chars:
        processed = preprocess_letter(char_img)
        prediction = letter_model.predict(processed, verbose=0)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_letter = letters[predicted_index]
        confidence = float(np.max(prediction)) * 100

        result_letters.append(predicted_letter)
        confidences.append(confidence)

    final_result = "".join(result_letters)
    avg_confidence = sum(confidences) / len(confidences)

    analysis = (
        f"The system detected {len(result_letters)} character(s) "
        f"and recognized the text as {final_result}."
    )

    # SWE015: Generate classical NLP corrections
    candidates = correct_word(final_result)

    candidate_words = [
        word for word, edit_distance, score in candidates
    ]

    # SWE015 - Context-aware correction
    sentence = request.form.get(
        "sentence",
        "I said [MASK] to my friend."
    ).strip()

    corrected_word = final_result
    bert_candidates = []

    try:
        corrected_word, bert_candidates = bert_correct_word(
            final_result,
            sentence
        )
    except Exception as error:
        print("BERT correction error:", error)

    return jsonify({
        "prediction": final_result,
        "raw_word": final_result,
        "corrected_word": corrected_word.upper(),
        "confidence": f"{avg_confidence:.2f}%",
        "analysis": analysis,
        "sentence_context": sentence,
        "correction_method": "Minimum Edit Distance + N-gram + INT8 BERT",
        "correction_candidates": [
            item["word"] for item in bert_candidates[:5]
        ]
    })




# SWE015 - Test INT8 CNN with an uploaded image

@app.route("/predict-emnist-test", methods=["POST"])
def predict_emnist_test():
    img, error_response, status_code = read_uploaded_image("image")

    if error_response is not None:
        return error_response, status_code

    chars = split_characters(img)

    if len(chars) == 0:
        chars = [img]

    results = []

    for char_img in chars:
        processed = preprocess_letter(char_img)

        class_id, confidence = predict_emnist_int8(
            processed
        )

        results.append({
            "class_id": class_id,
            "confidence": round(confidence, 2)
        })

    return jsonify({
        "model": "EMNIST CNN INT8",
        "characters_detected": len(results),
        "results": results
    })


# SWE015 - Classical NLP correction endpoint

@app.route("/correct-word", methods=["POST"])
def correct_word_api():
    data = request.get_json(silent=True) or {}

    raw_word = str(data.get("word", "")).strip()

    if not raw_word:
        return jsonify({
            "error": "Please provide a word."
        }), 400

    candidates = correct_word(raw_word)

    return jsonify({
        "raw_word": raw_word,
        "method": "Minimum Edit Distance + N-gram",
        "candidates": [
            {
                "word": word,
                "edit_distance": edit_distance,
                "ngram_score": score
            }
            for word, edit_distance, score in candidates
        ]
    })



# SWE015 - Quantized BERT correction endpoint

@app.route("/correct-word-bert", methods=["POST"])
def correct_word_bert_api():
    data = request.get_json(silent=True) or {}

    raw_word = str(data.get("word", "")).strip()
    sentence = str(data.get("sentence", "")).strip()

    if not raw_word:
        return jsonify({"error": "Word is required."}), 400

    if sentence.count("[MASK]") != 1:
        return jsonify({
            "error": "Sentence must contain exactly one [MASK]."
        }), 400

    try:
        corrected, ranking = bert_correct_word(
            raw_word,
            sentence
        )
    except Exception as error:
        return jsonify({"error": str(error)}), 500

    return jsonify({
        "raw_word": raw_word,
        "corrected_word": corrected,
        "sentence_context": sentence,
        "method": "Minimum Edit Distance + N-gram + INT8 BERT",
        "top_candidates": ranking[:5]
    })



if __name__ == "__main__":
    print("REGISTERED ROUTES:")
    print(app.url_map)

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )
