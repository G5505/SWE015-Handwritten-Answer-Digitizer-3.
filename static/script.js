
const imageInput = document.getElementById("imageInput");
const predictBtn = document.getElementById("predictBtn");
const resetBtn = document.getElementById("resetBtn");

const resultText = document.getElementById("resultText");
const confidenceText = document.getElementById("confidenceText");
const confidenceBar = document.getElementById("confidenceBar");
const analysisText = document.getElementById("analysisText");
const diffImage = document.getElementById("diffImage");

const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");

const statusBox = document.getElementById("statusBox");
const taskSelect = document.getElementById("taskSelect");
const tabButtons = document.querySelectorAll(".tab-btn");

const drawCanvas = document.getElementById("drawCanvas");
const clearCanvasBtn = document.getElementById("clearCanvasBtn");
const ctx = drawCanvas.getContext("2d");

let isDrawing = false;
let hasDrawing = false;

// Initialize drawing canvas
function initCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.strokeStyle = "black";
}

initCanvas();

// Drawing
drawCanvas.addEventListener("mousedown", (e) => {
  isDrawing = true;
  hasDrawing = true;
  previewPlaceholder.style.display = "none";

  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});

drawCanvas.addEventListener("mousemove", (e) => {
  if (!isDrawing) return;

  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
});

drawCanvas.addEventListener("mouseup", () => {
  isDrawing = false;
});

drawCanvas.addEventListener("mouseleave", () => {
  isDrawing = false;
});

// Clear drawing
function clearCanvas() {
  initCanvas();
  hasDrawing = false;

  previewImage.removeAttribute("src");
  previewImage.style.display = "none";

  drawCanvas.style.display = "block";
  previewPlaceholder.style.display = "flex";
}

clearCanvasBtn.addEventListener("click", () => {
  imageInput.value = "";
  clearCanvas();
});

// Switch between Digit and Letter
tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((btn) => btn.classList.remove("active"));
    button.classList.add("active");

    taskSelect.value = button.dataset.task;
    clearAllResults();
    setTaskMessage();
  });
});

function setTaskMessage() {
  if (taskSelect.value === "digit") {
    statusBox.textContent =
      "Digit mode selected. Upload or draw a handwritten number.";
  } else {
    statusBox.textContent =
      "Letter mode selected. Upload or draw handwritten characters.";
  }
}

// Confidence display
function setConfidence(valueText) {
  confidenceText.textContent = valueText || "--";

  if (!valueText || valueText === "--") {
    confidenceBar.style.width = "0%";
    return;
  }

  const numericValue = parseFloat(valueText.replace("%", ""));

  confidenceBar.style.width = numericValue + "%";

  if (numericValue < 50) {
    confidenceBar.style.background = "#e74c3c";
  } else if (numericValue < 80) {
    confidenceBar.style.background = "#f1c40f";
  } else {
    confidenceBar.style.background = "#2ecc71";
  }
}

// Reset results
function clearAllResults() {
  resultText.textContent = "No result yet.";
  resultText.className = "result-main";
  setConfidence("--");

  analysisText.textContent =
    "Upload an image or draw, then press predict.";

  if (diffImage) {
    diffImage.removeAttribute("src");
    diffImage.style.display = "none";
  }
}

function resetEverything() {
  imageInput.value = "";
  clearCanvas();
  clearAllResults();
  setTaskMessage();
}

resetBtn.addEventListener("click", resetEverything);

// Image upload preview
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];

  if (file) {
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";

    drawCanvas.style.display = "none";
    previewPlaceholder.style.display = "none";

    hasDrawing = false;
    statusBox.textContent = `Selected file: ${file.name}`;
  } else {
    clearCanvas();
  }
});

// Convert drawing to PNG
function canvasToBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/png");
  });
}

// Predict digit or letter
predictBtn.addEventListener("click", async () => {
  const task = taskSelect.value;
  const file = imageInput.files[0];

  if (!file && !hasDrawing) {
    resultText.textContent = "No input selected";
    resultText.className = "result-main error-text";
    analysisText.textContent =
      "Please upload an image or draw first.";
    return;
  }

  const formData = new FormData();

  if (task === "letter") {
    const sentence = document.getElementById("sentenceContext").value;
    formData.append("sentence", sentence);
  }


  if (file) {
    formData.append("image", file);
  } else {
    const blob = await canvasToBlob(drawCanvas);
    if (!blob) {
      analysisText.textContent = "Could not read drawing.";
      return;
    }
    formData.append("image", blob, "drawing.png");
  }

  const endpoint =
    task === "digit" ? "/predict-digit" : "/predict-letter";

  resultText.textContent = "Processing...";
  analysisText.textContent =
    "Please wait while the system analyzes the input.";
  setConfidence("--");

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    resultText.textContent = data.prediction || "No prediction";
    resultText.className = "result-main";

    setConfidence(data.confidence || "--");
    
    if (task === "letter" && data.corrected_word) {
      analysisText.textContent =
        `Raw Recognition: ${data.raw_word}\n` +
        `Corrected Word: ${data.corrected_word}\n` +
        `Method: ${data.correction_method}`;

      analysisText.style.whiteSpace = "pre-line";
      resultText.textContent =
        `${data.raw_word} → ${data.corrected_word}`;
    } else {
      analysisText.textContent =
        data.analysis || "No analysis available.";
    }


    if (diffImage) {
      if (data.diff_image) {
        diffImage.src =
          "data:image/png;base64," + data.diff_image;
        diffImage.style.display = "block";
      } else {
        diffImage.style.display = "none";
      }
    }

  } catch (error) {
    resultText.textContent = "Error";
    resultText.className = "result-main error-text";
    analysisText.textContent = error.message;
    console.error(error);
  }
});

setTaskMessage();
