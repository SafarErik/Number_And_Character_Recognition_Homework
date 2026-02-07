import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import os
import argparse

# --- IMPORT: Load from src folder! ---
from src.models import build_deep_hybrid_cnn

# --- Configuration ---
IMG_SIZE = 64
BRUSH_SIZE = 18
NUM_CLASSES = 63  # 0-62 classes

parser = argparse.ArgumentParser(description='Interactive Drawing Tester.')
parser.add_argument('--run_name', type=str, required=True,
                    help='Folder name of the model (e.g. v16_...)')
args = parser.parse_args()

MODEL_PATH = os.path.join('results', args.run_name, 'best_model.keras')


def get_character_from_label(label):
    """
    Returns the actual character based on the numeric label.
    """
    label = int(label)
    if 1 <= label <= 10:
        return chr(ord('0') + (label - 1))
    elif 11 <= label <= 36:
        return chr(ord('A') + (label - 11))
    elif 37 <= label <= 62:
        return chr(ord('a') + (label - 37))
    else:
        return "?"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Tester: {args.run_name}")

        print(f"Building model structure and loading weights from: {MODEL_PATH}...")

        try:
            # STEP 1: Build the empty model from code
            self.model = build_deep_hybrid_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=NUM_CLASSES)

            # STEP 2: Load only the weights
            self.model.load_weights(MODEL_PATH)

            print("✅ Model and weights loaded successfully!")

        except Exception as e:
            print(f"\nCRITICAL ERROR: Failed to load weights: {e}")
            print("Check if you are using the 'deep_hybrid' model!")
            self.root.destroy()
            return

        # Build UI
        self.canvas = tk.Canvas(root, width=320, height=320, bg='white', cursor="cross")
        self.canvas.pack(pady=10)

        # PIL image: White background (255) - For human eyes
        self.image = Image.new("L", (320, 320), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_predict = tk.Button(btn_frame, text="PREDICT", command=self.predict,
                                     bg="#dddddd", font=("Helvetica", 12, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear,
                                   bg="#ffcccc", font=("Helvetica", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.label_result = tk.Label(root, text="Draw a character!", font=("Helvetica", 20, "bold"), fg="blue")
        self.label_result.pack(pady=20)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=BRUSH_SIZE, outline="black")
        self.draw.ellipse([event.x - BRUSH_SIZE // 2, event.y - BRUSH_SIZE // 2,
                           event.x + BRUSH_SIZE // 2, event.y + BRUSH_SIZE // 2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (320, 320), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw a character!")

    def predict(self):
        # 1. Invert (Black background, White char - as the model learned)
        img_inverted = ImageOps.invert(self.image)

        # 2. Resize
        img_resized = img_inverted.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

        # 3. Normalize
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # 4. Predict
        probs = self.model.predict(img_ready, verbose=0)[0]
        pred_class = np.argmax(probs)
        confidence = np.max(probs) * 100

        pred_char = get_character_from_label(pred_class)

        fg_color = "green" if confidence > 80 else "orange" if confidence > 50 else "red"
        result_text = f"Prediction: '{pred_char}'\n({confidence:.1f}%)"

        self.label_result.config(text=result_text, fg=fg_color)
        print(f"Result: '{pred_char}' (Class ID: {pred_class}), Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    if not args.run_name:
        print("ERROR: Usage: python interactive_tester.py --run_name FOLDER_NAME")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()