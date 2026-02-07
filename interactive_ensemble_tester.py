import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
import os
import argparse

# --- Configuration ---
IMG_SIZE = 64
BRUSH_SIZE = 8

# Argument parser
parser = argparse.ArgumentParser(description='Interactive ENSEMBLE Tester.')
parser.add_argument('--runs', nargs='+', required=True,
                    help='Folder names of the models (e.g., expert_size expert_shape ...)')
parser.add_argument('--ensemble-weights', type=str, default=None,
                    help='Comma-separated weights (e.g. "0.6,0.2,0.2"). Must sum to 1.')
args = parser.parse_args()


def get_character_from_label(label):
    """
    Returns the actual character based on the numeric label.
    """
    label = int(label)

    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(ord('A') + (label - 10))
    elif 36 <= label <= 61:
        return chr(ord('a') + (label - 36))
    else:
        return "?"


class EnsembleApp:
    def __init__(self, root, run_names):
        self.root = root
        self.root.title(f"Ensemble Test ({len(run_names)} models)")

        # 1. Load Models
        self.models = []
        self.model_names = []

        print("\n--- LOADING MODELS ---")
        for name in run_names:
            path = os.path.join('results', name, 'best_model.keras')
            try:
                print(f"Loading: {name}...")
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                self.model_names.append(name)
            except Exception as e:
                print(f"ERROR: Failed to load: {path}\n{e}")
                self.root.destroy()
                return
        print("All models loaded!\n")

        # 2. Build UI
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_predict = tk.Button(btn_frame, text="ENSEMBLE PREDICT", command=self.predict,
                                     bg="#dddddd", font=("Helvetica", 12, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Clear", command=self.clear,
                                   bg="#ffcccc", font=("Helvetica", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.label_result = tk.Label(root, text="Draw something!", font=("Helvetica", 20, "bold"), fg="blue")
        self.label_result.pack(pady=20)

        self.label_details = tk.Label(root, text="", font=("Courier", 10), justify=tk.LEFT)
        self.label_details.pack(pady=5)

    def paint(self, event):
        # Match canvas (screen) to PIL (model input)
        # PIL draw.ellipse fills the circle. Canvas should do the same visually.
        r = BRUSH_SIZE // 2
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Draw something!")
        self.label_details.config(text="")

    def predict(self):
        # Preparation
        # 1. Invert (White bg -> Black bg)
        img_inverted = ImageOps.invert(self.image)
        
        img_resized = img_inverted.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # --- ENSEMBLE LOGIC ---
        all_probs = []
        details_text = "Individual Votes:\n"

        print("-" * 30)
        for i, model in enumerate(self.models):
            probs = model.predict(img_ready, verbose=0)[0]
            all_probs.append(probs)

            # Display individual prediction
            p_class = np.argmax(probs)
            p_char = get_character_from_label(p_class)
            p_conf = np.max(probs) * 100
            name_short = self.model_names[i].split('_')[0] + "..."

            print(f"{self.model_names[i]}: '{p_char}' ({p_class}) - {p_conf:.1f}%")
            details_text += f"{name_short}: '{p_char}' ({p_conf:.1f}%)\n"

        # Weighting
        weights = None
        
        if args.ensemble_weights:
            try:
                # Parse CLI weights
                w_list = [float(x) for x in args.ensemble_weights.split(',')]
                if len(w_list) != len(self.models):
                    print(f"WARNING: Weights count ({len(w_list)}) != Models count ({len(self.models)}). Ignoring weights.")
                elif not np.isclose(sum(w_list), 1.0):
                     print(f"WARNING: Weights sum to {sum(w_list)}, not 1.0. Ignoring weights.")
                else:
                    weights = w_list
            except ValueError:
                print("ERROR: Could not parse ensemble weights. Using defaults/average.")

        if weights is None and len(self.models) == 3:
            weights = [0.6, 0.2, 0.2]
            print(f"Using default weights for 3 models: {weights}")
            for i, name in enumerate(self.model_names):
                print(f"  - Model {i} ({name}): {weights[i]}")

        # Averaging
        avg_probs = np.average(np.array(all_probs), axis=0, weights=weights)

        final_class = np.argmax(avg_probs)
        final_char = get_character_from_label(final_class)
        final_conf = np.max(avg_probs) * 100

        print(f"==> FINAL DECISION: '{final_char}' ({final_class}) - {final_conf:.1f}%")

        # Display Result
        self.label_result.config(text=f"Prediction: '{final_char}' ({final_conf:.1f}%)")
        self.label_details.config(text=details_text)


if __name__ == "__main__":
    if not args.runs:
        print("ERROR: Please provide at least one model!")
    else:
        root = tk.Tk()
        app = EnsembleApp(root, args.runs)
        root.mainloop()