import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os
import argparse

# --- Konfiguráció ---
IMG_SIZE = 32
BRUSH_SIZE = 10

# Argumentum parser
parser = argparse.ArgumentParser(description='Interaktív rajzolós tesztelő (Egy modellhez).')
parser.add_argument('--run_name', type=str, required=True,
                    help='A modell mappájának neve (pl. v9_regularized...)')
args = parser.parse_args()

MODEL_PATH = os.path.join('results', args.run_name, 'best_model.keras')


def get_character_from_label(label):
    """
    Visszaadja a valós karaktert a numerikus címke alapján.
    Szabály: 1-10: 0-9, 11-36: A-Z, 37-62: a-z
    """
    label = int(label)

    if 1 <= label <= 10:
        # Számok '0'-'9'
        return chr(ord('0') + (label - 1))
    elif 11 <= label <= 36:
        # Nagybetűk 'A'-'Z'
        return chr(ord('A') + (label - 11))
    elif 37 <= label <= 62:
        # Kisbetűk 'a'-'z'
        return chr(ord('a') + (label - 37))
    else:
        return "?"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Tesztelő: {args.run_name}")

        # 1. Modell betöltése
        print(f"Modell betöltése: {MODEL_PATH}...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Modell sikeresen betöltve!")
        except Exception as e:
            print(f"HIBA: Nem sikerült betölteni a modellt: {e}")
            self.root.destroy()
            return

        # 2. UI Felépítése
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_predict = tk.Button(btn_frame, text="JÓSLÁS", command=self.predict,
                                     bg="#dddddd", font=("Helvetica", 12, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Törlés", command=self.clear,
                                   bg="#ffcccc", font=("Helvetica", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.label_result = tk.Label(root, text="Rajzolj egy karaktert!", font=("Helvetica", 20, "bold"), fg="blue")
        self.label_result.pack(pady=20)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=BRUSH_SIZE)
        self.draw.ellipse([event.x - BRUSH_SIZE // 2, event.y - BRUSH_SIZE // 2,
                           event.x + BRUSH_SIZE // 2, event.y + BRUSH_SIZE // 2], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Rajzolj egy karaktert!")

    def predict(self):
        # Előkészítés (ugyanaz, mint a tanításnál)
        img_resized = self.image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Jóslás
        probs = self.model.predict(img_ready, verbose=0)[0]
        pred_class = np.argmax(probs)
        confidence = np.max(probs) * 100

        # Átalakítás karakterré
        pred_char = get_character_from_label(pred_class)

        # Kiírás
        result_text = f"Tipp: '{pred_char}' ({confidence:.1f}%)"
        self.label_result.config(text=result_text)
        print(f"Jóslat: '{pred_char}' (Class: {pred_class}), Biztosság: {confidence:.2f}%")


if __name__ == "__main__":
    if not args.run_name:
        print("HIBA: Add meg a modell nevét! Pl: python interactive_tester.py --run_name v9_...")
    else:
        root = tk.Tk()
        app = App(root)
        root.mainloop()