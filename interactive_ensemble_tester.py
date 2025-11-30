import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os
import argparse

# --- Konfiguráció ---
IMG_SIZE = 32
BRUSH_SIZE = 8

# Argumentum parser
parser = argparse.ArgumentParser(description='Interaktív ENSEMBLE tesztelő.')
parser.add_argument('--runs', nargs='+', required=True,
                    help='A modellek mappanevei (pl. expert_size expert_shape ...)')
args = parser.parse_args()


def get_character_from_label(label):
    """
    Visszaadja a valós karaktert a numerikus címke alapján.
    """
    label = int(label)

    if 1 <= label <= 10:
        # Számok '0'-'9' (Label 1 -> '0')
        return chr(ord('0') + (label - 1))

    elif 11 <= label <= 36:
        # Nagybetűk 'A'-'Z' (Label 11 -> 'A')
        return chr(ord('A') + (label - 11))

    elif 37 <= label <= 62:
        # Kisbetűk 'a'-'z' (Label 37 -> 'a')
        return chr(ord('a') + (label - 37))

    else:
        return "?"


class EnsembleApp:
    def __init__(self, root, run_names):
        self.root = root
        self.root.title(f"Ensemble Teszt ({len(run_names)} modell)")

        # 1. Modellek betöltése
        self.models = []
        self.model_names = []

        print("\n--- MODELLEK BETÖLTÉSE ---")
        for name in run_names:
            path = os.path.join('results', name, 'best_model.keras')
            try:
                print(f"Betöltés: {name}...")
                model = tf.keras.models.load_model(path)
                self.models.append(model)
                self.model_names.append(name)
            except Exception as e:
                print(f"HIBA: Nem sikerült betölteni: {path}\n{e}")
                self.root.destroy()
                return
        print("Minden modell betöltve!\n")

        # 2. UI Felépítése
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (280, 280), 255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        self.btn_predict = tk.Button(btn_frame, text="ENSEMBLE JÓSLÁS", command=self.predict,
                                     bg="#dddddd", font=("Helvetica", 12, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="Törlés", command=self.clear,
                                   bg="#ffcccc", font=("Helvetica", 12))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        self.label_result = tk.Label(root, text="Rajzolj!", font=("Helvetica", 20, "bold"), fg="blue")
        self.label_result.pack(pady=20)

        self.label_details = tk.Label(root, text="", font=("Courier", 10), justify=tk.LEFT)
        self.label_details.pack(pady=5)

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
        self.label_result.config(text="Rajzolj!")
        self.label_details.config(text="")

    def predict(self):
        # Előkészítés
        img_resized = self.image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_ready = img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # --- ENSEMBLE LOGIKA ---
        all_probs = []
        details_text = "Egyéni szavazatok:\n"

        print("-" * 30)
        for i, model in enumerate(self.models):
            probs = model.predict(img_ready, verbose=0)[0]
            all_probs.append(probs)

            # Egyéni tipp megjelenítése
            p_class = np.argmax(probs)
            p_char = get_character_from_label(p_class)  # Karakterré alakítás
            p_conf = np.max(probs) * 100
            name_short = self.model_names[i].split('_')[0] + "..."

            print(f"{self.model_names[i]}: '{p_char}' ({p_class}) - {p_conf:.1f}%")
            details_text += f"{name_short}: '{p_char}' ({p_conf:.1f}%)\n"

        # Súlyozás
        if len(self.models) == 3:
            weights = [0.6, 0.2, 0.2]
        else:
            weights = None

        # Átlagolás
        avg_probs = np.average(np.array(all_probs), axis=0, weights=weights)

        final_class = np.argmax(avg_probs)
        final_char = get_character_from_label(final_class)  # Karakterré alakítás
        final_conf = np.max(avg_probs) * 100

        print(f"==> VÉGSŐ DÖNTÉS: '{final_char}' ({final_class}) - {final_conf:.1f}%")

        # Kiírás
        self.label_result.config(text=f"Tipp: '{final_char}' ({final_conf:.1f}%)")
        self.label_details.config(text=details_text)


if __name__ == "__main__":
    if not args.runs:
        print("HIBA: Adj meg legalább egy modellt!")
    else:
        root = tk.Tk()
        app = EnsembleApp(root, args.runs)
        root.mainloop()