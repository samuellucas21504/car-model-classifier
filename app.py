import io
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageClassifierApp:
    """
    Janela interativa para classificação de imagens usando EfficientNetB4.

    - Carrega modelo Keras salvo em arquivo .keras
    - Mostra arquitetura (summary) do modelo
    - Mostra ativações de camadas convolucionais internas
    """

    # === CONFIGURAÇÕES ALINHADAS AO BASELINE ===
    MODEL_PATH = "models/EfficientNetB4/EfficientNetB4_baseline_best.keras"
    DATASET_TRAIN_DIR = "dataset/train"  # mesmo diretório usado no baseline
    IMG_HEIGHT, IMG_WIDTH = 380, 380     # igual ao baseline EfficientNetB4
    IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("EfficientNetB4 Image Classifier")
        self.master.geometry("1200x800")
        self.master.configure(bg="#202531")

        # Estado interno
        self.model = None
        self.activation_model = None
        self.activation_layer_names = []
        self.current_activation_layer_index = 0
        self.current_image_array = None
        self.display_photo = None
        self.activation_canvas = None

        # CLASS_NAMES será carregado a partir de dataset/train
        self.CLASS_NAMES = self._load_class_names_from_dir()

        # Layout + modelo
        self._build_layout()
        self._load_model()

    # ==========================================================
    # LOAD DE CLASSES
    # ==========================================================
    def _load_class_names_from_dir(self):
        """
        Lê as subpastas de dataset/train e ordena alfabeticamente.

        Isso replica o comportamento do flow_from_directory, que usou
        essas pastas para criar o mapeamento class_indices no baseline.
        """
        train_dir = self.DATASET_TRAIN_DIR
        if not os.path.isdir(train_dir):
            print(f"[AVISO] Diretório '{train_dir}' não encontrado. "
                  f"Definindo CLASS_NAMES vazio.")
            return []

        # Apenas diretórios
        class_names = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        class_names = sorted(class_names)
        print("CLASS_NAMES carregado de dataset/train:", class_names)
        return class_names

    # ==========================================================
    # Layout
    # ==========================================================
    def _build_layout(self):
        self.left_frame = tk.Frame(self.master, bg="#202531")
        self.right_frame = tk.Frame(self.master, bg="#202531")
        self.bottom_frame = tk.Frame(self.master, bg="#202531")

        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # ---------------- LEFT: imagem + botões + predição ----------------
        title_label = tk.Label(
            self.left_frame,
            text="Classificador de Imagens\nEfficientNetB4 (baseline)",
            bg="#202531",
            fg="#ffffff",
            font=("Helvetica", 18, "bold"),
            justify=tk.CENTER,
        )
        title_label.pack(pady=(0, 10))

        self.image_frame = tk.Frame(self.left_frame, bg="#1e2430", bd=2, relief=tk.RIDGE)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.image_label = tk.Label(self.image_frame, bg="#1e2430")
        self.image_label.pack(expand=True)

        button_frame = tk.Frame(self.left_frame, bg="#202531")
        button_frame.pack(fill=tk.X, pady=(10, 5))

        self.load_button = tk.Button(
            button_frame,
            text="Carregar imagem",
            command=self.load_image,
            bg="#3b82f6",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))

        self.predict_button = tk.Button(
            button_frame,
            text="Prever classe",
            command=self.predict_image,
            bg="#22c55e",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.activations_button = tk.Button(
            button_frame,
            text="Ativações (próx. camada)",
            command=self.show_next_activation_layer,
            bg="#f97316",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=5,
        )
        self.activations_button.pack(side=tk.LEFT, padx=5)

        self.prediction_frame = tk.Frame(self.left_frame, bg="#111827", bd=2, relief=tk.GROOVE)
        self.prediction_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(
            self.prediction_frame,
            text="Predição (Top 3):",
            bg="#111827",
            fg="#e5e7eb",
            font=("Helvetica", 11, "bold"),
        ).pack(anchor="w", padx=5, pady=(3, 0))

        self.prediction_label = tk.Label(
            self.prediction_frame,
            text="Carregue uma imagem e clique em 'Prever classe'.",
            bg="#111827",
            fg="#9ca3af",
            justify=tk.LEFT,
            font=("Consolas", 10),
        )
        self.prediction_label.pack(anchor="w", padx=5, pady=3)

        # ---------------- RIGHT: arquitetura ----------------
        arch_title = tk.Label(
            self.right_frame,
            text="Arquitetura do Modelo (summary)",
            bg="#202531",
            fg="#ffffff",
            font=("Helvetica", 14, "bold"),
        )
        arch_title.pack(anchor="w", pady=(0, 5))

        self.arch_text_frame = tk.Frame(self.right_frame, bg="#111827", bd=2, relief=tk.GROOVE)
        self.arch_text_frame.pack(fill=tk.BOTH, expand=True)

        self.arch_text = tk.Text(
            self.arch_text_frame,
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            font=("Consolas", 9),
            wrap=tk.NONE,
        )
        self.arch_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        arch_scroll_y = tk.Scrollbar(self.arch_text_frame, command=self.arch_text.yview)
        arch_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.arch_text.configure(yscrollcommand=arch_scroll_y.set)

        arch_scroll_x = tk.Scrollbar(self.right_frame, orient=tk.HORIZONTAL, command=self.arch_text.xview)
        arch_scroll_x.pack(fill=tk.X)
        self.arch_text.configure(xscrollcommand=arch_scroll_x.set)

        # ---------------- BOTTOM: ativações ----------------
        bottom_title = tk.Label(
            self.bottom_frame,
            text="Ativações de Camadas Internas",
            bg="#202531",
            fg="#ffffff",
            font=("Helvetica", 14, "bold"),
        )
        bottom_title.pack(anchor="w", pady=(0, 5))

        self.activation_info_label = tk.Label(
            self.bottom_frame,
            text="Carregue uma imagem, clique em 'Prever classe' e depois em 'Ativações'.",
            bg="#202531",
            fg="#9ca3af",
            font=("Helvetica", 10),
        )
        self.activation_info_label.pack(anchor="w")

        self.activation_frame = tk.Frame(self.bottom_frame, bg="#111827", bd=2, relief=tk.GROOVE)
        self.activation_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

    # ==========================================================
    # Modelo / arquitetura
    # ==========================================================
    def _load_model(self):
        try:
            self.model = load_model(self.MODEL_PATH, compile=False)
            self._show_model_summary()
            self._build_activation_model()
        except Exception as e:
            messagebox.showerror("Erro ao carregar modelo", f"Não foi possível carregar o modelo:\n{e}")
            print(e)

    def _show_model_summary(self):
        if self.model is None:
            return
        buffer = io.StringIO()
        self.model.summary(print_fn=lambda x: buffer.write(x + "\n"))
        summary_str = buffer.getvalue()
        self.arch_text.delete("1.0", tk.END)
        self.arch_text.insert(tk.END, summary_str)

    def _build_activation_model(self):
        if self.model is None:
            return

        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or \
                    isinstance(layer, tf.keras.layers.SeparableConv2D) or \
                    isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                conv_layers.append(layer)

        if not conv_layers:
            self.activation_info_label.config(
                text="Nenhuma camada convolucional encontrada para ativação."
            )
            return

        selected_layers = []
        selected_names = []
        indices = sorted(set([0, 1, len(conv_layers) - 1]))
        for idx in indices:
            selected_layers.append(conv_layers[idx].output)
            selected_names.append(conv_layers[idx].name)

        self.activation_layer_names = selected_names
        self.activation_model = tf.keras.Model(self.model.input, selected_layers)
        self.current_activation_layer_index = 0

    # ==========================================================
    # Carregar / mostrar imagem
    # ==========================================================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("Todos os arquivos", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("RGB")

            img_resized = img.resize(self.IMG_SIZE)
            img_array = np.array(img_resized, dtype=np.float32)
            self.current_image_array = img_array

            max_preview_size = (350, 350)
            img_preview = img.copy()
            img_preview.thumbnail(max_preview_size)
            self.display_photo = ImageTk.PhotoImage(img_preview)
            self.image_label.config(image=self.display_photo)

            self.prediction_label.config(
                text="Imagem carregada. Clique em 'Prever classe'."
            )
            self.activation_info_label.config(
                text="Depois da predição, clique em 'Ativações (próx. camada)'."
            )
            self._clear_activation_canvas()

        except Exception as e:
            messagebox.showerror("Erro ao carregar imagem", f"Não foi possível abrir a imagem:\n{e}")
            print(e)

    # ==========================================================
    # Predição
    # ==========================================================
    def _get_preprocessed_batch(self):
        if self.current_image_array is None:
            return None
        x = np.array(self.current_image_array, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        # MESMO PREPROCESS DO BASELINE
        x = preprocess_input(x)
        return x

    def predict_image(self):
        if self.model is None:
            messagebox.showwarning("Modelo não carregado", "O modelo ainda não foi carregado.")
            return
        if self.current_image_array is None:
            messagebox.showwarning("Nenhuma imagem", "Carregue uma imagem primeiro.")
            return

        if not self.CLASS_NAMES:
            self.prediction_label.config(
                text="⚠️ CLASS_NAMES está vazio. Verifique o diretório dataset/train "
                     "ou defina manualmente as classes."
            )
            return

        try:
            x = self._get_preprocessed_batch()
            preds = self.model.predict(x)[0]

            num_classes = preds.shape[-1]
            if len(self.CLASS_NAMES) != num_classes:
                self.prediction_label.config(
                    text=f"⚠️ Número de classes do modelo ({num_classes}) "
                         f"não bate com CLASS_NAMES ({len(self.CLASS_NAMES)}).\n"
                         f"Verifique se dataset/train tem as mesmas pastas usadas no treino."
                )
                return

            top_indices = preds.argsort()[-3:][::-1]
            result_lines = []
            for i in top_indices:
                cls_name = self.CLASS_NAMES[i]
                prob = preds[i] * 100
                result_lines.append(f"{cls_name:20s} -> {prob:6.2f}%")

            result_str = "\n".join(result_lines)
            self.prediction_label.config(text=result_str)

        except Exception as e:
            messagebox.showerror("Erro na predição", f"Ocorreu um erro ao prever:\n{e}")
            print(e)

    # ==========================================================
    # Ativações internas
    # ==========================================================
    def _clear_activation_canvas(self):
        if self.activation_canvas is not None:
            self.activation_canvas.get_tk_widget().destroy()
            self.activation_canvas = None

    def show_next_activation_layer(self):
        if self.activation_model is None or not self.activation_layer_names:
            self.activation_info_label.config(
                text="Modelo de ativações não disponível. Verifique se o modelo foi carregado corretamente."
            )
            return
        if self.current_image_array is None:
            messagebox.showwarning("Nenhuma imagem", "Carregue uma imagem antes de visualizar as ativações.")
            return

        self.current_activation_layer_index = (
                                                      self.current_activation_layer_index + 1
                                              ) % len(self.activation_layer_names)

        self._show_activations_for_current_layer()

    def _show_activations_for_current_layer(self):
        self._clear_activation_canvas()

        x = self._get_preprocessed_batch()
        if x is None:
            return

        activations = self.activation_model.predict(x)
        layer_index = self.current_activation_layer_index
        feature_maps = activations[layer_index][0]  # (H, W, C)
        layer_name = self.activation_layer_names[layer_index]

        n_maps = min(8, feature_maps.shape[-1])
        n_rows = 2
        n_cols = 4

        fig = Figure(figsize=(6, 3))
        axes = fig.subplots(n_rows, n_cols)

        idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if idx < n_maps:
                    ax.imshow(feature_maps[:, :, idx])
                    ax.set_title(f"Filtro {idx}", fontsize=8)
                ax.axis("off")
                idx += 1

        fig.suptitle(f"Ativações - camada: {layer_name}", fontsize=10)

        self.activation_canvas = FigureCanvasTkAgg(fig, master=self.activation_frame)
        self.activation_canvas.draw()
        self.activation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.activation_info_label.config(
            text=f"Visualizando ativações da camada '{layer_name}' "
                 f"({layer_index + 1}/{len(self.activation_layer_names)})."
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
