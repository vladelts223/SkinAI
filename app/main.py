import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import numpy as np
import cv2

MODEL_PATH = "model/fitzpatrick_vit_small_best.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["Fitzpatrick I", "Fitzpatrick II", "Fitzpatrick III", "Fitzpatrick IV", "Fitzpatrick V",
               "Fitzpatrick VI"]


def load_model():
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=6)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


model = load_model()

captured_features = None


def hook_fn(module, input, output):
    global captured_features
    captured_features = output.detach().cpu()


for name, module in model.named_modules():
    if name == "blocks.11":
        module.register_forward_hook(hook_fn)


def process_feature_map(image_pil, features):
    patches = features[0, 1:, :]
    importance = torch.mean(torch.abs(patches), dim=-1).numpy()
    size = int(np.sqrt(importance.shape[0]))
    attn_map = importance.reshape(size, size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = cv2.resize(attn_map, (image_pil.width, image_pil.height))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(image_pil.convert("RGB"))
    return cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DESCRIPTIONS = {
    "Fitzpatrick I": "Дуже світла шкіра. Завжди обгорає, ніколи не засмагає. Найвищий ризик фотопошкоджень.",
    "Fitzpatrick II": "Світла шкіра. Легко обгорає, засмагає мінімально. Високий ризик.",
    "Fitzpatrick III": "Середній тип. Помірно обгорає, засмагає до світло-коричневого кольору.",
    "Fitzpatrick IV": "Оливкова або світло-коричнева шкіра. Рідко обгорає, добре засмагає.",
    "Fitzpatrick V": "Темно-коричнева шкіра. Дуже рідко обгорає, засмагає інтенсивно.",
    "Fitzpatrick VI": "Чорна шкіра. Практично ніколи не обгорає. Найнижчий ризик опіків."
}


def predict(image):
    if image is None: return None, None, ""
    global captured_features
    captured_features = None

    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    top_class = CLASS_NAMES[torch.argmax(probs).item()]
    description = f"### ℹ️ Про цей тип:\n{DESCRIPTIONS[top_class]}"

    visual_result = process_feature_map(image, captured_features) if captured_features is not None else np.array(image)

    return confidences, visual_result, description


with gr.Blocks() as demo:
    gr.Markdown("# 🧬 Fitzpatrick Skin Type Detector")
    gr.Markdown("Vision Transformer аналізує текстуру та колір шкіри для визначення фототипу")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Фото шкіри")
            analyze_btn = gr.Button("🔍 Аналізувати", variant="primary")
            desc_output = gr.Markdown(label="Опис")

        with gr.Column():
            confidence_output = gr.Label(label="Визначений фототип", num_top_classes=3)
            attention_output = gr.Image(label="Карта фокусу моделі (XAI)")

    analyze_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[confidence_output, attention_output, desc_output]
    )

    gr.Markdown(
        """
        ---
        ⚠ **Застереження:**  
        Результат не є медичним діагнозом.  
        Візуалізація attention використовується для підвищення інтерпретованості моделі.
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())