from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import time
import hashlib

app = FastAPI(title="Image Editor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RECAPTCHA_SECRET_KEY = "6LdXCBUsAAAAAOIFxbbQ83zmv3xM18ZUyC8IaOxa"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def verify_recaptcha(recaptcha_response: str) -> bool:
    return True


def apply_color_to_part(image_part: np.ndarray, color: str) -> np.ndarray:
    """Применяет выбранный цвет к части изображения"""
    colored_part = image_part.copy()

    if color == "red":
        # Усиливаем красный канал, ослабляем другие
        colored_part[:, :, 0] = np.minimum(colored_part[:, :, 0] + 50, 255)  # Красный
        colored_part[:, :, 1] = colored_part[:, :, 1] * 0.7  # Зеленый
        colored_part[:, :, 2] = colored_part[:, :, 2] * 0.7  # Синий
    elif color == "green":
        # Усиливаем зеленый канал, ослабляем другие
        colored_part[:, :, 0] = colored_part[:, :, 0] * 0.7  # Красный
        colored_part[:, :, 1] = np.minimum(colored_part[:, :, 1] + 50, 255)  # Зеленый
        colored_part[:, :, 2] = colored_part[:, :, 2] * 0.7  # Синий
    elif color == "blue":
        # Усиливаем синий канал, ослабляем другие
        colored_part[:, :, 0] = colored_part[:, :, 0] * 0.7  # Красный
        colored_part[:, :, 1] = colored_part[:, :, 1] * 0.7  # Зеленый
        colored_part[:, :, 2] = np.minimum(colored_part[:, :, 2] + 50, 255)  # Синий

    return colored_part


def swap_image_parts(image: Image.Image, operation: str, left_color: str, right_color: str) -> Image.Image:
    """Меняет местами части изображения с применением цветов"""
    try:
        width, height = image.size
        img_array = np.array(image)

        if operation == "swap_lr":
            # Меняем левую и правую половины
            mid = width // 2
            left_part = img_array[:, :mid].copy()
            right_part = img_array[:, mid:].copy()

            # Применяем цвета к частям
            left_part_colored = apply_color_to_part(left_part, left_color)
            right_part_colored = apply_color_to_part(right_part, right_color)

            # Убеждаемся что размеры совпадают
            if left_part.shape[1] != right_part.shape[1]:
                min_width = min(left_part.shape[1], right_part.shape[1])
                left_part_colored = left_part_colored[:, :min_width]
                right_part_colored = right_part_colored[:, :min_width]

            # Создаем новый массив с правильными размерами
            new_array = np.zeros_like(img_array)
            new_array[:, :mid] = right_part_colored  # Правая часть с цветом становится слева
            new_array[:,
            mid:mid + left_part_colored.shape[1]] = left_part_colored  # Левая часть с цветом становится справа

            return Image.fromarray(new_array)

        elif operation == "swap_ud":
            # Меняем верхнюю и нижнюю половины
            mid = height // 2
            upper_part = img_array[:mid, :].copy()
            lower_part = img_array[mid:, :].copy()

            # Применяем цвета к частям (для верхней/нижней используем left_color и right_color)
            upper_part_colored = apply_color_to_part(upper_part, left_color)
            lower_part_colored = apply_color_to_part(lower_part, right_color)

            # Убеждаемся что размеры совпадают
            if upper_part.shape[0] != lower_part.shape[0]:
                min_height = min(upper_part.shape[0], lower_part.shape[0])
                upper_part_colored = upper_part_colored[:min_height, :]
                lower_part_colored = lower_part_colored[:min_height, :]

            # Создаем новый массив
            new_array = np.zeros_like(img_array)
            new_array[:mid, :] = lower_part_colored
            new_array[mid:mid + upper_part_colored.shape[0], :] = upper_part_colored

            return Image.fromarray(new_array)

        return image

    except Exception as e:
        print(f"Error in swap_image_parts: {e}")
        return image


def create_color_histogram(image: Image.Image) -> Image.Image:
    """Создает и возвращает график распределения цветов (гистограмму)."""
    try:
        rgb_image = image.convert('RGB')
        r, g, b = rgb_image.split()

        r_hist = np.array(r.histogram())
        g_hist = np.array(g.histogram())
        b_hist = np.array(b.histogram())

        plt.figure(figsize=(8, 4))
        plt.plot(r_hist, color='red', label='Red', alpha=0.7)
        plt.plot(g_hist, color='green', label='Green', alpha=0.7)
        plt.plot(b_hist, color='blue', label='Blue', alpha=0.7)
        plt.title('Color Distribution Histogram')
        plt.xlabel('Color Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80)
        plt.close()
        buf.seek(0)

        histogram_image = Image.open(buf)
        return histogram_image.convert('RGB')

    except Exception as e:
        print(f"Error creating histogram: {e}")
        return Image.new('RGB', (400, 200), color='white')


def save_image_to_static(image: Image.Image, filename: str) -> str:
    """Сохраняет изображение в папку static и возвращает URL."""
    try:
        os.makedirs("static", exist_ok=True)

        timestamp = str(int(time.time()))
        unique_id = hashlib.md5(filename.encode()).hexdigest()[:8]
        clean_filename = f"{timestamp}_{unique_id}.jpg"

        static_path = f"static/{clean_filename}"

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(static_path, "JPEG")
        return f"/static/{clean_filename}"

    except Exception as e:
        print(f"Error saving image: {e}")
        error_path = "static/error.jpg"
        if not os.path.exists(error_path):
            error_img = Image.new('RGB', (100, 100), color='red')
            error_img.save(error_path, "JPEG")
        return "/static/error.jpg"


@app.post("/process", response_class=HTMLResponse)
async def process_image(
        request: Request,
        file: UploadFile = File(...),
        operation: str = Form(...),
        left_color: str = Form(...),
        right_color: str = Form(...),
        captcha_response: str = Form(...)
):
    """Обрабатывает загруженное изображение."""
    try:
        print(
            f"Processing image: {file.filename}, operation: {operation}, left_color: {left_color}, right_color: {right_color}")

        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Файл должен быть изображением")

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Файл пустой")

        try:
            original_image = Image.open(io.BytesIO(contents)).convert("RGB")
            print(f"Image size: {original_image.size}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Неверный формат изображения: {str(e)}")

        # Обрабатываем изображение с применением цветов
        processed_image = swap_image_parts(original_image, operation, left_color, right_color)

        # Создаем гистограмму
        histogram_image = create_color_histogram(original_image)

        # Сохраняем изображения
        original_url = save_image_to_static(original_image, f"original_{file.filename}")
        processed_url = save_image_to_static(processed_image, f"processed_{file.filename}")
        histogram_url = save_image_to_static(histogram_image, f"histogram_{file.filename}")

        print(f"Saved images: {original_url}, {processed_url}, {histogram_url}")

        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image": original_url,
            "processed_image": processed_url,
            "color_histogram": histogram_url
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in process_image: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)