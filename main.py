from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import requests
import os

app = FastAPI(title="Image Editor")

# Монтируем папки для статических файлов и шаблонов
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RECAPTCHA_SECRET_KEY = "6LdXCBUsAAAAAOIFxbbQ83zmv3xM18ZUyC8IaOxa"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница с формой загрузки."""
    return templates.TemplateResponse("index.html", {"request": request})


def verify_recaptcha(recaptcha_response: str) -> bool:
    """Проверяем ответ Google reCAPTCHA."""
    if not recaptcha_response:
        return False

    payload = {
        'secret': RECAPTCHA_SECRET_KEY,
        'response': recaptcha_response
    }
    try:
        response = requests.post('https://www.google.com/recaptcha/api/siteverify', data=payload, timeout=10)
        result = response.json()
        return result.get('success', False)
    except:
        return False


def swap_image_parts(image: Image.Image, operation: str) -> Image.Image:
    """Меняет местами части изображения."""
    width, height = image.size
    img_array = np.array(image)

    if operation == "swap_lr":
        # Меняем левую и правую половины
        mid = width // 2
        left_part = img_array[:, :mid].copy()
        right_part = img_array[:, mid:].copy()
        img_array[:, :mid] = right_part
        img_array[:, mid:] = left_part
    elif operation == "swap_ud":
        # Меняем верхнюю и нижнюю половины
        mid = height // 2
        upper_part = img_array[:mid, :].copy()
        lower_part = img_array[mid:, :].copy()
        img_array[:mid, :] = lower_part
        img_array[mid:, :] = upper_part

    return Image.fromarray(img_array)


def create_color_histogram(image: Image.Image) -> Image.Image:
    """Создает и возвращает график распределения цветов (гистограмму)."""
    # Конвертируем в RGB на случай, если изображение в другом режиме
    rgb_image = image.convert('RGB')
    r, g, b = rgb_image.split()

    # Считаем гистограммы для каждого канала
    r_hist = np.array(r.histogram())
    g_hist = np.array(g.histogram())
    b_hist = np.array(b.histogram())

    # Создаем график
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

    # Сохраняем график в буфер памяти
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    plt.close()
    buf.seek(0)
    return Image.open(buf)


def save_image_to_static(image: Image.Image, filename: str) -> str:
    """Сохраняет изображение в папку static и возвращает URL."""
    # Создаем папку если её нет
    os.makedirs("static", exist_ok=True)

    static_path = f"static/{filename}"
    image.save(static_path, "JPEG")
    return f"/static/{filename}"


@app.post("/process", response_class=HTMLResponse)
async def process_image(
        request: Request,
        file: UploadFile = File(...),
        operation: str = Form(...),
        captcha_response: str = Form(...)
):
    """Обрабатывает загруженное изображение: меняет части и строит гистограмму."""
    # 1. Проверяем CAPTCHA
    if not verify_recaptcha(captcha_response):
        raise HTTPException(status_code=400, detail="Ошибка проверки CAPTCHA. Пожалуйста, попробуйте снова.")

    # 2. Читаем и проверяем изображение
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой")

    try:
        original_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат изображения: {str(e)}")

    # 3. Обрабатываем изображение (меняем части)
    processed_image = swap_image_parts(original_image, operation)

    # 4. Создаем гистограмму цветов
    histogram_image = create_color_histogram(original_image)

    # 5. Сохраняем все изображения в static и получаем их URL
    import time
    timestamp = str(int(time.time()))

    original_url = save_image_to_static(original_image, f"original_{timestamp}_{file.filename}")
    processed_url = save_image_to_static(processed_image, f"processed_{timestamp}_{file.filename}")
    histogram_url = save_image_to_static(histogram_image, f"histogram_{timestamp}.png")

    # 6. Возвращаем шаблон с результатами
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image": original_url,
        "processed_image": processed_url,
        "color_histogram": histogram_url
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)