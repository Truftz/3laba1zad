from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)


def test_read_root():
    """Тест главной страницы."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Обработчик изображений" in response.text


def test_swap_image_logic():
    """Тест логики обмена частей изображения."""
    from PIL import Image
    import numpy as np
    from main import swap_image_parts

    # Создаем тестовое изображение 4x4 пикселя
    test_img = Image.new('RGB', (4, 4), color='red')
    # Заливаем правую половину синим
    np_img = np.array(test_img)
    np_img[:, 2:] = [0, 0, 255]  # Синий цвет
    test_img = Image.fromarray(np_img)

    # Проверяем, что функция выполняется без ошибок
    try:
        result_img = swap_image_parts(test_img, "swap_lr")
        assert result_img is not None
        assert result_img.size == test_img.size
        print("Logic test passed!")
    except Exception as e:
        pytest.fail(f"Logic test failed: {e}")


def test_histogram_creation():
    """Тест создания гистограммы."""
    from PIL import Image
    from main import create_color_histogram

    # Создаем тестовое изображение
    test_img = Image.new('RGB', (10, 10), color='red')

    try:
        histogram = create_color_histogram(test_img)
        assert histogram is not None
        assert histogram.size[0] > 0  # Проверяем что изображение создано
        print("Histogram test passed!")
    except Exception as e:
        pytest.fail(f"Histogram test failed: {e}")


def test_invalid_file_upload():
    """Тест загрузки невалидного файла."""
    # Пытаемся загрузить текстовый файл вместо изображения
    response = client.post(
        "/process",
        data={"operation": "swap_lr", "captcha_response": "test"},
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_missing_captcha():
    """Тест без CAPTCHA."""
    # Создаем тестовое изображение
    from PIL import Image
    import io

    test_img = Image.new('RGB', (10, 10), color='red')
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    response = client.post(
        "/process",
        data={"operation": "swap_lr", "captcha_response": ""},  # Пустая CAPTCHA
        files={"file": ("test.jpg", img_bytes.getvalue(), "image/jpeg")}
    )
    # Должна быть ошибка валидации
    assert response.status_code == 400