"""Additional tests for the Flask hand-sign recognition application."""

from io import BytesIO
import re
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

import app as app_module
from model import preprocess_img


ERROR_MESSAGE = b"File cannot be processed."


@pytest.fixture
def web_client():
    """Provide a Flask test client without starting a web server."""
    with app_module.app.test_client() as client:
        yield client


def create_image(width=100, height=100):
    """Create a valid RGB image in memory for uploading or preprocessing."""
    image = Image.new("RGB", (width, height), color=(255, 128, 0))
    image_file = BytesIO()
    image.save(image_file, format="PNG")

    # Return to the beginning so Pillow can read the image.
    image_file.seek(0)
    return image_file


# 1. HOME PAGE — NORMAL CASE
def test_home_page_loads(web_client):
    """The home page should display an image-upload form."""
    response = web_client.get("/")

    assert response.status_code == 200
    assert b"<form" in response.data
    assert b'name="file"' in response.data
    assert b'type="file"' in response.data


# 2. HOME PAGE — EDGE CASE
def test_home_page_with_unexpected_query(web_client):
    """An unused query parameter should not prevent the page from loading."""
    response = web_client.get("/?unexpected=value")

    assert response.status_code == 200
    assert b"<form" in response.data
    assert b'name="file"' in response.data


# 3. VALID PREDICTION — NORMAL CASE
def test_valid_upload_displays_prediction(web_client, monkeypatch):
    """A valid upload should display the digit returned by prediction."""
    # Replace prediction only; real image preprocessing still runs.
    fake_predict = Mock(return_value=5)
    monkeypatch.setattr(app_module, "predict_result", fake_predict)

    response = web_client.post(
        "/prediction",
        data={"file": (create_image(), "hand.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ERROR_MESSAGE not in response.data

    # Check the result heading, not just any occurrence of the digit.
    assert re.search(rb"<h2\b[^>]*>\s*5\s*</h2>", response.data)

    # Confirm the route actually called prediction with a processed image.
    fake_predict.assert_called_once()
    processed_image = fake_predict.call_args.args[0]
    assert processed_image.shape == (1, 224, 224, 3)


# 4. VALID PREDICTION — EDGE CASE
def test_zero_prediction_is_displayed(web_client, monkeypatch):
    """Zero is a valid prediction and must not disappear as a false value."""
    fake_predict = Mock(return_value=0)
    monkeypatch.setattr(app_module, "predict_result", fake_predict)

    response = web_client.post(
        "/prediction",
        data={"file": (create_image(), "zero.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ERROR_MESSAGE not in response.data
    assert re.search(rb"<h2\b[^>]*>\s*0\s*</h2>", response.data)
    fake_predict.assert_called_once()


# 5. MISSING FILE — TYPICAL ERROR CASE
def test_upload_without_file_field(web_client):
    """Submitting no file field should display the existing error message."""
    response = web_client.post(
        "/prediction",
        data={},
        content_type="multipart/form-data",
    )

    # The existing application returns its error page with status 200.
    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


# 6. MISSING FILE — EDGE CASE
def test_upload_with_empty_filename_and_content(web_client):
    """An empty file field should be handled without crashing."""
    response = web_client.post(
        "/prediction",
        data={"file": (BytesIO(b""), "")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


# 7. INVALID INPUT — TYPICAL ERROR CASE
def test_text_file_upload(web_client):
    """A plain text file should produce an image-processing error."""
    response = web_client.post(
        "/prediction",
        data={
            "file": (
                BytesIO(b"This is a text document."),
                "document.txt",
                "text/plain",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


# 8. INVALID INPUT — EDGE CASE
def test_invalid_content_disguised_as_jpeg(web_client):
    """An image filename and MIME type should not make invalid bytes valid."""
    response = web_client.post(
        "/prediction",
        data={
            "file": (
                BytesIO(b"This is not JPEG image data."),
                "fake.jpg",
                "image/jpeg",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert ERROR_MESSAGE in response.data


# 9. PREPROCESSING — NORMAL CASE
def test_standard_rgb_image_preprocessing():
    """A standard RGB image should be resized and normalized correctly."""
    processed_image = preprocess_img(create_image(100, 100))

    assert processed_image.shape == (1, 224, 224, 3)
    assert np.isfinite(processed_image).all()
    assert processed_image.min() >= 0
    assert processed_image.max() <= 1

    # Check a known pixel to confirm division by 255, not just the range.
    np.testing.assert_allclose(
        processed_image[0, 0, 0],
        [1.0, 128 / 255.0, 0.0],
        atol=1e-6,
    )


# 10. PREPROCESSING — EDGE CASE
def test_one_pixel_rgb_image_preprocessing():
    """A valid 1-by-1 image should resize to the model's required shape."""
    processed_image = preprocess_img(create_image(1, 1))

    assert processed_image.shape == (1, 224, 224, 3)
    assert np.isfinite(processed_image).all()
    assert processed_image.min() >= 0
    assert processed_image.max() <= 1

    # Resizing a single-colour image should preserve its colour throughout.
    expected_colour = np.array([1.0, 128 / 255.0, 0.0])
    assert np.allclose(processed_image, expected_colour, atol=1e-6)
