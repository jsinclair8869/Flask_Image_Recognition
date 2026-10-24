# Hand Digit Image Recognition

This web app is made using Flask framework and is all about predicting the hand sign digit images based on the deep learning model trained using transfer learning.

![Python](https://img.shields.io/badge/python-8338ec)
![Keras](https://img.shields.io/badge/keras-219ebc)
![Numpy](https://img.shields.io/badge/numpy-ffafcc)
![Pillow](https://img.shields.io/badge/pillow-00bbf9)
![Flask](https://img.shields.io/badge/flask-f72585)
![Bootstrap](https://img.shields.io/badge/bootstrap-57cc99)

# Glimpse

https://user-images.githubusercontent.com/72191416/201943098-c8f5fd8b-ec7d-4e5d-883d-8b69109b946f.mp4

# Dependencies

- Flask
- TensorFlow (Keras)
- Numpy
- Pillow (PIL)

## UI Framework

- Bootstrap

## Setup

Install Git and Python 3.10 (64-bit) before starting.

### 1. Clone the repository

Open PowerShell in the folder where you want to save the project:

```powershell
git clone https://github.com/jsinclair8869/Flask_Image_Recognition.git
cd Flask_Image_Recognition
```

### 2. Create and activate a virtual environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The terminal should show `(.venv)` when the environment is active.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses TensorFlow 2.10.0. The requirements also pin Werkzeug to 2.3.8 to maintain compatibility with Flask 2.2.1.

## Running the Application

From the project folder, with the virtual environment active, run:

```powershell
python -m flask --app app run
```

Open http://127.0.0.1:5000 in your browser.

Keep the terminal open while using the application. Press `Ctrl+C` in the terminal to stop it.

### Making a Prediction

1. Choose a hand-sign digit image from the `test_images` folder.
2. Check the image preview.
3. Click **Submit**.
4. View the predicted digit on the results page.

### Setup Notes

- Run commands from the project folder so the application can find `digit_model.h5`.
- Do not commit the `.venv` folder to Git.
- TensorFlow may display a missing CUDA library warning on computers without a configured NVIDIA GPU. The application can still run using the CPU.
## Testing and Continuous Integration

### Running Checks Locally

With the virtual environment active, install the development tools:

```powershell
python -m pip install pylint pre-commit
python -m pre_commit install
```

Run code quality checks and all tests:

```powershell
python -m pylint app.py model.py
python -m pre_commit run --all-files
python -m pytest -v
```

To run only the additional assignment tests:

```powershell
python -m pytest test_assignment2.py -v
```

### Additional Test Coverage

The `test_assignment2.py` file adds 10 tests covering:

- Home page access and unexpected query parameters.
- Valid image uploads and displaying a prediction of zero.
- Missing uploads and empty filenames/content.
- Text uploads and invalid content disguised as a JPEG.
- Standard RGB image preprocessing and a one-pixel image.

Prediction responses are mocked in the upload success tests to make the expected results predictable. Image preprocessing is tested directly.

### GitHub Actions

The workflow in `.github/workflows/ci.yml` automatically checks pushes and pull requests targeting `main`.

It checks out the repository, sets up Python 3.10, installs dependencies, runs Pylint, and runs the complete pytest suite.

Workflow results and detailed logs are available in the repository's **Actions** tab.
