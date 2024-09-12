# Django Web Application for Spell Correction and Error Detection

## Overview

This web application is built using Django and serves two main purposes:
1. **Spell Correction** for non-native learners of the Persian language.
2. **Error Detection** for identifying grammatical, syntactic, and spelling errors in Persian text using ParsBERT.

The application provides two endpoints:
- `localhost/`: Endpoint for interacting with the spell correction model.
- `localhost/error_detection`: Endpoint for interacting with the error detection model.

## Features

- **Spell Correction**: Automatically corrects spelling mistakes in Persian text, with a focus on errors made by non-native speakers. Additionally, the model provides insights based on the learner's nationality to better tailor corrections.
- **Error Detection**: Detects and highlights various language errors (spelling, grammar, and syntax) in Persian sentences using the ParsBERT model.

 **final_django/**: Contains the core configuration files for the Django project.
- **spell_correction/**: Application that handles spell correction tasks.
- **error_detection/**: Application for detecting language errors using ParsBERT.
- **templates/**: HTML templates used for rendering the web interface.
  
## Setup Instructions

### Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.x
- Django 3.x or higher
- TensorFlow (for spell correction model)
- PyTorch (for ParsBERT in error detection)
- Other dependencies listed in the `requirements.txt` file.

### Installation

1. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run database migrations**:
    ```bash
    python manage.py migrate
    ```

4. **Start the Django development server**:
    ```bash
    python manage.py runserver
    ```
### URLs

- **Spell Correction**: Access the spell correction model at:
  ```url
  http://localhost:8000
  ```
- **Error Detection**: Access the Error detection model at:
```url
http://localhost:8000/error_detection
```


## Future Enhancements

- **Improved User Interface**: Enhance the user interface for a more interactive and user-friendly experience. This includes better styling, error highlighting, and responsive design.

- **Model Tuning and Optimization**: Further fine-tuning of both the spell correction and error detection models using larger datasets and more sophisticated algorithms to improve accuracy and performance.

- **Multilingual Support**: Expand the application's capabilities by adding support for spell correction and error detection in other languages beyond Persian.

- **Real-Time Feedback**: Implement real-time text input analysis for spell correction and error detection, providing instant feedback as the user types.

- **Integration with Educational Tools**: Link the application to language learning platforms, providing learners with additional resources for improving their writing skills in Persian.

- **API for External Use**: Create an API that allows other applications to integrate spell correction and error detection functionalities into their own platforms.
