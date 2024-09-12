# Django Web Application for Spell Correction and Error Detection

## Overview

This web application is built using Django and serves two main purposes:
1. **Spell Correction** for non-native learners of the Persian language.
2. **Error Detection** for identifying grammatical, syntactic, and spelling errors in Persian text using ParsBERT.

The application provides two endpoints:
- `localhost/spell_correction`: Endpoint for interacting with the spell correction model.
- `localhost/error_detection`: Endpoint for interacting with the error detection model.

## Features

- **Spell Correction**: Automatically corrects spelling mistakes in Persian text, with a focus on errors made by non-native speakers. Additionally, the model provides insights based on the learner's nationality to better tailor corrections.
- **Error Detection**: Detects and highlights various language errors (spelling, grammar, and syntax) in Persian sentences using the ParsBERT model.

## Project Structure

The project follows the standard Django structure:   
. ├── myproject/ │ ├── settings.py │ ├── urls.py │ ├── views.py │ ├── models.py │ └── templates/ │ └── base.html ├── spell_correction/ │ ├── urls.py │ ├── views.py │ └── models.py ├── error_detection/ │ ├── urls.py │ ├── views.py │ └── models.py ├── manage.py └── 
