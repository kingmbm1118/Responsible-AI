# AI Content Detector

## Overview

The AI Content Detector is a Streamlit application designed to analyze and classify whether content (text, images, or videos) is AI-generated or real. Utilizing pre-trained machine learning models, the application provides predictions with visual indicators for each result.

## Features

- **Text Analysis**: Detects if the provided text is AI-generated or human-written.
- **Image Analysis**: Classifies images to identify if they were AI-generated or captured by a camera.
- **Video Analysis**: Analyzes video frames to predict whether the content was generated by AI.
- **Probability Scores**: Displays the probability of each prediction.

## Technologies Used

- **Streamlit**: Interactive web application framework.
- **Transformers**: Model pipeline from Hugging Face for text and image classification.
- **OpenCV**: Video frame extraction.
- **Pillow (PIL)**: Image processing library.
- **MoviePy**: For handling video files.

## How It Works

### Text Detection

- Users input text to analyze.
- A RoBERTa-based model classifies the text as REAL or FAKE.

### Image Detection

- Users upload an image file.
- A specialized image-classification model classifies the image.

### Video Detection

- Users upload a video file.
- Frames are extracted from the video and analyzed individually. Results for each frame are displayed.

## Installation

### Prerequisites

- Python 3.8 or later
- pip package manager

### Steps

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd ai-content-detector
   ```
2.Create a virtual environment:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3.Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```

4.Run the application:
   ```bash
    streamlit run app.py
   ```

## Directory Structure
   ```bash
.
├── app.py                 # Main Streamlit app
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation
   ```
## Usage
Select the type of content to analyze (Text, Image, or Video) from the sidebar.
Upload the file or input text.
Click the "Analyze" button to start the detection process.
View results with probability scores and visual labels.


###  Example
| Example | Input | Output |
|-------|-----------|--------|
| Text Analysis | "This is a sample text." | Prediction: REAL, Probability: 0.98 |
| Image Analysis | Upload an image file. | Prediction: FAKE, Probability: 0.85 |
| Video Analysis | Upload a video file. | Results are displayed for each extracted frame |


## Responsible AI
Responsible AI ensures that AI systems are developed and deployed ethically, minimizing biases and fostering trust. This application aligns with responsible AI principles by providing transparent predictions and encouraging fair use of AI technologies.

## Authors

### Mahmoud Boghdady
   [https://www.linkedin.com/in/mahmoud-boghdady-msc-pmp%C2%AE-6b694033/](https://www.linkedin.com/in/mahmoud-boghdady-msc-pmp%C2%AE-6b694033/)

### Abu Bakar Rasheed
[https://www.linkedin.com/in/abu-bakar-rasheed-9b65b616/](https://www.linkedin.com/in/abu-bakar-rasheed-9b65b616/)

## License
This project is licensed under the MIT License. See the LICENSE file for details.
