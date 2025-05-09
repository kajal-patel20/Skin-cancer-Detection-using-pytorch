# Skin Cancer Classification with Heatmap

This project implements a skin cancer classification system using a ResNet-50 model. It includes Grad-CAM visualization to highlight the areas of the image that influenced the model's predictions.

## Features

-   **Skin Cancer Classification**: Classifies skin lesion images into predefined categories.
-   **Grad-CAM Visualization**: Generates heatmaps to show which areas of the image influenced the prediction.
-   **Web Interface**: Upload images and view predictions and heatmaps through a user-friendly web interface.

## Dataset

The dataset used for this project is the **HAM10000** dataset. You can download it from the following link:

[HAM10000 Dataset](https://assets.supervisely.com/remote/eyJsaW5rIjogImZzOi8vYXNzZXRzLzEzMThfU2tpbiBDYW5jZXI6IEhBTTEwMDAwL3NraW4tY2FuY2VyOi1oYW0xMDAwMC1EYXRhc2V0TmluamEudGFyIiwgInNpZyI6ICJ0MG9heU0xaGQrQSs1ZnBDMjc5ei9ncCt5REJDYVBPS1RXeFQ4dkRKZzBVPSJ9)

After downloading, organize the dataset as follows:

```
ds/
├── img/  # Contains all image files
├── ann/  # Contains all annotation JSON files
```

## Setup Instructions

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/jacktheboss220/train.git
    cd train
    ```

2. **Create a Virtual Environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download and Prepare the Dataset**:

    - Download the dataset from the link above.
    - Place the dataset in the `ds/` directory as described.

5. **Train the Model**:
   Run the training script to train the model:

    ```bash
    python trainModel.py
    ```

6. **Run the Web Application**:
   Start the Flask web application:
    ```bash
    python app.py
    ```
    Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage

-   Upload an image of a skin lesion through the web interface.
-   View the classification result and the Grad-CAM heatmap.

## File Structure

```
train/
├── app.py                # Flask web application
├── trainModel.py         # Model training script
├── templates/            # HTML templates for the web interface
├── static/               # Static files (CSS, JS)
├── ds/                   # Dataset directory (images and annotations)
├── checkpoints/          # Directory for saving model checkpoints
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
