# chuy-music-ai-gen
music ai gen 
# chuy music ai Generator — LSTM-Based Music Generation

> **Generating Original MIDI Music Using Deep Learning**

An end-to-end AI music generation project that uses a stacked LSTM neural network to generate original symbolic MIDI music. The system processes classical MIDI compositions, learns musical note and chord patterns, generates new sequences using temperature-based sampling, converts them into MIDI files, and provides an interactive Streamlit web application.

## Project Overview

This project focuses on **symbolic music generation** rather than direct audio synthesis.

MIDI compositions are processed using `music21`. Musical notes and chords are converted into stable tokens, and fixed-length sequences are created for next-token prediction. A stacked LSTM neural network learns the sequential relationships between these musical tokens.

During generation, the trained model receives a seed composition and predicts the next musical tokens one by one. Temperature-based sampling controls the creativity and randomness of the generated sequence. The final symbolic sequence is converted back into a playable MIDI file using `music21`.

The project is designed as a complete machine learning application covering:

* Dataset acquisition
* MIDI preprocessing
* Feature/token extraction
* Sequence generation
* LSTM model training
* Model inference
* Temperature-based sampling
* MIDI generation
* Streamlit application development
* Automated testing
* Reproducible deployment

## Key Features

| Capability             | Implementation                                    |
| ---------------------- | ------------------------------------------------- |
| MIDI Dataset           | Classical MIDI compositions with source metadata  |
| MIDI Processing        | `music21`                                         |
| Musical Representation | Notes and chords converted into tokens            |
| Vocabulary             | Token-to-integer and integer-to-token mappings    |
| Sequence Preparation   | Fixed-length sliding-window sequences             |
| Deep Learning Model    | Stacked LSTM neural network                       |
| Model Architecture     | Embedding → LSTM → Dropout → LSTM → Dense Softmax |
| Sampling               | Temperature-controlled probabilistic sampling     |
| MIDI Generation        | Generated sequences converted back to MIDI        |
| Web Application        | Streamlit                                         |
| Model Loading          | Cached model loading for efficient inference      |
| Testing                | Pytest-based unit tests                           |
| Reproducibility        | Training scripts and dependency requirements      |
| Output                 | Downloadable `.mid` music files                   |

## How the System Works

```text
Classical MIDI Dataset
        ↓
     music21
        ↓
Note & Chord Extraction
        ↓
Token Vocabulary
        ↓
Fixed-Length Training Sequences
        ↓
Embedding Layer
        ↓
LSTM Layer
        ↓
Dropout
        ↓
LSTM Layer
        ↓
Dropout
        ↓
Dense + Softmax
        ↓
Next-Token Prediction
        ↓
Temperature Sampling
        ↓
Generated Musical Sequence
        ↓
     music21
        ↓
Generated MIDI File
        ↓
Streamlit Download
```

## Repository Structure

```text
AI-Music-Generator/

├── app.py
├── train.py
├── generator.py
├── preprocessing.py
├── model.py
├── download_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── midi/
│       ├── .gitkeep
│       └── DATASET_SOURCE.json
│
├── models/
│   ├── config.json
│   ├── note_to_int.json
│   ├── int_to_note.json
│   └── music_generator.keras
│
├── generated/
│
├── notebooks/
│
└── tests/
    ├── test_preprocessing.py
    └── test_model_and_generation.py
```

## Project Files

### `app.py`

Streamlit application that provides the interactive music generation interface.

### `train.py`

Main training script responsible for:

* Loading MIDI files
* Preprocessing musical data
* Creating training sequences
* Training the LSTM model
* Saving model artifacts
* Saving training configuration and history

### `generator.py`

Handles:

* Loading the trained model
* Loading vocabulary mappings
* Seed selection
* Next-token prediction
* Temperature sampling
* MIDI generation

### `preprocessing.py`

Responsible for:

* Reading MIDI files
* Parsing notes
* Parsing chords
* Converting musical elements into tokens
* Creating vocabularies
* Creating fixed-length sequences

### `model.py`

Defines the TensorFlow/Keras LSTM architecture used for symbolic music generation.

### `download_dataset.py`

Downloads a small reproducible MIDI dataset and records source metadata.

### `app.py`

Provides the user-facing Streamlit music generation studio.

## Dataset

The project uses a small classical MIDI dataset for training.

The initial dataset can be acquired from the [Mutopia Project](https://www.mutopiaproject.org/), which provides classical sheet music and MIDI resources under different licensing conditions.

Dataset source information is recorded in:

```text
data/midi/DATASET_SOURCE.json
```

The metadata file helps maintain information about:

* Source URLs
* Downloaded files
* Dataset sources
* Licensing information

Always review the individual license associated with a MIDI composition before redistributing downloaded files.

## Installation

### 1. Clone the Repository

Clone the GitHub repository:

```bash
git clone <your-repository-url>
```

Then enter the project directory:

```bash
cd AI-Music-Generator
```

### 2. Create a Virtual Environment

Create a dedicated Python environment:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Acquisition

Download a small reproducible MIDI dataset:

```bash
python download_dataset.py --limit 8
```

The downloaded files are stored inside:

```text
data/midi/
```

The dataset source information is recorded in:

```text
data/midi/DATASET_SOURCE.json
```

## Training the Model

Train the LSTM model using:

```bash
python train.py --epochs 12 --batch-size 32 --sequence-length 32
```

The training pipeline performs the following operations:

1. Loads MIDI files from `data/midi/`.
2. Parses the compositions using `music21`.
3. Extracts notes and chords.
4. Converts musical elements into tokens.
5. Builds the token vocabulary.
6. Converts tokens into integer representations.
7. Creates fixed-length input sequences.
8. Creates next-token prediction targets.
9. Splits the dataset into training and validation data.
10. Builds the stacked LSTM model.
11. Trains the neural network.
12. Applies validation-based checkpointing and early stopping.
13. Saves the trained model and configuration files.

## Quick Training

For a faster local demonstration, use fewer epochs:

```bash
python train.py --epochs 3 --batch-size 16
```

This is useful for testing the complete pipeline before performing longer training.

## Model Architecture

The project uses a stacked LSTM architecture designed for sequential symbolic music generation.

```text
Input Token Sequence
        ↓
Embedding
        ↓
LSTM
        ↓
Dropout
        ↓
LSTM
        ↓
Dropout
        ↓
Dense Layer
        ↓
Softmax
        ↓
Next Musical Token
```

The model learns relationships between previous musical tokens and the probability distribution of the next token.

## Model Artifacts

After training, model-related files are stored inside the `models/` directory.

```text
models/

├── config.json
├── note_to_int.json
├── int_to_note.json
├── music_generator.keras
└── history.json
```

These files contain:

* Model configuration
* Token vocabulary
* Reverse vocabulary
* Trained neural network
* Training history

Large generated model files can be excluded from GitHub when necessary using `.gitignore`.

## Running the Streamlit Application

After training is completed, start the application:

```bash
streamlit run app.py
```

The application opens a browser-based music generation interface.

The interface provides:

* Seed composition selection
* Generation length control
* Temperature control
* Music generation button
* Generation status
* Generation information
* MIDI download button

## Temperature Sampling

Temperature controls the randomness of the generated music.

### Lower Temperature

Lower values make the model more conservative.

```text
Lower Temperature
        ↓
More predictable
        ↓
More consistent patterns
```

### Higher Temperature

Higher values increase variation.

```text
Higher Temperature
        ↓
More randomness
        ↓
More experimental output
```

The application provides a temperature control so users can experiment with different musical generation styles.

## MIDI Generation

The generated token sequence is converted back into a MIDI composition using `music21`.

The generated MIDI file can then be downloaded from the Streamlit interface.

Generated files are stored in:

```text
generated/
```

The output format is:

```text
.mid
```

MIDI files can be opened using compatible MIDI players, digital audio workstations, and music software.

## Testing

Run the complete test suite using:

```bash
pytest -q
```

The test suite validates important parts of the application, including:

* MIDI parsing
* Note extraction
* Chord extraction
* Token generation
* Vocabulary creation
* Sequence generation
* LSTM model construction
* Prediction output shape
* Temperature sampling
* Invalid sampling parameters
* Seed-based generation
* MIDI file creation
* Output file generation

## Error Handling

The application includes validation and error handling for common situations such as:

* Missing model files
* Missing vocabulary files
* Invalid generation parameters
* Invalid temperature values
* Missing MIDI seed files
* MIDI generation errors
* Model loading errors

Streamlit error messages are displayed when an operation cannot be completed.

## Performance Considerations

The model architecture is intentionally compact so that it can be trained and tested on CPU-oriented environments.

The Streamlit application uses model caching so that the trained model does not need to be loaded repeatedly during every interaction.

This improves application responsiveness and avoids unnecessary model initialization.

## Deployment

The project can be deployed to platforms that support Python and Streamlit applications.

The main application file is:

```text
app.py
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

Start the application using:

```bash
streamlit run app.py
```

For server environments, the application can be started using:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Deployment Requirements

A deployment environment should contain:

```text
app.py
requirements.txt
generator.py
preprocessing.py
model.py
models/
data/
```

If the trained model is not included in the deployment repository, it should be generated using the training pipeline before launching the application.

## GitHub Publication

The repository is structured to keep large generated files and unnecessary local files out of version control.

Recommended files to commit include:

```text
Source code
README.md
requirements.txt
.gitignore
Dataset metadata
Tests
Configuration files
```

Large generated artifacts such as trained model binaries, generated MIDI files, Python cache directories, and virtual environments should normally be excluded through `.gitignore`.

## Git Commands

Initialize Git if the repository has not already been initialized:

```bash
git init
```

Check the repository status:

```bash
git status
```

Add project files:

```bash
git add .
```

Create a commit:

```bash
git commit -m "Build LSTM-based AI music generator"
```

Set the main branch:

```bash
git branch -M main
```

Connect the GitHub repository:

```bash
git remote add origin https://github.com/<username>/AI-Music-Generator.git
```

Push the project:

```bash
git push -u origin main
```

For future updates:

```bash
git add .
git commit -m "Update AI music generator"
git push
```

## Requirements Coverage

| Component              | Status   |
| ---------------------- | -------- |
| MIDI Dataset           | Complete |
| MIDI Parsing           | Complete |
| Note Extraction        | Complete |
| Chord Extraction       | Complete |
| Token Vocabulary       | Complete |
| Sequence Generation    | Complete |
| LSTM Architecture      | Complete |
| Model Training         | Complete |
| Model Saving           | Complete |
| Model Inference        | Complete |
| Temperature Sampling   | Complete |
| MIDI Generation        | Complete |
| Streamlit Interface    | Complete |
| MIDI Download          | Complete |
| Error Handling         | Complete |
| Automated Testing      | Complete |
| Reproducible Training  | Complete |
| GitHub-Ready Structure | Complete |

## Future Improvements

Future versions of the project could include:

* Transformer-based music generation
* Multi-instrument music generation
* Genre-conditioned generation
* Style-conditioned generation
* Chord progression control
* Rhythm and duration modeling
* Longer musical context
* Improved musical coherence
* Real-time music generation
* Instrument selection
* MIDI-to-audio rendering
* Audio preview inside the web application
* User-customized seed melodies
* Interactive composition controls

## Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **music21**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Pytest**
* **Git**
* **GitHub**

## Learning Outcomes

This project demonstrates practical experience with:

* Deep learning
* Recurrent neural networks
* LSTM architectures
* Sequential data modeling
* Natural symbolic representation
* MIDI processing
* Music information representation
* Model training
* Model inference
* Probabilistic sampling
* Streamlit application development
* Testing and validation
* Git and GitHub
* Machine learning project deployment

## Project Highlights

### Deep Learning

A stacked LSTM neural network learns sequential relationships between musical notes and chords.

### Symbolic Music Generation

Instead of generating raw audio, the system works with symbolic MIDI representations.

### Interactive Generation

Users can control generation parameters through a Streamlit interface.

### Temperature Control

Users can adjust the creativity and randomness of the generated music.

### Downloadable Output

Generated compositions can be exported as MIDI files.

### Reproducible Pipeline

The project includes dataset acquisition, preprocessing, training, inference, testing, and deployment components.

## License and Dataset Notice

The source code of this project should be distributed according to the license selected by the project owner.

The MIDI dataset may contain files with different licensing terms. Dataset source information is maintained in:

```text
data/midi/DATASET_SOURCE.json
```

Review the license of each individual MIDI composition before redistributing the dataset or generated materials based on restricted source files.

## References

* [Mutopia Project](https://www.mutopiaproject.org/) — Free classical sheet music and MIDI resources.
* [music21](https://web.mit.edu/music21/) — Toolkit for computer-aided musicology.
* [TensorFlow](https://www.tensorflow.org/) — Machine learning framework.
* [Streamlit](https://streamlit.io/) — Framework for building data and AI applications.

---

## Author

**Anisha**

AI & Data Professional | Technology Innovator

Building intelligent solutions with AI, data, and technology to solve real-world problems.

---

**AI Music Generator — Turning Musical Patterns into New Compositions with Deep Learning.*
