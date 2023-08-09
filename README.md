# DocAI_App

The project is a "Document Analysis System" that allows users to upload PNG images or PDF files. It performs Optical Character Recognition (OCR) on the uploaded documents to extract text and visualize tables using bounding boxes. The system uses the Streamlit framework to create a user-friendly interface for interacting with the code.

**Code Explanation:**

1. **Import Statements:** The code starts by importing the required libraries and modules,and various utilities from the Hugging Face Transformers library.

2. **Function `detect_tables`:** This function takes an image and uses the `DetrImageProcessor` and `TableTransformerForObjectDetection` models to detect tables within the image. It preprocesses the image, passes it through the model, and then post-processes the output to extract bounding box coordinates of the detected tables.

3. **Function `load_and_preprocess_image`:** This function loads and preprocesses an image using the PIL library.

4. **Function `perform_ocr`:** This function performs OCR on an image using PyTesseract. It also utilizes the `detect_tables` function to identify tables and visualize them using bounding boxes.

5. **Function `process_pdf_file`:** This function processes a PDF file. It converts the PDF pages into images, performs OCR using the `perform_ocr` function, and optionally visualizes the images if the `Analyze` flag is set to true.

6. **Function `main`:** This is the main function that sets up the Streamlit app. It displays a title and provides a file uploader for users to upload PNG images or PDF files. Depending on the uploaded file type, the code performs OCR and table detection. It also handles displaying images and extracted text, and provides a button to analyze PDF pages if needed. Additionally, it allows users to download the extracted text as a `.txt` file.

7. **`if __name__ == '__main__':` block:** This block of code ensures that the `main` function is executed only when the script is run directly (not imported as a module).

**Setup:**

To run the code, you need to set up a Python environment with the required dependencies installed. Here's what you need to do:

1. **Install Required Libraries:**
   - Open a terminal or command prompt.
   - Run the following command to install the required libraries:
     ```
     pip install -r requirements.txt
     ```

2. **Download PyTesseract Data:**
   - You need to download the language data for PyTesseract. You can do this by visiting the following link: https://github.com/tesseract-ocr/tessdata
   - Download the `eng.traineddata` file and place it in a directory. Note the path to this directory.

3. **Run the Code:**
   - Save the provided code in a `.py` file (e.g., `app.py`).
   - Open a terminal or command prompt and navigate to the directory containing the code file.
   - Run the following command to start the Streamlit app:
     ```
     streamlit run app.py
     ```

4. **Upload Files:**
   - In the Streamlit app, you can upload PNG images or PDF files using the file uploader.

**In Summary:**
The project involves building a Document Analysis System using the Streamlit framework. The code uses various libraries for image processing, OCR, and table detection. It offers an interactive interface for users to upload documents, extract text, visualize text and tables, and download the extracted text. The setup involves installing dependencies and running the code using the Streamlit command.