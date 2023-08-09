import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import torch
from pdf2image import convert_from_bytes
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import io
import tempfile

# Function to detect tables in an image using Table Transformer model
def detect_tables(image):
    image = Image.fromarray(image).convert("RGB")
    # Load the image
    w, h = image.size
    # Preprocess the image using DetrFeatureExtractor
    feature_extractor = DetrImageProcessor()
    encoding = feature_extractor(image, return_tensors="pt")

    # Load the Table Transformer model
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(**encoding)
    
    w, h = image.size
    # Post-process the output
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(h, w)])[0]

    table = [[int(coord) for coord in box] for box in results['boxes'].tolist()]
    return table

# Function to load and preprocess an image
def load_and_preprocess_image(image_file):
    img = Image.open(image_file)
    return img

# Function to perform OCR and visualize bounding boxes on tables
def perform_ocr(image, language='eng'):
    image_np = np.array(image)

    # Perform OCR using Pytesseract with output as data
    result = pytesseract.image_to_data(image_np, lang='eng', output_type=pytesseract.Output.DICT)
    # Process each word region in the result
    table = detect_tables(image_np)
    for box in table:
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    
    for i in range(len(result['text'])):
        word = result['text'][i]
        conf = int(result['conf'][i])
        (x, y, w, h) = (result['left'][i], result['top'][i], result['width'][i], result['height'][i])

        # Filter out low confidence words
        if conf > 20:
            skip_box = False
            for table_box in table:
                table_xmin, table_ymin, table_xmax, table_ymax = table_box
                if x > table_xmin and x + w < table_xmax and y > table_ymin and y + h < table_ymax:
                    skip_box = True
                    break
            
            if not skip_box:
                # Draw the bounding box rectangle
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 1)
    
    ocr_text = pytesseract.image_to_string(image, lang='eng') # Specify the language(s) for OCR
    return ocr_text, image_np

# Function to process a PDF file, perform OCR, and visualize pages
def process_pdf_file(uploaded_file, Analyze=False):
    pdf_file = io.BytesIO(uploaded_file.read())
    pdf_pages = convert_from_bytes(pdf_file.getbuffer())
    text_output = ""
    images = []

    for page_num, page in enumerate(pdf_pages):
        image = np.array(page)
        image_pil = Image.fromarray(image)
        ocr_text, img = perform_ocr(image_pil)
        if Analyze:
            st.image(img, caption=f"Page {page_num+1}", use_column_width=True)
        text_output += ocr_text + "\n\n"
        images.append(image_pil)

    return text_output.strip(), images

# Main function to run the Streamlit app
def main():
    st.title("Document Analysis System")
    uploaded_file = st.file_uploader("Upload file as PNG or PDF", type=["png", "pdf"])
    extracted_text = ""
    extracted_images = []

    if uploaded_file is not None:
        if uploaded_file.type == "image/png":
            image = load_and_preprocess_image(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("")
            st.write("processing...")
            extracted_text, img = perform_ocr(image)
            st.image(img, caption="Image with Bounding Boxes", use_column_width=True)

        elif uploaded_file.type == "application/pdf":
            button = st.button('Analyze')
            if button:
                extracted_text, extracted_images = process_pdf_file(uploaded_file, Analyze=True)
                st.write(f"Total Pages: {len(extracted_images)}")

        else:
            st.error("Invalid file format. Please upload a PNG image or a PDF file.")

        # Function to save extracted text to a temporary file
        def save_text_to_file(text):
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            temp_file.write(text)
            temp_file.close()
            with open(temp_file.name, 'r') as file:
                file_data = file.read()
            return file_data

        txt_download = save_text_to_file(extracted_text)
        st.download_button(
            "Download Extracted Text",
            data=txt_download,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

if __name__ == '__main__':
    main()
