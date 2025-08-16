from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
import tiktoken
from dotenv import load_dotenv
from utils import generate_embeddings
import hashlib
import json
import base64
import streamlit as st
import csv
import re
import fitz  # PyMuPDF
import requests
from io import StringIO
import logging
import pandas as pd
import zipfile
import tempfile
import numpy as np
logging.getLogger("watchdog").setLevel(logging.ERROR)

load_dotenv()

# Configuration
API_URL = "https://router.huggingface.co/nscale/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
}

def extract_and_process_zip(zip_path, user, private):
    """
    Extract ZIP file and process all PDF files inside it.
    
    Parameters:
        zip_path (str): Path to the ZIP file.
        user (str): Username for private processing.
        private (bool): If True, save to user's private folder; if False, save to public uploads.
    
    Returns:
        list: List of successfully processed PDF filenames.
    """
    processed_pdfs = []
    
    try:
        # Determine output folders based on private flag
        if private:
            base_output_path = f"data/user_uploads/{user}"
            image_output_path = os.path.join(base_output_path, "extracted_images")
            csv_output_path = os.path.join(base_output_path, "csv_files")
        else:
            base_output_path = "data/public_uploads"
            image_output_path = os.path.join(base_output_path, "extracted_images")
            csv_output_path = os.path.join(base_output_path, "csv_files")
        
        # Create base directories
        os.makedirs(base_output_path, exist_ok=True)
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find all PDF files in extracted content (including subdirectories)
            pdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            print(f"Found {len(pdf_files)} PDF files in ZIP")
            
            # Process each PDF file (extract images only)
            for pdf_path in pdf_files:
                try:
                    print(f"Step 1: Extracting images from {os.path.basename(pdf_path)}...")
                    extracted_images = extract_images_from_pdf(pdf_path, image_output_path)
                    
                    if extracted_images:
                        processed_pdfs.append(os.path.basename(pdf_path))
                        print(f"Successfully extracted images from: {os.path.basename(pdf_path)}")
                    
                except Exception as e:
                    print(f"Error processing {os.path.basename(pdf_path)}: {e}")
                    continue
            
            # After processing all PDFs, create individual CSVs and one final merged file
            if processed_pdfs:
                print("Step 2: Processing all images to individual CSV files...")
                individual_csvs = process_images_to_individual_csv(image_output_path, csv_output_path, processed_pdfs)
                
                if individual_csvs:
                    print("Step 3: Creating final merged CSV for all PDFs...")
                    final_merged_csv = create_final_merged_csv(csv_output_path)
                    
                    if final_merged_csv:
                        fix_term_career_totals(final_merged_csv, final_merged_csv)
                        print(f"Final merged CSV for ZIP: {os.path.basename(final_merged_csv)}")
        
        # Clean up the original ZIP file
        try:
            os.remove(zip_path)
        except:
            pass
        
        return processed_pdfs
        
    except Exception as e:
        print(f"Error extracting ZIP file {zip_path}: {e}")
        return []
    
def process_images_to_individual_csv(image_folder_path: str, csv_output_path: str, processed_pdf_names: list = None) -> list:
    """
    Process all images in the folder and convert them to individual CSV files only.
    
    Parameters:
        image_folder_path (str): Path to folder containing extracted images.
        csv_output_path (str): Path to folder where CSV files will be saved.
        processed_pdf_names (list): List of processed PDF names.
    
    Returns:
        list: List of individual CSV file paths created.
    """
    try:
        # Create CSV folder if it doesn't exist
        os.makedirs(csv_output_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("No image files found to process.")
            return []
        
        processed_files = []
        
        # Loop through each image and process it
        for image_file in image_files:
            image_path = os.path.join(image_folder_path, image_file)
            image_path = os.path.abspath(image_path).replace("\\", "/")
            print(f"Processing: {image_file}")
            try:
                response = query_with_local_image(image_path)
                if response and "choices" in response:
                    content = response["choices"][0]["message"]['content']
                    csv_file_path = save_transcript_to_csv(content, image_file, csv_output_path)
                    if csv_file_path:
                        processed_files.append(csv_file_path)
                else:
                    print(f"Invalid response for {image_file}")
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
        
        return processed_files
            
    except Exception as e:
        print(f"Error processing images to individual CSVs: {e}")
        return []

def create_final_merged_csv(csv_output_path: str) -> str:
    """
    Creates a single merged CSV from all individual CSV files in the folder.
    
    Parameters:
        csv_output_path (str): Path to the CSV output folder.
    
    Returns:
        str: Path to the final merged CSV file.
    """
    try:
        csv_files = [f for f in os.listdir(csv_output_path) 
                    if f.endswith('.csv') and not f.startswith('merged_')]
        
        if not csv_files:
            print("No individual CSV files found to merge.")
            return None
        
        # Extract unique PDF names from CSV filenames for merged filename
        pdf_names = set()
        for csv_file in csv_files:
            # Extract PDF name from filename like "Barrett_Trista_page_1.csv"
            base_name = os.path.splitext(csv_file)[0]
            # Remove page suffix
            pdf_name = re.sub(r'_page_\d+$', '', base_name)
            # Fix multiple underscores
            pdf_name = re.sub(r'_+', '_', pdf_name)
            pdf_names.add(pdf_name)
        
        # Create merged filename
        sorted_names = sorted(list(pdf_names))
        merged_filename = f"merged_{'_'.join(sorted_names)}.csv"
        # Fix double underscores
        merged_filename = re.sub(r'_+', '_', merged_filename)
        merged_filepath = os.path.join(csv_output_path, merged_filename)
        
        header_written = False
        total_rows = 0
        
        with open(merged_filepath, 'w', newline='', encoding='utf-8') as merged_file:
            merged_writer = csv.writer(merged_file)
            
            for csv_file in sorted(csv_files):  # Sort for consistent order
                csv_filepath = os.path.join(csv_output_path, csv_file)
                print(f"Merging: {csv_file}")
                
                try:
                    with open(csv_filepath, 'r', encoding='utf-8') as individual_file:
                        csv_reader = csv.reader(individual_file)
                        rows = list(csv_reader)
                        
                        if rows:
                            if not header_written:
                                # Write header from first file
                                merged_writer.writerow(rows[0])
                                header_written = True
                                # Write all rows including data rows
                                for row in rows[1:]:
                                    if row:  # Skip empty rows
                                        merged_writer.writerow(row)
                                        total_rows += 1
                            else:
                                # Skip header row for subsequent files, write only data rows
                                for row in rows[1:]:
                                    if row:  # Skip empty rows
                                        merged_writer.writerow(row)
                                        total_rows += 1
                                        
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue
        
        print(f"Successfully merged {len(csv_files)} CSV files")
        print(f"Total data rows merged: {total_rows}")
        print(f"Final merged file: {os.path.basename(merged_filepath)}")
        
        return merged_filepath
        
    except Exception as e:
        print(f"Error creating final merged CSV: {e}")
        return None

def search_query(user_query, collection, top_k=3):
    """Search ChromaDB for relevant documents based on user query"""
    # Generate embedding for the query
    query_embedding = generate_embeddings(user_query)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    retrieved_chunks = results['documents'][0]  # Top k chunks
    chunk_metadata = results['metadatas'][0]    # Metadata for each chunk
    distances = results['distances'][0]         # Distance scores
    
    # Get the unique titles from the retrieved chunks
    retrieved_titles = list(set([metadata['title'] for metadata in chunk_metadata]))
    
    return retrieved_titles, retrieved_chunks, distances

def generate_answer(user_query, retrieved_chunks, tab_data, communication_language):
    """
    Generates an answer to the user's query using the LLaMA model (via ChatGroq).
    """
    chunk_context = "\n\n".join(retrieved_chunks)

    fallback_messages = {
    "English": "I'm sorry, but that question is outside the scope of the provided information.",
    "Spanish": "Lo siento, pero esa pregunta está fuera del alcance de la información proporcionada.",
    "French": "Je suis désolé, mais cette question dépasse le cadre des informations fournies."
    }
    fallback_response = fallback_messages.get(communication_language, fallback_messages["English"])

    prompt = f"""
                You are an expert assistant helping users. 
                Answer the user's question primarily using the information provided below.
                
                ---

                ### Provided Information:
                {chunk_context}

                ---

                ### Question:
                {user_query}

                ---

                ### Instructions for Answering:
                - The language of communication must be in user chosen {communication_language} language, you must respond in {communication_language} language.
                - First, check if the answer is found in the provided information:
                  * If the answer IS found in the provided information, respond clearly using that information.
                  * If the answer is PARTIALLY found, use what's available from the documents and clearly indicate which parts of your response come from the provided information.
                  * If the answer is NOT found in the provided information, you may provide a helpful response based on your general knowledge, but preface it with: "This information is not found in the provided documents. Based on general knowledge: "
                - Use a warm and helpful tone.
                - Use bullet points, bold text, or headings if it improves clarity.
                - Always be transparent about the source of your information (documents vs. general knowledge).
                - If you're completely uncertain about information outside the provided documents, acknowledge the limitations with: "I don't have specific information about this in the provided documents or in my general knowledge. {fallback_response}"

                ---

                ### Answer:
                """

    llm = ChatGroq(
        model="Llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=4192,
        timeout=60,
        max_retries=2,
    )

    # Count tokens in the prompt
    token_count = count_tokens(prompt)
    print(f"📊 Sending {token_count} tokens to the LLM")

    # Check if we're close to the limit
    if token_count > 6000:
        print(f"⚠️ WARNING: Token count ({token_count}) is approaching or exceeding Groq's limit of 6000 TPM")
        
    response = llm.invoke(prompt)
    return response

def count_tokens(text, model="cl100k_base"):
    """Count the number of tokens in a text string using tiktoken"""
    try:
        encoder = tiktoken.get_encoding(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Rough estimation if tiktoken fails
        return len(text.split()) * 1.3
    

# File to store hash metadata
METADATA_FILE = "data/metadata.json"

# Function to calculate hash of a file
def calculate_file_hash(file_path):
    """Calculate MD5 hash of file to detect changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Function to load or initialize metadata
def get_metadata():
    """Load metadata from file or create default"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    else:
        return {"tab_data_hash": "", "last_updated": ""}

# Function to save metadata
def save_metadata(metadata):
    """Save metadata to file"""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

# Function to load CSS from file
def load_css(css_file):
    with open(css_file, 'r') as f:
        return f.read()

# Function to load HTML template from file
def load_html_template(template_file):
    with open(template_file, 'r') as f:
        return f.read()
    
# Function to get base64 encoded image
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
# --- Load SVG as base64 ---
def load_svg_base64(svg_path):
    with open(svg_path, "rb") as f:
        svg_data = f.read()
    return base64.b64encode(svg_data).decode("utf-8")

def parse_pdf_to_individual_csv(pdf_path, image_output_path, csv_output_path):
    """
    Parse PDF and create individual CSV files only (no merging).
    
    Parameters:
        pdf_path (str): Path to the input PDF file.
        image_output_path (str): Path to save extracted images.
        csv_output_path (str): Path to save individual CSV files.
    
    Returns:
        str: PDF filename if successful, None if failed.
    """
    try:
        # Step 1: Extract images from PDF
        print(f"Step 1: Extracting images from {os.path.basename(pdf_path)}...")
        extracted_images = extract_images_from_pdf(pdf_path, image_output_path)
        
        if not extracted_images:
            print(f"Failed to extract images from {os.path.basename(pdf_path)}")
            return None
        
        # Step 2: Process images to individual CSV files only
        print(f"Step 2: Processing images to individual CSV files...")
        individual_csvs = process_images_to_individual_csv(image_output_path, csv_output_path, [os.path.basename(pdf_path)])
        
        if not individual_csvs:
            print(f"No CSV files were created from {os.path.basename(pdf_path)}")
            return None
        
        print(f"Successfully created {len(individual_csvs)} individual CSV files from {os.path.basename(pdf_path)}")
        return os.path.basename(pdf_path)
        
    except Exception as e:
        print(f"Error in parse_pdf_to_individual_csv: {e}")
        return None

def extract_and_process_zip_images_only(zip_path, image_output_path, csv_output_path):
    """
    Extract ZIP file and process all PDF files inside it to individual CSVs only.
    
    Parameters:
        zip_path (str): Path to the ZIP file.
        image_output_path (str): Path to save extracted images.
        csv_output_path (str): Path to save individual CSV files.
    
    Returns:
        list: List of successfully processed PDF filenames.
    """
    processed_pdfs = []
    
    try:
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find all PDF files in extracted content (including subdirectories)
            pdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            print(f"Found {len(pdf_files)} PDF files in ZIP")
            
            # Process each PDF file (extract images and create individual CSVs only)
            for pdf_path in pdf_files:
                try:
                    result = parse_pdf_to_individual_csv(pdf_path, image_output_path, csv_output_path)
                    if result:
                        processed_pdfs.append(result)
                    
                except Exception as e:
                    print(f"Error processing {os.path.basename(pdf_path)}: {e}")
                    continue
        
        # Clean up the original ZIP file
        try:
            os.remove(zip_path)
        except:
            pass
        
        return processed_pdfs
        
    except Exception as e:
        print(f"Error extracting ZIP file {zip_path}: {e}")
        return []

def process_images_to_individual_csv(image_folder_path: str, csv_output_path: str, processed_pdf_names: list = None) -> list:
    """
    Process images in the folder and convert them to individual CSV files only.
    Only processes images that match the current PDF being processed.
    
    Parameters:
        image_folder_path (str): Path to folder containing extracted images.
        csv_output_path (str): Path to folder where CSV files will be saved.
        processed_pdf_names (list): List of current PDF names being processed.
    
    Returns:
        list: List of individual CSV file paths created.
    """
    try:
        # Create CSV folder if it doesn't exist
        os.makedirs(csv_output_path, exist_ok=True)
        
        # Get all image files
        all_image_files = [f for f in os.listdir(image_folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Filter images that belong to current PDFs being processed
        image_files = []
        if processed_pdf_names:
            for image_file in all_image_files:
                for pdf_name in processed_pdf_names:
                    # Clean PDF name for matching
                    clean_pdf_name = os.path.splitext(pdf_name)[0]
                    clean_pdf_name = re.sub(r'[^\w\-_.]', '_', clean_pdf_name)
                    clean_pdf_name = re.sub(r'_+', '_', clean_pdf_name)
                    
                    if clean_pdf_name in image_file:
                        image_files.append(image_file)
                        break
        else:
            image_files = all_image_files
        
        if not image_files:
            print("No matching image files found to process.")
            return []
        
        processed_files = []
        
        # Loop through each image and process it
        for image_file in image_files:
            # Skip if CSV already exists for this image
            csv_name = os.path.splitext(image_file)[0] + '.csv'
            csv_path = os.path.join(csv_output_path, csv_name)
            
            if os.path.exists(csv_path):
                print(f"CSV already exists for {image_file}, skipping...")
                processed_files.append(csv_path)
                continue
            
            image_path = os.path.join(image_folder_path, image_file)
            image_path = os.path.abspath(image_path).replace("\\", "/")
            print(f"Processing: {image_file}")
            
            try:
                response = query_with_local_image(image_path)
                if response and "choices" in response:
                    content = response["choices"][0]["message"]['content']
                    csv_file_path = save_transcript_to_csv(content, image_file, csv_output_path)
                    if csv_file_path:
                        processed_files.append(csv_file_path)
                else:
                    print(f"Invalid response for {image_file}")
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
        
        return processed_files
            
    except Exception as e:
        print(f"Error processing images to individual CSVs: {e}")
        return []
    
def extract_images_from_pdf(input_pdf_path, output_folder_path, dpi_width=800, dpi_height=600):
    """
    Extracts images from a PDF using fitz (PyMuPDF) and saves them in the specified resolution.
    """
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        extracted_images = []
        
        # Get PDF filename without extension and clean it
        pdf_name = os.path.splitext(os.path.basename(input_pdf_path))[0]
        pdf_name = re.sub(r'[^\w\-_.]', '_', pdf_name)
        pdf_name = re.sub(r'_+', '_', pdf_name)
        pdf_name = pdf_name.replace(' ', '_')  # Replace spaces with underscores

        doc = fitz.open(input_pdf_path)

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)

            # Calculate zoom factors
            rect = page.rect
            zoom_x = dpi_width / rect.width
            zoom_y = dpi_height / rect.height

            # Render page to image with PDF name
            matrix = fitz.Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Save image with PDF name prefix
            output_path = os.path.join(output_folder_path, f"{pdf_name}_page_{page_number + 1}.png")
            pix.save(output_path)
            # Normalize to forward slashes for all downstream use
            output_path = os.path.normpath(output_path).replace("\\", "/")
            extracted_images.append(output_path)
            print(f"Saved page {page_number + 1} to {os.path.basename(output_path)}")

        doc.close()
        return extracted_images
    
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
        return []

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_string
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def query_with_local_image(image_path):
    """
    Send image to LLM API for transcript data extraction.
    
    Parameters:
        image_path (str): Path to the image file.
    
    Returns:
        dict: API response containing extracted transcript data, or None if the request fails.
    """
    try:
        image_base64 = encode_image_to_base64(image_path)
        if not image_base64:
            print(f"Failed to encode image {image_path} to base64")
            return None
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are a data extraction expert.

Carefully extract all structured data from the academic transcript image.

Output Requirements:

Produce one complete table in CSV format that includes:

General Info (repeat for each row):
College Name

Student Name

Advisor(s)

Term (e.g., Transfer Term, 2024–2025 Fall, 2024–2025 Spring)

Subterm (if any; e.g., 1st 8 weeks, 2nd 8 weeks)

Organization Name (e.g., MURRAY STATE COLLEGE)

Course-Level Fields:
Course Number

Course Title

Grade

Rpt

CR Type

Completion Date (if available)

Credit Hours Attempted (Hrs Att)

Credit Hours Earned (Hrs Ern)

Credit Hours GPA (Hrs GPA)

Quality Points (Qual Pts)

GPA

Totals:
Include all Term Totals and Career Totals as labeled rows in the table.

⚠️ Instructions:

Capture every single course — including transfer credits, regular, WIP, or ND grades.

Maintain column consistency.

Preserve values like "TR", "CR", "WIP", "ND", "NM", etc., as-is.

No information should be omitted — even if it appears in small fonts or different sections.

🔚 Output only the final structured CSV. No explanation or summary needed.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        # Check if the response status is successful
        if response.status_code != 200:
            print(f"API request failed for {image_path} with status code {response.status_code}: {response.text}")
            return None
        
        # Check if the response content is not empty
        if not response.text:
            print(f"Empty response received for {image_path}")
            return None
        
        # Attempt to parse JSON response
        try:
            return response.json()
        except ValueError as json_error:
            print(f"Error parsing JSON response for {image_path}: {json_error}")
            print(f"Response content: {response.text}")
            return None
    
    except Exception as e:
        print(f"Error querying LLM with image {image_path}: {e}")
        return None

def extract_csv_from_content(content: str) -> str:
    """
    Extracts CSV data from the content.
    Looks for lines that appear to be CSV formatted.
    """
    try:
        lines = content.split('\n')
        csv_lines = []
        
        for line in lines:
            line = line.strip()
            # Check if line looks like CSV (contains quotes and commas)
            if line and ('","' in line or line.startswith('"') and line.endswith('"')):
                csv_lines.append(line)
        
        if csv_lines:
            return '\n'.join(csv_lines)
        
        # Alternative: if the entire content is CSV-like
        if '","' in content and content.count('"') > 10:
            return content.strip()
            
        return None
        
    except Exception as e:
        print(f"Error extracting CSV from content: {e}")
        return None

def save_transcript_to_csv(content: str, image_name: str, csv_output_path: str) -> None:
    """
    Extracts student name and college name from the transcript content
    and saves the CSV data to a file named after the image.
    """
    try:
        # Extract Full Name and College Name for validation
        name_match = re.search(r"Full Name of the Student:\s*(.*)", content)
        college_match = re.search(r"College Name:\s*(.*)", content)

        if not name_match or not college_match:
            print(f"Warning: Could not extract student/college name from {os.path.basename(image_name)}")
        
        # Extract CSV data from content
        csv_data = extract_csv_from_content(content)
        
        if csv_data:
            # Create filename based on image name (remove extension and clean)
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            base_name = re.sub(r'[^\w\-_.]', '_', base_name)  # Replace special chars
            base_name = base_name.replace(' ', '_')  # Replace spaces
            csv_filename = f"{base_name}.csv"
            csv_filepath = os.path.join(csv_output_path, csv_filename)
            
            # Save CSV data to file
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                csvfile.write(csv_data)
            
            print(f"Saved CSV data to: {os.path.basename(csv_filepath)}")
            return csv_filepath
        else:
            print(f"No CSV data found in content for {os.path.basename(image_name)}")
            return None

    except Exception as e:
        print(f"Error saving transcript for {os.path.basename(image_name)}: {e}")
        return None

def merge_all_csv_files(csv_output_path: str, processed_pdf_names: list = None) -> str:
    """
    Merges all CSV files in the csv_folder into a single CSV file.
    
    Parameters:
        csv_output_path (str): Path to the CSV output folder.
        processed_pdf_names (list): List of processed PDF names.
    
    Returns:
        str: Path to the merged CSV file.
    """
    try:
        csv_files = [f for f in os.listdir(csv_output_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found to merge.")
            return None
        
        # Generate merged filename based on processed PDFs
        if processed_pdf_names and len(processed_pdf_names) > 0:
            # Clean PDF names and join them
            clean_names = []
            for pdf_name in processed_pdf_names:
                base_name = os.path.splitext(os.path.basename(pdf_name))[0]
                # Replace special characters with empty string, spaces with underscore
                clean_name = re.sub(r'[^\w\s\-_.]', '', base_name).replace(' ', '_')
                # Remove multiple underscores
                clean_name = re.sub(r'_+', '_', clean_name)
                clean_names.append(clean_name)
            merged_filename = f"merged_{'_'.join(clean_names)}.csv"
        else:
            merged_filename = 'merged_student_transcript.csv'
        
        # Remove multiple underscores from final filename
        merged_filename = re.sub(r'_+', '_', merged_filename)
        merged_filepath = os.path.join(csv_output_path, merged_filename)
        header_written = False
        total_rows = 0
        
        with open(merged_filepath, 'w', newline='', encoding='utf-8') as merged_file:
            merged_writer = csv.writer(merged_file)
            
            for csv_file in csv_files:
                # Skip any existing merged files
                if csv_file.startswith('merged_'):
                    continue
                    
                csv_filepath = os.path.join(csv_output_path, csv_file)
                print(f"Merging: {csv_file}")
                
                try:
                    with open(csv_filepath, 'r', encoding='utf-8') as individual_file:
                        csv_reader = csv.reader(individual_file)
                        rows = list(csv_reader)
                        
                        if rows:
                            if not header_written:
                                # Write header from first file
                                merged_writer.writerow(rows[0])
                                header_written = True
                                # Write all rows including data rows
                                for row in rows[1:]:
                                    if row:  # Skip empty rows
                                        merged_writer.writerow(row)
                                        total_rows += 1
                            else:
                                # Skip header row for subsequent files, write only data rows
                                for row in rows[1:]:
                                    if row:  # Skip empty rows
                                        merged_writer.writerow(row)
                                        total_rows += 1
                                        
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue
        
        print(f"Successfully merged {len([f for f in csv_files if not f.startswith('merged_')])} CSV files")
        print(f"Total data rows merged: {total_rows}")
        print(f"Final merged file: {os.path.basename(merged_filepath)}")
        
        return merged_filepath
        
    except Exception as e:
        print(f"Error merging CSV files: {e}")
        return None

def process_images_to_csv(image_folder_path: str, csv_output_path: str, processed_pdf_names: list = None) -> str:
    """
    Process all images in the folder and convert them to CSV files.
    
    Parameters:
        image_folder_path (str): Path to folder containing extracted images.
        csv_output_path (str): Path to folder where CSV files will be saved.
        processed_pdf_names (list): List of processed PDF names.
    
    Returns:
        str: Path to the final merged CSV file.
    """
    try:
        # Create CSV folder if it doesn't exist
        os.makedirs(csv_output_path, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print("No image files found to process.")
            return None
        
        processed_files = []
        
        # Loop through each image and process it
        for image_file in image_files:
            image_path = os.path.join(image_folder_path, image_file)
            image_path = os.path.abspath(image_path).replace("\\", "/")
            print(f"Processing: {image_file}")
            try:
                response = query_with_local_image(image_path)
                if response and "choices" in response:
                    content = response["choices"][0]["message"]['content']
                    csv_file_path = save_transcript_to_csv(content, image_file, csv_output_path)
                    if csv_file_path:
                        processed_files.append(csv_file_path)
                else:
                    print(f"Invalid response for {image_file}")
            except Exception as e:
                print(f"Failed to process {image_file}: {e}")
        
        # Merge all CSV files into one final file
        if processed_files:
            merged_csv_path = merge_all_csv_files(csv_output_path, processed_pdf_names)
            return merged_csv_path
        else:
            print("No CSV files were created to merge.")
            return None
            
    except Exception as e:
        print(f"Error processing images to CSV: {e}")
        return None
    
def fix_term_career_totals(csv_path, output_path):
    # Load the CSV
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    
    # Check current number of columns and add extra columns if needed
    current_cols = len(df.columns)
    required_cols = 20
    
    if current_cols < required_cols:
        # Add extra columns
        for i in range(current_cols, required_cols):
            df[f'Extra_Col_{i}'] = ''
    
    # Iterate through rows to find "Term Totals :" and "Career Totals :"
    for idx in df.index:
        # Check all columns in this row for "Term Totals :" or "Career Totals :"
        row_values = [str(df.iloc[idx, col]) for col in range(len(df.columns))]
        
        # Check if this row contains any of the totals patterns
        has_term_totals = any("Term Totals" in val for val in row_values)
        has_career_totals = any("Career Totals" in val for val in row_values)
        has_subterm_totals = any("Subterm Totals" in val for val in row_values)
        has_division_career_totals = any("Division Career Totals" in val for val in row_values)
        
        if has_term_totals:
            # print(f"Found 'Term Totals' at row {idx}")
            
            # Find all numeric values in this row (skip text columns)
            numeric_values = []
            for col_idx in range(len(df.columns)):
                val = df.iloc[idx, col_idx]
                # Check if it's a number (not text like the totals labels)
                if str(val).strip() not in ["Term Totals :", "Career Totals :", "Subterm Totals :", "Division Career Totals :", "", "nan", "NaN"]:
                    try:
                        # Try to convert to number to verify it's numeric
                        float(val)
                        numeric_values.append(val)
                    except:
                        pass
            
            # print(f"Numeric values found: {numeric_values}")
            
            # Only clear columns from the Completion Date column onwards (index 11 and beyond)
            for col_idx in range(11, len(df.columns)):
                df.iloc[idx, col_idx] = ''
            
            # Place "Term Totals :" in Completion Date column (index 11)
            if len(df.columns) > 11:
                df.iloc[idx, 11] = "Term Totals"
            
            # Place the numeric values starting from column M (index 12)
            for i, value in enumerate(numeric_values):
                col_idx = 12 + i
                if col_idx < len(df.columns):
                    df.iloc[idx, col_idx] = value
        
        elif has_subterm_totals:
            # print(f"Found 'Subterm Totals' at row {idx}")
            
            # Find all numeric values in this row (skip text columns)
            numeric_values = []
            for col_idx in range(len(df.columns)):
                val = df.iloc[idx, col_idx]
                # Check if it's a number (not text like the totals labels)
                if str(val).strip() not in ["Term Totals :", "Career Totals :", "Subterm Totals :", "Division Career Totals :", "", "nan", "NaN"]:
                    try:
                        # Try to convert to number to verify it's numeric
                        float(val)
                        numeric_values.append(val)
                    except:
                        pass
            
            # print(f"Numeric values found: {numeric_values}")
            
            # Only clear columns from the Completion Date column onwards (index 11 and beyond)
            for col_idx in range(11, len(df.columns)):
                df.iloc[idx, col_idx] = ''
            
            # Place "Subterm Totals" in Completion Date column (index 11)
            if len(df.columns) > 11:
                df.iloc[idx, 11] = "Subterm Totals"
            
            # Place the numeric values starting from column M (index 12)
            for i, value in enumerate(numeric_values):
                col_idx = 12 + i
                if col_idx < len(df.columns):
                    df.iloc[idx, col_idx] = value
        
        elif has_division_career_totals:
            # print(f"Found 'Division Career Totals' at row {idx}")
            
            # Find all numeric values in this row (skip text columns)
            numeric_values = []
            for col_idx in range(len(df.columns)):
                val = df.iloc[idx, col_idx]
                # Check if it's a number (not text like the totals labels)
                if str(val).strip() not in ["Term Totals :", "Career Totals :", "Subterm Totals :", "Division Career Totals :", "", "nan", "NaN"]:
                    try:
                        # Try to convert to number to verify it's numeric
                        float(val)
                        numeric_values.append(val)
                    except:
                        pass
            
            # print(f"Numeric values found: {numeric_values}")
            
            for col_idx in range(11, len(df.columns)):
                df.iloc[idx, col_idx] = ''
            
            # Place "Division Career Totals" in Completion Date column (index 11)
            if len(df.columns) > 11:
                df.iloc[idx, 11] = "Division Career Totals"
            
            # Place the numeric values starting from column M (index 12)
            for i, value in enumerate(numeric_values):
                col_idx = 12 + i
                if col_idx < len(df.columns):
                    df.iloc[idx, col_idx] = value
        
        elif has_career_totals:
            # print(f"Found 'Career Totals' at row {idx}")
            
            # Find all numeric values in this row (skip text columns)
            numeric_values = []
            for col_idx in range(len(df.columns)):
                val = df.iloc[idx, col_idx]
                # Check if it's a number (not text like the totals labels)
                if str(val).strip() not in ["Term Totals :", "Career Totals :", "Subterm Totals :", "Division Career Totals :", "", "nan", "NaN"]:
                    try:
                        # Try to convert to number to verify it's numeric
                        float(val)
                        numeric_values.append(val)
                    except:
                        pass
            
            # print(f"Numeric values found: {numeric_values}")
            
            # Only clear columns from the Completion Date column onwards (index 11 and beyond)
            for col_idx in range(11, len(df.columns)):
                df.iloc[idx, col_idx] = ''
            
            # Place "Career Totals :" in Completion Date column (index 11)
            if len(df.columns) > 11:
                df.iloc[idx, 11] = "Career Totals"
            
            # Place the numeric values starting from column M (index 12)
            for i, value in enumerate(numeric_values):
                col_idx = 12 + i
                if col_idx < len(df.columns):
                    df.iloc[idx, col_idx] = value
                    
    unwanted_values = ["Term Totals", "Career Totals", "Subterm Totals", "Division Career Totals"]
    # Create a boolean mask using str.startswith for multiple values
    mask = df["Course Title"].astype(str).apply(lambda x: any(x.startswith(val) for val in unwanted_values))
    # Replace matching rows with blank
    df.loc[mask, "Course Title"] = ''
    df.loc[mask, "Course Number"] = ''
    df.to_csv(output_path, index=False)

def parse_and_index_pdf(pdf_path, user, private):
    """
    Main function to parse PDF and process transcript data.
    
    Parameters:
        pdf_path (str): Path to the input PDF file.
        user (str): Username for private processing.
        private (bool): If True, save to user's private folder; if False, save to public uploads.
    
    Returns:
        str: Path to the final merged CSV file, or None if processing failed.
    """
    try:
        # Determine output folders based on private flag
        if private:
            base_output_path = f"data/user_uploads/{user}"
            image_output_path = os.path.join(base_output_path, "extracted_images")
            csv_output_path = os.path.join(base_output_path, "csv_files")
        else:
            base_output_path = "data/public_uploads"
            image_output_path = os.path.join(base_output_path, "extracted_images")
            csv_output_path = os.path.join(base_output_path, "csv_files")
        
        # Create base directories
        os.makedirs(base_output_path, exist_ok=True)
        
        # Step 1: Extract images from PDF
        print("Step 1: Extracting images from PDF...")
        extracted_images = extract_images_from_pdf(pdf_path, image_output_path)
        
        if not extracted_images:
            st.error("Failed to extract images from PDF")
            return None
        
        # Step 2: Process images to individual CSV files only (no merging yet)
        print("Step 2: Processing images to individual CSV files...")
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        individual_csvs = process_images_to_individual_csv(image_output_path, csv_output_path, [pdf_name])
        
        if not individual_csvs:
            st.error("No CSV files were created from the images")
            return None
        
        # Step 3: Create final merged CSV from all individual CSVs in the folder
        print("Step 3: Creating final merged CSV...")
        final_csv_path = create_final_merged_csv(csv_output_path)

        # Clean data
        if final_csv_path:
            fix_term_career_totals(final_csv_path, final_csv_path)
            print(f"Final processed CSV: {os.path.basename(final_csv_path)}")
        
        return final_csv_path
        
    except Exception as e:
        print(f"Error in parse_and_index_pdf: {e}")
        st.error(f"Error processing PDF: {e}")
        return None