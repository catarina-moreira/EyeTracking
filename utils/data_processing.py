import PyPDF2
import io
import re

import openai
from openai import OpenAI

import pandas as pd
import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

OPENAI_API_KEY='not_available'


def extract_text_from_pdf(pdf_file_path):

    with open(pdf_file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        extracted_text = ""

        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

    extracted_text = extracted_text.replace("\n", "")
    return extracted_text

def get_csv_columns_from_pdf(pdf_text):
    """
    Function to extract CSV columns and their explanations from a given PDF text.

    Parameters:
    - pdf_text (str): The text extracted from the PDF file.
    """

    client = OpenAI(api_key=OPENAI_API_KEY)

    results = {}

    prompt = f"""Rearrange the following text into a structured format where fields are paired with their explanations:\n\n{pdf_text}. 
            Do not include the lines that contain a trail of '....... '
            Ignore the Date,
            Ignore the line 'CSV Column',
            Ignore the numbers. Start with the field TrialID"""
    try:
        response =  client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt}]}],
            max_tokens=2000, 
            temperature=0 
        )
        rearranged_text = response.choices[0].message.content.strip()
        rearranged_lines = rearranged_text.split('\n')

        for line in rearranged_lines:

            if "CSV Columns" in line:
                continue

            if len(line) < 2:
                continue

            key = re.sub(r'^\d+\.\s*', '', line.split(":")[0].strip())
            key = key.replace("**", "")
            key = key.replace("-", "")
            key = key.replace(" ", "")
            value = line.split(":")[1].strip()
            results[key] = value

    except Exception as e:
        rearranged_text = "Error"
        rearranged_lines = "Error"
        print(f"An error occurred: {e}")
    
    return results
 

def load_image(img_filename):

    img_path = os.path.join("data/imgs", img_filename)
    img = Image.open(img_path)
    plt.imshow(img, cmap='bone')
    plt.axis('off')
    plt.grid(False)
    plt.show()

    img_np = np.array(img)

    return img_np




