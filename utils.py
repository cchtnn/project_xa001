import os
import json
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin
from collections import defaultdict
import torch

def get_data_from_website(url):
    """
    Fetches data from a website and saves it as a JSON file.
    """
    response = requests.get(url)
    if response.status_code == 500:
        print("Server error")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    tab_titles = soup.find_all("div", class_="elementor-tab-title")
    tab_data = {}

    for title_div in tab_titles:
        tab_id = title_div.get("data-tab")
        tab_title = title_div.get_text(strip=True)
        matching_content = soup.find("div", class_="elementor-tab-content", attrs={"data-tab": tab_id})
        tab_content = matching_content.get_text(separator="\n", strip=True) if matching_content else ""
        tab_data[tab_title] = tab_content

    os.makedirs("data", exist_ok=True)
    with open("data/tab_data.json", "w", encoding="utf-8") as f:
        json.dump(tab_data, f, ensure_ascii=False, indent=2)
    print("Data saved to data/tab_data.json")

def append_ferpa_data(url, output_filename="data//tab_data.json"):
    """
    Fetches data from a FERPA-related website, processes it, and appends it to a JSON file.

    Args:
        url (str): The URL of the website to scrape.
        output_filename (str, optional): The name of the JSON file to save/append data to.
            Defaults to "data//tab_data.json".
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    data = {}
    h3_tags = soup.find_all('h3')
    for h3_tag in h3_tags:
        question = h3_tag.text.strip()
        answer_parts = []
        sibling = h3_tag.find_next_sibling()
        while sibling and sibling.name in ['p', 'div', 'ul', 'ol']:
            answer_parts.append(sibling.text.strip())
            sibling = sibling.find_next_sibling()
        answer = " ".join(answer_parts).strip()
        if question and answer:
            data[question] = answer

    modified_data = {}
    for key, value in data.items():
        new_key = key.rstrip(":")
        modified_value = value.replace("Back to Top", "").strip()
        if len(modified_value.split()) >= 5:
            modified_data[new_key] = modified_value

    _append_to_json(modified_data, output_filename)

def append_civil_rights_data(url, output_filename="data//tab_data.json"):
    """
    Fetches data from a civil rights laws website, processes it, and appends it to a JSON file.

    Args:
        url (str): The URL of the website to scrape.
        output_filename (str, optional): The name of the JSON file to save/append data to.
            Defaults to "data//tab_data.json".
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    final_data = {}

    # Extract the main heading and description
    main_heading = soup.find('h1', class_='usa-hero__heading')
    main_desc = soup.find('div', class_='field--name-body')

    if main_heading and main_desc:
        title = main_heading.text.strip()
        description = main_desc.text.strip()
        final_data[title] = description

    # Now extract the cards information
    cards = soup.find_all('div', class_='card-image-top-txt')

    for card in cards:
        card_title_tag = card.find('div', class_='field--name-field-ed-card-image-top-title')
        card_summary_tag = card.find('div', class_='field--name-field-ed-card-image-top-summary')
        card_link_tag = card.find('div', class_='field--name-field-ed-card-image-top-link')

        if card_title_tag and card_summary_tag:
            card_title = card_title_tag.text.strip()
            card_summary = card_summary_tag.text.strip()

            # Get the link if available
            link = ""
            if card_link_tag and card_link_tag.find('a'):
                href = card_link_tag.find('a')['href']
                if href.startswith("/"):
                    href = "https://www.ed.gov" + href
                link = href
            final_data[card_title] = f"{card_summary} link :- {link}".strip()

    _append_to_json(final_data, output_filename)

def append_file_complaint_data(url, output_filename="data//tab_data.json"):
    """
    Fetches data from the file a complaint website, processes it, and appends it to a JSON file.

    Args:
        url (str): The URL of the website to scrape.
        output_filename (str, optional): The name of the JSON file to save/append data to.
            Defaults to "data//tab_data.json".
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unnecessary tags
    for tag in soup(["script", "style", "footer", "nav", "header", "aside"]):
        tag.decompose()

    # Remove known banners or sections
    for div in soup.find_all(['div', 'section'], class_=[
        'usa-banner', 'header', 'navigation', 'menu', 'site-header',
        'usa-footer', 'main-header', 'branding', 'footer-links'
    ]):
        div.decompose()

    for elem in soup.find_all(id=[
        'header', 'footer', 'navbar', 'skip-link', 'back-to-top'
    ]):
        elem.decompose()

    # Get heading
    heading = soup.find('h1')
    key = heading.get_text(strip=True) if heading else "No Heading Found"

    # Collect all visible text
    body_text = soup.get_text(separator='\n')
    lines = [line.strip() for line in body_text.splitlines() if line.strip()]

    # Keywords/phrases to exclude
    unwanted_keywords = [
        "Complaint Forms", "Electronic Complaint Form Learn how to file", "How OCR Evaluates Complaints",
        "FAQs on the Complaint Process", "Customer Service Standards for the Case Resolution Process",
        "Complainant and Interviewee Rights and Protections", "Rights and protections",
        "Office of Communications and Outreach", "Page Last Reviewed"
    ]

    # Remove lines matching unwanted sections
    filtered_lines = [
        line for line in lines
        if not any(keyword.lower() in line.lower() for keyword in unwanted_keywords)
    ]

    # Try to add Electronic Complaint Form and Fillable PDF Complaint Form links
    extra_links_text = ""
    electronic_form = soup.find('a', string=lambda text: text and 'Electronic Complaint Form' in text)
    pdf_form = soup.find('a', string=lambda text: text and 'Fillable PDF Complaint Form' in text)

    if electronic_form:
        href = electronic_form.get('href')
        extra_links_text += f"\nElectronic Complaint Form: {href}"
    if pdf_form:
        href = pdf_form.get('href')
        extra_links_text += f"\nFillable PDF Complaint Form: {href}"

    # Final value
    value = ' '.join(filtered_lines) + extra_links_text

    # Result dict
    result = {key: value}

    _append_to_json(result, output_filename)

def extract_table_as_text(table):
    rows = []
    for row in table.find_all('tr'):
        cols = [col.get_text(strip=True) for col in row.find_all(['th', 'td'])]
        rows.append('\t'.join(cols))
    return '\n'.join(rows)

def extract_text_with_links(element, base_url):
    result = ""
    for child in element.children:
        if child.name == 'a':
            link_text = child.get_text(strip=True)
            link_url = child.get('href', '')
            if link_url and not link_url.startswith(('http://', 'https://')):
                link_url = urljoin(base_url, link_url)
            result += f"{link_text} [{link_url}]" if link_url else link_text
        elif isinstance(child, str):
            result += child
        elif child.name:
            result += extract_text_with_links(child, base_url)
    return result.strip()

def extract_list_content(heading_element, base_url):
    content = []
    current = heading_element.next_sibling

    while current:
        if isinstance(current, NavigableString):
            current = current.next_sibling
            continue
        if current.name == 'ul':
            for li in current.find_all('li', recursive=True):
                item_content = extract_text_with_links(li, base_url)
                content.append(item_content)
            break
        elif current.name in ['h2', 'h3']:
            break
        current = current.next_sibling

    return content

def extract_h3_with_paragraphs(soup):
    result = {}
    panels = soup.find_all("div", class_="panel panel-primary")
    for panel in panels:
        heading = panel.find("div", class_="panel-heading")
        body = panel.find("div", class_="panel-body")
        if heading and body:
            h3 = heading.find("h3")
            p = body.find("p")
            if h3 and p:
                heading_text = h3.get_text(strip=True)
                paragraph_text = p.get_text(strip=True)
                result[heading_text] = paragraph_text
    return result

def append_fafsa_data(url, output_filename="data/tab_data.json"):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    base_url = url
    soup = BeautifulSoup(response.content, 'html.parser')
    container = soup.find('div', class_='field field--name-body field--type-text-with-summary field--label-hidden field__item')
    if not container:
        print("No content container found.")
        return

    result = defaultdict(str)
    current_header = None

    for tag in container.find_all(recursive=False):
        if tag.name and tag.name.startswith('h'):
            current_header = tag.get_text(strip=True)
            result[current_header] = ''

        elif tag.name == 'p' and current_header:
            paragraph_text = tag.get_text(strip=True)
            if paragraph_text:
                result[current_header] += paragraph_text + '\n'

        elif tag.name == 'table' and current_header:
            table_text = extract_table_as_text(tag)
            if table_text:
                result[current_header] += '\n' + table_text + '\n'

        elif tag.name == 'div' and current_header:
            nested_table = tag.find('table')
            if nested_table:
                table_text = extract_table_as_text(nested_table)
                if table_text:
                    result[current_header] += '\n' + table_text + '\n'

    result = {k: v.strip() for k, v in result.items() if len(v.strip()) > 1}

    headers = container.find_all(['h2', 'h3'])
    for header in headers:
        header_text = header.get_text(strip=True)
        list_items = extract_list_content(header, base_url)
        if list_items:
            if header_text in result:
                result[header_text] += '\n' + '\n'.join(list_items)
            else:
                result[header_text] = '\n'.join(list_items)

    panel_data = extract_h3_with_paragraphs(soup)
    for heading, paragraph in panel_data.items():
        if heading in result:
            result[heading] += '\n' + paragraph
        else:
            result[heading] = paragraph

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    _append_to_json(result, output_filename)
    print(f"Data appended to {output_filename}")
    return result

def _append_to_json(new_data, output_filename):
    """
    Appends a dictionary of data to an existing JSON file or creates a new one.

    Args:
        new_data (dict): The dictionary data to append.
        output_filename (str): The name of the JSON file.
    """
    try:
        with open(output_filename, 'r+', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                existing_data.update(new_data)
                f.seek(0)
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
                f.truncate() # Remove remaining part if new data is shorter
            except json.JSONDecodeError:
                print("Error decoding existing JSON file. Overwriting with new data.")
                f.seek(0)
                json.dump(new_data, f, ensure_ascii=False, indent=4)
                f.truncate()
    except FileNotFoundError:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)


def generate_embeddings(documents):
    """
    Generates embeddings for a list of documents using a SentenceTransformer model.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


def create_faiss_index(embeddings):
    """
    Creates and saves a FAISS index for the given embeddings.
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance metric
    index.add(embeddings)
    faiss.write_index(index, 'data/faiss_index.idx')

    return index


def load_faiss_index():
    """
    Loads the saved FAISS index.
    """
    index = faiss.read_index('data/faiss_index.idx')
    return index


def load_metadata():
    """
    Loads the metadata (e.g., tab titles) used for reverse lookup in FAISS index.
    """
    with open('data/faiss_metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata

def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    return model
