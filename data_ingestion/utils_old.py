import fitz  # PyMuPDF
import io
import os
import base64
from langchain_core.messages import AIMessage, HumanMessage

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def image_summarize(model, img_base64, prompt):
    """Make image summary"""
    msg = model(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(path):
    # Store base64 encoded images
    img_base64_list = []

    # Store image summaries
    image_summaries = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """

    # Apply to images
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".png"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return img_base64_list, image_summaries


def pdf_to_image_data_urls(pdf_path):
    try:
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        data_urls = []

        # Iterate through each page
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            
            # Write the image to a bytes buffer
            bytes_buffer = io.BytesIO(pix.tobytes())
            page_bytes = bytes_buffer.getvalue()
            base64_page = base64.b64encode(page_bytes).decode('utf-8')
            
            # Create a Data URL for the page
            data_url = f"data:image/png;base64,{base64_page}"
            data_urls.append(data_url)
        
        return data_urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    pdf_path = '/Users/ashique/Downloads/download.pdf'
    pages_bytes = pdf_to_image_data_urls(pdf_path)

    for i, page_bytes in enumerate(pages_bytes):
        print(f"Page {i + 1} as bytes: {page_bytes[:100]}...")  # Print first 100 bytes of each page as a sample
