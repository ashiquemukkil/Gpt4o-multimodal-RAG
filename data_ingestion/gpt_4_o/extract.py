from config import (
    CLIENT as client,
    AZURE_DEPLOYMENT_NAME,
    PROMPT_PDF_PARSER,
    MODEL_NAME
)


def parse_pdf_single_page(single_page_data_url):

    # Generate description using the GPT-4 Vision Preview API
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT_PDF_PARSER},
                    {"type": "image_url", "image_url": {"url": single_page_data_url}},
                ],
            }
        ],
        # max_tokens=150,
    )

    # Extract the generated description from the response
    description = response.choices[0].message.content
    return description

if __name__ == "__main__":
    from data_ingestion.utils_old import pdf_to_image_data_urls

    pdf_path = '/Users/ashique/Downloads/download.pdf'
    pages_bytes = pdf_to_image_data_urls(pdf_path)

    print(parse_pdf_single_page(pages_bytes[1]))