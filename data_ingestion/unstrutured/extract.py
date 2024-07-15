from unstructured.partition.pdf import partition_pdf

def read_pdf(pdf_file_name,image_output_path="images"):
    raw_pdf_elements = partition_pdf(
        filename=pdf_file_name,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_output_path,
    )

    # Categorize extracted elements from a PDF into tables and texts.
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    
    return texts,tables


if __name__ == "__main__":
    PATH = r"/Users/ashique/Downloads/download.pdf"
    _,tables = read_pdf(PATH)
    print(len(tables),tables[1])