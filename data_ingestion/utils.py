from typing import Optional,Tuple,Dict,Union,List
import pandas as pd
import glob
import time
import fitz 
import numpy as np
import os
from openai import OpenAI
import pathlib
from PIL import Image as PIL_Image  
import base64

from data_ingestion.gpt_4_o.config import OPENAI_KEY


MODEL_NAME = "gpt-4o"
CLIENT =  OpenAI(
    api_key=OPENAI_KEY
)

class Image:
    """The image that can be sent to a generative model."""

    _image_bytes: bytes
    _loaded_image: Optional["PIL_Image.Image"] = None

    @staticmethod
    def load_from_file(location: str) -> "Image":
        """Loads image from file.

        Args:
            location: Local path from where to load the image.

        Returns:
            Loaded image as an `Image` object.
        """
        image_bytes = pathlib.Path(location).read_bytes()
        image = Image()
        image._image_bytes = image_bytes
        return image

    @staticmethod
    def from_bytes(data: bytes) -> "Image":
        """Loads image from image bytes.

        Args:
            data: Image bytes.

        Returns:
            Loaded image as an `Image` object.
        """
        image = Image()
        image._image_bytes = data
        return image

    @property
    def _pil_image(self) -> "PIL_Image.Image":
        if self._loaded_image is None:
            if not PIL_Image:
                raise RuntimeError(
                    "The PIL module is not available. Please install the Pillow package."
                )
            self._loaded_image = PIL_Image.open(io.BytesIO(self._image_bytes))
        return self._loaded_image

    @property
    def _mime_type(self) -> str:
        """Returns the MIME type of the image."""
        _FORMAT_TO_MIME_TYPE = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
        }
        if PIL_Image:
            return _FORMAT_TO_MIME_TYPE[self._pil_image.format.lower()]
        else:
            # Fall back to jpeg
            return "image/jpeg"

    @property
    def data(self) -> bytes:
        """Returns the image data."""
        return self._image_bytes

    def _repr_png_(self):
        return self._pil_image._repr_png_()



class TextEmbeddingModel:
    def __init__(self,
                 model:str = "text-embedding-3-small",
                 token:str = ""
                ) -> None:
        self.model = model
        self.openai_token = token
        self.client = OpenAI(api_key=self.openai_token)
    
    def get_embeddings(self,text:str) -> list:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(
            input = [text], 
            model=self.model
            ).data[0].embedding
    

def get_image_metadata_df(
    filename: str, image_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and an image metadata dictionary as input,
    iterates over the image metadata dictionary and extracts the image path,
    image description, and image embeddings for each image, creates a Pandas
    DataFrame with the extracted data, and returns it.

    Args:
        filename: The filename of the document.
        image_metadata: A dictionary containing the image metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted image path, image description, and image embeddings for each image.
    """

    final_data_image: List[Dict] = []
    for key, values in image_metadata.items():
        for _, image_values in values.items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["img_num"] = int(image_values["img_num"])
            data["img_path"] = image_values["img_path"]
            data["img_desc"] = image_values["img_desc"]
    
            final_data_image.append(data)

    return_df = pd.DataFrame(final_data_image).dropna()
    return_df = return_df.reset_index(drop=True)
    return return_df


def get_text_metadata_df(
    filename: str, text_metadata: Dict[Union[int, str], Dict]
) -> pd.DataFrame:
    """
    This function takes a filename and a text metadata dictionary as input,
    iterates over the text metadata dictionary and extracts the text, chunk text,
    and chunk embeddings for each page, creates a Pandas DataFrame with the
    extracted data, and returns it.

    Args:
        filename: The filename of the document.
        text_metadata: A dictionary containing the text metadata for each page.

    Returns:
        A Pandas DataFrame with the extracted text, chunk text, and chunk embeddings for each page.
    """

    final_data_text: List[Dict] = []

    for key, values in text_metadata.items():
        for chunk_number, chunk_text in values["chunked_text_dict"].items():
            data: Dict = {}
            data["file_name"] = filename
            data["page_num"] = int(key) + 1
            data["text"] = values["text"]
            data["text_embedding_page"] = values["page_text_embeddings"][
                "text_embedding"
            ]
            data["chunk_number"] = chunk_number
            data["chunk_text"] = chunk_text
            data["text_embedding_chunk"] = values["chunk_embeddings_dict"][chunk_number]

            final_data_text.append(data)

    return_df = pd.DataFrame(final_data_text)
    return_df = return_df.reset_index(drop=True)
    return return_df


text_embedding_model = TextEmbeddingModel()

def get_image_for_multi_modal(
    doc: fitz.Document,
    image: tuple,
    image_no: int,
    image_save_dir: str,
    file_name: str,
    page_num: int,
) -> Tuple[Image, str]:
    """
    Extracts an image from a PDF document, converts it to JPEG format, saves it to a specified directory,
    and loads it as a PIL Image Object.

    Parameters:
    - doc (fitz.Document): The PDF document from which the image is extracted.
    - image (tuple): A tuple containing image information.
    - image_no (int): The image number for naming purposes.
    - image_save_dir (str): The directory where the image will be saved.
    - file_name (str): The base name for the image file.
    - page_num (int): The page number from which the image is extracted.

    Returns:
    - Tuple[Image.Image, str]: A tuple containing the Gemini Image object and the image filename.
    """

    # Extract the image from the document
    xref = image[0]
    pix = fitz.Pixmap(doc, xref)

    # Convert the image to JPEG format
    pix.tobytes("jpeg")
    image_name = f"{image_save_dir}/{file_name}_image_{page_num}_{image_no}_{xref}.jpeg"
    os.makedirs(image_save_dir, exist_ok=True)

    # Save the image to the specified location
    pix.save(image_name)
    image_for_multimodal = Image.load_from_file(image_name)

    return image_for_multimodal, image_name


def get_pdf_doc_object(pdf_path: str) -> tuple[fitz.Document, int]:
    doc: fitz.Document = fitz.open(pdf_path)

    # Get the number of pages in the PDF file
    num_pages: int = len(doc)

    return doc, num_pages



def get_text_overlapping_chunk(
    text: str, character_limit: int = 1000, overlap: int = 100
) -> dict:
    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Initialize variables
    chunk_number = 1
    chunked_text_dict = {}

    # Iterate over text with the given limit and overlap
    for i in range(0, len(text), character_limit - overlap):
        end_index = min(i + character_limit, len(text))
        chunk = text[i:end_index]

        # Encode and decode for consistent encoding
        chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
            "utf-8", "ignore"
        )

        # Increment chunk number
        chunk_number += 1

    return chunked_text_dict

def get_text_embedding_from_text_embedding_model(
    text: str,
    return_array: Optional[bool] = False,
) -> list:

    if not return_array:
        embeddings = text_embedding_model.get_embeddings(text)
        text_embedding = embeddings
    else:
        text_embedding = [text_embedding_model.get_embeddings(tx) for tx in text]

    # returns 768 dimensional array
    return text_embedding


def get_page_text_embedding(text_data: Union[dict, str]) -> dict:
    embeddings_dict = {}

    if not text_data:
        return embeddings_dict

    if isinstance(text_data, dict):
        for chunk_number, chunk_value in text_data.items():
            text_embed = get_text_embedding_from_text_embedding_model(text=chunk_value)
            embeddings_dict[chunk_number] = text_embed
    else:
        # Process the first 1000 characters of the page text
        text_embed = get_text_embedding_from_text_embedding_model(text=text_data)
        embeddings_dict["text_embedding"] = text_embed

    return embeddings_dict


def get_chunk_text_metadata(
    page: fitz.Page,
    character_limit: int = 1000,
    overlap: int = 100,
    embedding_size: int = 128,
) -> tuple[str, dict, dict, dict]:
    if overlap > character_limit:
        raise ValueError("Overlap cannot be larger than character limit.")

    # Extract text from the page
    text: str = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")
    page_text_embeddings_dict: dict = get_page_text_embedding(text)
    chunked_text_dict: dict = get_text_overlapping_chunk(text, character_limit, overlap)
    # print(chunked_text_dict)

    # Get embeddings for the chunks
    chunk_embeddings_dict: dict = get_page_text_embedding(chunked_text_dict)
    return text, page_text_embeddings_dict, chunked_text_dict, chunk_embeddings_dict


def image_summarize(img_base64, prompt):
    """Make image summary"""
    # Generate description using the GPT-4 Vision Preview API
    response = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url":f"data:image/png;base64,{img_base64}"}},
                ],
            }
        ],
        # max_tokens=150,
    )

    # Extract the generated description from the response
    description = response.choices[0].message.content
    return description

def generate_img_summaries(base64_image):

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """
    image_summary = image_summarize(base64_image, prompt)

    return base64_image, image_summary

def encode_image(image_path):
    """Getting the base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_document_metadata(
    generative_multimodal_model,
    pdf_folder_path: str,
    image_save_dir: str,
    image_description_prompt: str,
    embedding_size: int = 128,
    generation_config: Optional[dict] = {},
    safety_settings: Optional[dict] = {},
    add_sleep_after_page: bool = False,
    sleep_time_after_page: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    text_metadata_df_final, image_metadata_df_final = pd.DataFrame(), pd.DataFrame()

    for pdf_path in glob.glob(pdf_folder_path + "/*.pdf"):
        print(
            "\n\n",
            "Processing the file: ---------------------------------",
            pdf_path,
            "\n\n",
        )

        doc, num_pages = get_pdf_doc_object(pdf_path)

        file_name = pdf_path.split("/")[-1]

        text_metadata: Dict[Union[int, str], Dict] = {}
        image_metadata: Dict[Union[int, str], Dict] = {}

        for page_num in range(num_pages):
            print(f"Processing page: {page_num + 1}")

            page = doc[page_num]

            text = page.get_text()
            (
                text,
                page_text_embeddings_dict,
                chunked_text_dict,
                chunk_embeddings_dict,
            ) = get_chunk_text_metadata(page, embedding_size=embedding_size)

            text_metadata[page_num] = {
                "text": text,
                "page_text_embeddings": page_text_embeddings_dict,
                "chunked_text_dict": chunked_text_dict,
                "chunk_embeddings_dict": chunk_embeddings_dict,
            }

            images = page.get_images()
            image_metadata[page_num] = {}

            for image_no, image in enumerate(images):
                image_number = int(image_no + 1)
                image_metadata[page_num][image_number] = {}

                image_for_gemini, image_name = get_image_for_multi_modal(
                    doc, image, image_no, image_save_dir, file_name, page_num
                )

                print(
                    f"Extracting image from page: {page_num + 1}, saved as: {image_name}"
                )
                base64_img = encode_image(image_name)
                _,image_description = generate_img_summaries(base64_image=base64_img)

                image_metadata[page_num][image_number] = {
                    "img_num": image_number,
                    "img_path": image_name,
                    "image_base64":base64_img,
                    "img_desc": image_description,
                }

            # Add sleep to reduce issues with Quota error on API
            if add_sleep_after_page:
                time.sleep(sleep_time_after_page)
                print(
                    "Sleeping for ",
                    sleep_time_after_page,
                    """ sec before processing the next page to avoid quota issues. You can disable it: "add_sleep_after_page = False"  """,
                )

        text_metadata_df = get_text_metadata_df(file_name, text_metadata)
        image_metadata_df = get_image_metadata_df(file_name, image_metadata)

        text_metadata_df_final = pd.concat(
            [text_metadata_df_final, text_metadata_df], axis=0
        )
        image_metadata_df_final = pd.concat(
            [
                image_metadata_df_final,
                image_metadata_df.drop_duplicates(subset=["img_desc"]),
            ],
            axis=0,
        )

        text_metadata_df_final = text_metadata_df_final.reset_index(drop=True)
        image_metadata_df_final = image_metadata_df_final.reset_index(drop=True)

    return text_metadata_df_final, image_metadata_df_final


if __name__ == "__main__":
    # Specify the image description prompt. Change it
    image_description_prompt = """Explain what is going on in the image.
    If it's a table, extract all elements of the table.
    If it's a graph, explain the findings in the graph.
    Do not include any numbers that are not mentioned in the image.
    """

    a,b = get_document_metadata(generative_multimodal_model="",
                              pdf_folder_path="/Users/ashique/Downloads/multimodal_rag",
                              image_save_dir="image/",
                              image_description_prompt="",
                              )
    a.to_csv("text_metadata_df_final.csv")
    b.to_csv("image_metadata_df_final.csv")
    # print(text_embedding_model.get_embeddings(image_description_prompt))