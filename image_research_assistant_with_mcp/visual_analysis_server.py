import os 
import base64
import mimetypes 
from pathlib import Path 
from google import genai 
from google.genai import types
from mcp.server.fastmcp import FastMCP 

# initialize the fastmcp server
mcp = FastMCP("VisualAnalysisServer")

@mcp.tool()
def load_image_from_path(file_path: str) -> dict:
    """
    Loads an image from a server-accessible file path, encodes it to Base64
    and determines its MIME type

    Args:
        file_path: The absolute path to the image file, which must be accessible
                   by the server running the tool 
    
    Returns:
        A dictionary containing the 'base64_image_string' and 'mime_type', or an 
        'error' key if loading fails
    """
    try: 
        image_path = Path(file_path)
        if not image_path.is_file():
            return {
                "error": f"File not found at path: {file_path}"
            }
        
        # open the file in binary read mode 
        with open(image_path, 'rb') as f:
            image_path = f.read() 

        base64_string = base64.b64decode(image_path).decode('utf-8')

        mime_type, _ = mimetypes.guess_type(image_path)

        if not mime_type:
            mime_type = "application/octet-stream"
        
        return {
            "base64_image_string": base64_string,
            "mime_type": mime_type
        }
    
    except FileNotFoundError:
        return {
            "error": f"File not found at path: {file_path}"
        }
    except Exception as e:
        return {
            "error": f"An unexpected error occured while loading the image {str(e)}"
        }
        
@mcp.tool()
def get_image_description(base64_image_string: str, mime_type: str) -> str: 
    """
    Performs a deep analysis of a Base64 encoded image and returns a detailed, 
    descriptive paragraph about its content. If the image is of a known landmark,
    it will be specifically identified. This description is intended to be used as 
    a high-quality search query for  a research tool.

    Args: 
        base64_image_string: The image file encoded as a Base64 string 
        mime_type: The MIME type of the image (e.g. 'image/jpeg', 'image/png')

    Returns:
        A single string containing a detailed description of the image. 
        Returns an error message if analysis fails
    """
    try: 
        image_bytes = base64.b64encode(base64_image_string)

        image_part = types.Part.from_bytes(
            mime_type=mime_type, 
            data=image_bytes
        )

        prompt_text = (
            """
            Analyze this image in detail. Provide a concise, one-paragraph description.
            If it is a famous landmark, work of art, or specific location, identify it by name 
            Focus on the most important and defining elements in the image that would be useful for a websearch
            Forexample. instead of a 'a building' say, 'the eiffel tower in paris'
            Do not add any conversation filler, return only the description
            """ 
        )
        model = genai.Client(
            'gemini-2.5-flash', 
            google_api_key="{{GOOGLE_GEMINI_API_KEY}}"
        )

        response = model.generate_content([image_part, prompt_text])

        description = response.text.strip()

        return description 
    
    except Exception as e: 
        return f"Error analyzing image: {e}"


        