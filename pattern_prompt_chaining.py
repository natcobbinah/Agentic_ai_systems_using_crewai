from typing import Optional, List
from pydantic import BaseModel, Field
from openai import OpenAI
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=("{{OPENAPI_API_KEY}}")
)
model="gpt-4.1",

class DocumentOutline(BaseModel):
    """
    First LLM call: Generate a structured outline for a document.
    """
    topic: str = Field(description="The main topic of the document")
    sections:  List[str] = Field(
        description="A list of section titles for the document outline"
    )

class OutlineValidation(BaseModel):
    """
    Second LLM call: Validate the generated outline against quality criteria.
    """
    is_valid: bool = Field(
        description="Whether the outline is logical, comprehensive, and well-structured"
    )
    reasoning: str = Field(
        description="A brief explanation for why the outline is or is not valid"
    )
    confidence_score: float = Field(
        description="Confidence score between 0  and 1 on the validity of the outline"
    )

class FinalDocument(BaseModel):
    """
    Third LLM call: Generate the full document content from the outline
    """
    title: str = Field(description="A suitable title for the final document")
    full_content: str = Field(
        description="The complete, well-written content of the document, based on the provided outline"
    )

# first agent: generating the outline
def generate_document_outline(topic:str) -> DocumentOutline:
    """
    First LLM call: generate a structured outline from a topic
    
    :param topic: Description
    :type topic: str
    :return: Description
    :rtype: DocumentOutline
    """
    logger.info(f"Starting outline generation for topic: {topic}")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    You're an expert content strategist.
                    Create a logical and comprehensive outline for a document on the given topic.
                    The outline should include an introduction, several body sections, and a conclusion.
                    """
                )
            }, 
            {
                "role":"user",
                "content": topic,
            }
        ],
        response_format=DocumentOutline
    )

    result = completion.choices[0].message.parsed
    logger.info("Outline generated successfully")
    return result

# second agent: validate document outline
def validate_document_outline(outline: DocumentOutline) -> OutlineValidation:
    """
    Second LLM Call to validate the quality of the generated outline
    
    :param outline: Description
    :type outline: DocumentOutline
    :return: Description
    :rtype: OutlineValidation
    """
    logger.info("Starting outline validation")

    completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "system",
                "content": (
                    """ 
                    You are a critical quality assurance editor. Your primary goal is to REJECT
                    low-quality or vague outlines. An outline is considered invalid if the original
                    topic is too vague, ambiguous or lacks a clear focus (eg. 'stuff', 'things', 'an article')
                    Be stric. If the topic is bad, the outline is bad.
                    Provide a brief reason for your decision
                    """
                )
            },
            {
                "role": "user",
                "content": str(outline.model_dump())
            }
        ],
        response_format=OutlineValidation
    )
    result = completion.choices[0].message.parsed 
    logger.info(
        f"Validation complete - Is valid: {result.is_valid}, confidence: {result.confidence_score:.2f}"
    )

    # log the reasoning, especially for failures
    if not result.is_valid:
        logger.warning(f"Validation failed. Reasoning: {result.reasoning}")
    return result

# third agent. generate the final document
def generate_final_document(outline: DocumentOutline) -> FinalDocument:
    """
    Third LLM call: expand the validated outline into a full document
    
    :param outline: Description
    :type outline: DocumentOutline
    :return: Description
    :rtype: FinalDocument
    """
    logger.info("Generating final document from outline")

    completion = client.beta.chat.completions.parse(
        model=model, 
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    You are a skilled author.
                    Write a comprehensive, well-structured document based on the provided outline.
                    Include an engaging title, clear section headings, and a concise conclusion
                    """
                )
            },
            {
                "role":"user",
                "content": str(outline.model_dump())
            }
        ],
        response_format=FinalDocument,
    )

    result = completion.choices[0].message.parsed 
    logger.info(f"Final document generated with title: '{result.title}'")
    return result

# orchestrate the entire prompt chain
def create_document_from_topic(topic:str) -> Optional[FinalDocument]:
    """
    Main function implementing the prompt chain with a vliadation gate
    
    :param topic: Description
    :type topic: str
    :return: Description
    :rtype: FinalDocument | None
    """
    logger.info(f"Starting document creation process for topic: '{topic}'")

    # first llm call: generate the outline
    document_outline = generate_document_outline(topic)

    # second llm call: validate the outline
    validation_result = validate_document_outline(document_outline)

    # gate check: verify if the outline is valid with sufficient confidence
    if not validation_result.is_valid or validation_result.confidence_score < 0.8:
        logger.warning(
            f"Gate check failed - outline not valid or confidence too low ({validation_result.confidence_score:.2f})"
        )
        logger.warning(f"Reasoning: {validation_result.reasoning}")
    
    logger.info("Gate check passed, proceeding with final document generation")


    # third llm calll: Generate the full document 
    final_document = generate_final_document(document_outline)

    logger.info("Document creation process completed successfully")

    return final_document


topic_input = "The benefits of remote work for small businesses"

final_document_result = create_document_from_topic(topic_input)

if final_document_result:
    print(f"\nTitle: {final_document_result.title}")
    print("\n----Document content-----")

    # printing only the first 500 characters for brevity
    print(final_document_result.full_content[:500] + "...")
else:
    print("Failed to generate a valid document for the topic")