from pydantic import Field, BaseModel, List
from typing import Dict
from openai import OpenAI
import logging

client = OpenAI(
    api_key=("{{OPENAPI_API_KEY}}")
)
model = "gpt-4.1"

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class TranslationResponse(BaseModel):
    """Response from the translator LLM"""
    translated_text: str = Field(description="The translated text")
    reasoning: str = Field(description="Brief explanation of translation choices made")

class EvaluationCriteria(BaseModel):
    """Evaluation criteria for translation quality"""
    accuracy: int = Field(description="Accuracy of translation (1-10) scale", ge=1, le=10)
    fluency: int = Field(description="Natural flow in target language (1-10) scale", ge=1, le=10)
    cultural_appropriateness: int = Field(description="Cultural context preservation (1-10) scale", ge=1, le=10)
    style_preservation: int = Field(description="Preservation of original style/tone (1-10) scale", ge=1, le=10)

class TranslationEvaluation(BaseModel):
    """Evaluation of a translation attempt"""
    overall_score: float = Field(description="Overall quality score (1-10 scale)", ge=1, le=10)
    criteria_scores: EvaluationCriteria = Field(description="Detailed scoring breakdown")
    specific_feedback: List[str] = Field(description="Specific areas for improvement")
    is_satisfactory: bool = Field(description="Whether translation meets quality threshold")
    confidence: float = Field(description="Evaluator's confidence in assessment (0-1)", ge=0, le=1)


class OptimizedTranslation(BaseModel):
    """Improved translation based on evaluation feedback"""
    improved_text: str = Field(description="The improved translation")
    changes_made: List[str] = Field(description="Specific improvements implemented")
    reasoning: str = Field(description="Explanation of how feedback was addressed")

def generate_initial_translation(source_text: str, target_language: str, source_language:str="English") -> TranslationResponse:
    """Generate initial translation of the source text"""
    logger.info(f"Generating initial translation from {source_language} to {target_language}")

    completion = client.beta.chat.completions.parse(
        model=model, 
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert translator specializing in {source_language} to {target_language} translation.
                Your goal is to create accurate, fluent translations that preserve the original meaning, style, and cultural context.
                Pay attention to:
                - Accuracy of meaning
                - Natural flow in the target language
                - Cultural appropriateness
                - Preservation of original tone and style

                Provide your translation along with brief reasoning for your choices.
                """
            },
            {
                "role": "user",
                "content": f"Please translate the following {source_language} text to {target_language}:\n\n{source_text}"
            }
        ],
        response_format=TranslationResponse
    )
    result = completion.choices[0].message.parsed 
    logger.info("Initial translation generated successfully")
    return result 


def evaluate_translation(source_text: str, translated_text: str, target_language: str, source_language: str="English") -> TranslationEvaluation:
    """Evaluate the quality of the translation"""
    logger.info("Evaluating translation quality")

    completion = client.beta.chat.completions.parse(
        model=model, 
        messages=[
            {
                "role": "system",
                "content": f"""
                You are a critical translation evaluator with expertise in both {source_language} and {target_language}
                Evaluate translations based on:
                1. Accuracy: How well does it convey the original meaning?
                2. Fluency: How natural does it sound in {target_language}
                3. Cultural appropriateness: Are cultural nuances properly handled?
                4. Style preservation: Is the original tone/style maintained?

                Be thorough and constructive in your feedback. A translation is satisfactory only if scores 7+ overall.
                Provide specific, actionable feedback for improvements.
                """
            },
            {
                "role": "user",
                "content":f"""
                Please evaluate this translation:

                Original ({source_language}) : {source_text}

                Translation ({target_language}): {translated_text}

                Provide detailed scoring and specific feedback for improvement
                """
            }
        ],
        response_format=TranslationEvaluation
    )
    result = completion.choices[0].message.parsed 
    logger.info(f"Translation evaluated - Overall score: {result.overall_socre:.1f}, Satisfactory: {result.is_satisfactory}")

    if not result.is_satisfactory:
        logger.info(f"Key feedback ares {','.join(result.specific_feedback[:3])}")
    return result 


def optimize_translation(source_text: str, current_translation: str, evaluation: TranslationEvaluation,
                         target_language: str, source_language: str="English") -> OptimizedTranslation:
    """Improve the translation based on evaluation feedback"""
    logger.info("Optimizing translation based on feedback")

    feedback_summary = "\n".join([f"- {feedback}" for feedback in evaluation.specific_feedback])

    completion = client.beta.chat.completions.parse(
        model=model, 
        messages=[
            {
                "role": "system",
                "content": f"""
                You are an expert translator tasked with improving a translation based on evaluation feedback.
                Address the specific issues raised while maintaining the strengths of the current translation.
                Focus on making targeted improvements rather completely rewriting
                """
            },
            {
                "role": "user",
                "content": f"""
                Please improve this translation

                Original ({source_language}): {source_text}

                Current Translation: ({target_language}) : {current_translation}

                Evaluation Feedback: 
                Overall score: {evaluation.overall_score}/10
                Specific issues: {feedback_summary}

                Please provide an improved version that addresses these specific concerns
                """
            }
        ],
        response_format=OptimizedTranslation
    )
    result = completion.choices[0].message.parsed
    logger.info("Translation optimization completed")
    return result

def evaluator_optimizer_translation(source_text: str, target_language: str, source_language: str="English",
    max_iterations: int=3, quality_threshold: float = 7) -> Dict: 
    
    """
    Main fuction implementing the evaluator-optimizer pattern for translation

    Args:
        source_text: Text to translate
        target_language: Target language for translation
        source_language: Source language (default: English)
        max_iterations: Maximum number of optimization iterations 
        quality_threshold: Minimum score to consider translation satisfactory

    Returns:
        Dictionary with final translation and process history
    """
    logger.info(f"Starting evaluator-optimizer translation process")
    logger.info(f"Source: {source_language} -> Target: {target_language}")
    logger.info(f"Max iterations: {max_iterations}, Quality threshold: {quality_threshold}")

    # generate initial translation
    current_translation = generate_initial_translation(source_text, target_language, source_language)

    iterations = []
    iteration_count = 0 

    while iteration_count < max_iterations:
        iteration_count += 1
        logger.info(f"---Iteration {iteration_count} ---")

        # evaluate current translation
        evaluation = evaluate_translation(source_text, current_translation.translated_text,
                                          target_language, source_language)
        
        iteration_data = {
            "iteration": iteration_count,
            "translation": current_translation.translated_text,
            "evaluation": evaluation,
            "optimization": None
        }

        # check if translation meets quality threshold 
        if evaluation.is_satisfactory and evaluation.overall_score >= quality_threshold:
            logger.info(f"Quality threshold met! Final socre: {evaluation.overall_score:.1f}")
            iteration_data["final"] = True 
            iterations.append(iteration_data)
            break 

        # if not final iteration, optimize based on feedback 
        if iteration_count < max_iterations:
            logger.info(f"Score {evaluation.overall_score:.1f} below threshold, optimizing...")
            optimization = optimize_translation(source_text, current_translation.translated_text,
                                                evaluation, target_language, source_language)
            
            # update current translation for next iteration
            current_translation=TranslationResponse(
                translated_text=optimization.improved_text,
                reasoning=optimization.reasoning
            )

if __name__ == "__main__":
    # exmpale 1. literary translation with cultural nuances
    source_text="""
    The old man sate by the window, wathcing the rain dance on the cobblestones. His weathered hands
    held a cup of tea that had long grown cold, but he didn't notice. In his mind, he was young again,
    walking those same streets with here, when the world was full of promise and their love felt eternal.
    """
    result = evaluator_optimizer_translation(
        source_text=source_text,
        target_language="French",
        source_language="English",
        max_iterations=3,
        quality_threshold=9
    )

    print(f"\n FINAL RESULT")
    print(f"\n Final translation: {result["final_translation"]}")
    print(f"\n Total Iterations: {result["total_iterations"]}")
    print(f"\n Final Score: {result["final_score"]:.1f}/10")

    # example 2: technical text with specific terminology
    technical_text = """
    The machine learning model exhibited overfitting behavior, achieving 99% 
    accuracy on the training dataset but only 65% 
    on the validatioun set.
    This performance gap  suggests the need for regularization techniques sucha as 
    dropout or L2 penalty
    """

    result_technical = evaluator_optimizer_translation(
        source_text=source_text,
        target_language="Spanish",
        source_language="English",
        max_iterations=2,
        quality_threshold=9.5
    )

    print(f"\n FINAL RESULT")
    print(f"\n Final translation: {result_technical["final_translation"]}")
    print(f"\n Total Iterations: {result_technical["total_iterations"]}")
    print(f"\n Final Score: {result_technical["final_score"]:.1f}/10")
