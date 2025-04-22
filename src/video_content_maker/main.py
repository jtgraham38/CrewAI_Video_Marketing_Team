#!/usr/bin/env python
import sys
import warnings
import os
import traceback
from datetime import datetime
from pathlib import Path
from crewai import LLM

from video_content_maker.crew import VideoContentMaker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # llm = LLM(
    #     model="mistral/mistral-small-latest",
    #     api_key=os.getenv('MISTRAL_API_KEY')
    # )
    # x = llm.call("Hello, world!")
    # print(x)
    
    inputs = {
        'organization': "The Bearded Butchers",
        'website_url': "https://beardedbutchers.com/",
        'youtube_url': "https://www.youtube.com/@thebeardedbutchers",
        'video_url': "https://www.youtube.com/watch?v=DyKmsnhCrjM",
        'current_year': str(datetime.now().year)
    }
    
    try:
        VideoContentMaker().crew().kickoff(inputs=inputs)
    except Exception as e:
        print("Error occurred while running the crew:")
        print("Exception:", str(e))
        print("\nFull stack trace:")
        traceback.print_exc()
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        VideoContentMaker().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        print("Error occurred while training the crew:")
        print("Exception:", str(e))
        print("\nFull stack trace:")
        traceback.print_exc()
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        VideoContentMaker().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        print("Error occurred while replaying the crew:")
        print("Exception:", str(e))
        print("\nFull stack trace:")
        traceback.print_exc()
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        VideoContentMaker().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        print("Error occurred while testing the crew:")
        print("Exception:", str(e))
        print("\nFull stack trace:")
        traceback.print_exc()
        raise Exception(f"An error occurred while testing the crew: {e}")
