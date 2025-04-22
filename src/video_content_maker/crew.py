from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import YoutubeVideoSearchTool, YoutubeChannelSearchTool
from .tools.brave_search import BraveSearchTool
from langchain_mistralai import ChatMistralAI
import os
import traceback
import logging
from datetime import datetime
import json
from tenacity import RetryError
from requests.exceptions import HTTPError

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"youtube_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a custom formatter that includes more details
class DetailedFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'api_request'):
            record.msg = f"API Request: {json.dumps(record.api_request, indent=2)}\n{record.msg}"
        if hasattr(record, 'api_response'):
            record.msg = f"API Response: {json.dumps(record.api_response, indent=2)}\n{record.msg}"
        return super().format(record)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture more information
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Apply the custom formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(DetailedFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger("youtube_api")

def handle_api_error(e, context):
    """Handle API errors with detailed logging"""
    if isinstance(e, RetryError):
        logger.error(f"Retry error in {context}: {str(e)}")
        if hasattr(e, 'last_attempt'):
            logger.error(f"Last attempt error: {str(e.last_attempt.exception())}")
    elif isinstance(e, HTTPError):
        logger.error(f"HTTP error in {context}: {str(e)}")
        if hasattr(e, 'response'):
            logger.error(f"Response status: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
    else:
        logger.error(f"Unexpected error in {context}: {str(e)}")
    logger.error("Full stack trace:", exc_info=True)

#TOOLS
def create_youtube_search_tool(video_url):
    try:
        logger.info(f"Creating YouTube search tool for video: {video_url}")
        
        # Log the configuration being used
        config = dict(
            llm=dict(
                provider="mistralai",
                config=dict(
                    model="mistral-large-latest",
                    api_key=os.getenv('MISTRAL_API_KEY')
                )
            ),
            embedder=dict(
                provider="mistralai",
                config=dict(
                    model="mistral-large-latest",
                    api_key=os.getenv('MISTRAL_API_KEY')
                )
            )
        )
        
        logger.debug("Tool configuration:", extra={'api_request': config})
        
        tool = YoutubeVideoSearchTool(
            youtube_video_url=video_url,
            search_query="What are the main topics and key points discussed in this video?",
            config=config
        )
        
        logger.info("YouTube search tool created successfully")
        return tool
    except Exception as e:
        handle_api_error(e, "YouTube search tool creation")
        raise

def create_youtube_channel_search_tool(channel_url):
    try:
        logger.info(f"Creating YouTube channel search tool for channel: {channel_url}")
        
        # Log the configuration being used
        config = dict(
            llm=dict(
                provider="mistralai",
                config=dict(
                    model="mistral-large-latest",
                    api_key=os.getenv('MISTRAL_API_KEY')
                )
            ),
            embedder=dict(
                provider="mistralai",
                config=dict(
                    model="mistral-large-latest",
                    api_key=os.getenv('MISTRAL_API_KEY')
                )
            )
        )
        
        logger.debug("Tool configuration:", extra={'api_request': config})
        
        tool = YoutubeChannelSearchTool(
            youtube_channel_url=channel_url,
            search_query="What are the main topics and key points discussed in this channel?",
            config=config
        )
        
        logger.info("YouTube channel search tool created successfully")
        return tool
    except Exception as e:
        handle_api_error(e, "YouTube channel search tool creation")
        raise

brave_search_tool = BraveSearchTool()


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators'

@CrewBase
class VideoContentMaker():
    """VideoContentMaker crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def brand_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['brand_strategist'],
            verbose=True,
            tools=[
                brave_search_tool,
                create_youtube_channel_search_tool("{channel_url}")
            ]
        )

    @agent
    def email_marketing_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['email_marketing_manager'],
            verbose=True,
            tools=[
                create_youtube_search_tool("{video_url}")
            ]
        )

    @agent
    def blog_content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['blog_content_writer'],
            verbose=True,
            tools=[
                create_youtube_search_tool("{video_url}")
            ]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def brand_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['brand_strategy_task'],
        )

    @task
    def email_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['email_outline_task'],
            output_file='email_outline.md'
        )

    @task
    def blog_post_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['blog_post_outline_task'],
            output_file='blog_post_outline.md'
        )
    @crew
    def crew(self) -> Crew:
        """Creates the VideoContentMaker crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            llm=ChatMistralAI(
                model="mistral-large-latest",
                temperature=0.7,
                mistral_api_key=os.getenv('MISTRAL_API_KEY')
            )
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
