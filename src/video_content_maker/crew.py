from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import WebsiteSearchTool, YoutubeVideoSearchTool, YoutubeChannelSearchTool
from langchain_mistralai import ChatMistralAI
from langchain.embeddings import HuggingFaceEmbeddings
import os


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

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
        embeddings = HuggingFaceEmbeddings(model_name="mistralai/Mistral-7B-Instruct-v0.2")
        return Agent(
            config=self.agents_config['brand_strategist'],
            verbose=True,
            tools=[
                WebsiteSearchTool(embedding_model=embeddings),
                YoutubeChannelSearchTool(embedding_model=embeddings)
            ]
        )

    @agent
    def email_marketing_manager(self) -> Agent:
        embeddings = HuggingFaceEmbeddings(model_name="mistralai/Mistral-7B-Instruct-v0.2")
        return Agent(
            config=self.agents_config['email_marketing_manager'],
            verbose=True,
            tools=[
                YoutubeVideoSearchTool(embedding_model=embeddings)
            ]
        )

    @agent
    def blog_content_writer(self) -> Agent:
        embeddings = HuggingFaceEmbeddings(model_name="mistralai/Mistral-7B-Instruct-v0.2")
        return Agent(
            config=self.agents_config['blog_content_writer'],
            verbose=True,
            tools=[
                YoutubeVideoSearchTool(embedding_model=embeddings)
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
