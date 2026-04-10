import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from observability.tools import DiagnoseBottleneckTool, NormalizeMetricsTool


@CrewBase
class Observability:
    """Observability crew."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def bottleneck_detective(self) -> Agent:
        return Agent(
            config=self.agents_config["bottleneck_detective"],  # type: ignore[index]
            llm=LLM(
                model=os.getenv(
                    "OPENROUTER_MODEL",
                    "openrouter/meta-llama/llama-3.2-3b-instruct",
                ),
                provider="openai",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            ),
            tools=[NormalizeMetricsTool(), DiagnoseBottleneckTool()],
            verbose=True,
        )

    @task
    def diagnose_training_bottleneck(self) -> Task:
        return Task(
            config=self.tasks_config["diagnose_training_bottleneck"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Observability crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
