# app/agents/orchestrator_agent.py

import logging
from langchain.agents import create_agent
from langchain.tools import tool


logger = logging.getLogger(__name__)


class OrchestratorAgent:

    def __init__(self, report_service):

        self.report_service = report_service

        @tool
        def run_batch(project_hint: str) -> str:
            """
            Start batch summary asynchronously.
            """
            run_id = self.report_service.create_run(project_hint, "batch")
            self.report_service.execute_batch_report_async(run_id, project_hint)
            return f"Batch started. run_id={run_id}"

        @tool
        def run_global(project_hint: str) -> str:
            """
            Start final report asynchronously.
            """
            run_id = self.report_service.create_run(project_hint, "final")
            self.report_service.execute_final_report_async(run_id, project_hint)
            return f"Final started. run_id={run_id}"

        @tool
        def run_full(project_hint: str) -> str:
            """
            Start full workflow asynchronously.
            """
            run_id = self.report_service.create_run(project_hint, "full")
            self.report_service.execute_full_report_async(run_id, project_hint)
            return f"Full workflow started. run_id={run_id}"

        ORCH_SYSTEM_PROMPT = """
You are a workflow orchestrator.

You DO NOT execute heavy processing.
You only start workflows.

Available tools:
- run_batch
- run_global
- run_full

Always return run_id to user.
Never analyze emails yourself.
"""

        self.agent = create_agent(
            model="deepseek-reasoner",
            tools=[run_batch, run_global, run_full],
            system_prompt=ORCH_SYSTEM_PROMPT,
        )

    def invoke(self, message: str):
        return self.agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
