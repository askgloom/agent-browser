"""
Reasoning system for browser agent.
Handles decision making, task planning, and content analysis
using large language models.
"""

from typing import Dict, List, Optional, Union, Tuple
import asyncio
import json
from dataclasses import dataclass
import numpy as np

from ..core.types import PageData, InteractionResult, Confidence
from ..utils.logger import get_logger
from ..core.context import Context

# Import your preferred LLM client
from openai import AsyncOpenAI

logger = get_logger(__name__)

@dataclass
class ReasoningResult:
    """Structure for reasoning outputs."""
    decision: str
    confidence: float
    explanation: str
    next_actions: List[Dict]
    metadata: Dict

class Reasoning:
    """Reasoning system using LLMs for decision making."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        context_window: int = 5,
        api_key: Optional[str] = None
    ):
        """
        Initialize reasoning system.

        Args:
            model: LLM model to use
            temperature: Model temperature
            max_tokens: Maximum tokens per request
            context_window: Number of previous interactions to consider
            api_key: Optional API key
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        
        # Initialize LLM client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Initialize context
        self.context = Context()
        
        logger.info(f"Reasoning system initialized with model: {model}")

    async def analyze(self, page_data: PageData) -> Dict:
        """
        Analyze page content and structure.

        Args:
            page_data: Page data to analyze

        Returns:
            Analysis results
        """
        prompt = self._create_analysis_prompt(page_data)
        
        response = await self._get_completion(prompt)
        
        try:
            analysis = json.loads(response)
            return {
                "content": analysis.get("content", {}),
                "structure": analysis.get("structure", {}),
                "actions": analysis.get("possible_actions", []),
                "importance": analysis.get("importance", 0.5)
            }
        except json.JSONDecodeError:
            logger.error("Failed to parse analysis response")
            return {}

    async def plan_task(self, task: str) -> List[Dict]:
        """
        Generate plan for completing task.

        Args:
            task: Task description

        Returns:
            List of planned steps
        """
        prompt = self._create_planning_prompt(task)
        
        response = await self._get_completion(prompt)
        
        try:
            plan = json.loads(response)
            return plan.get("steps", [])
        except json.JSONDecodeError:
            logger.error("Failed to parse planning response")
            return []

    async def evaluate_route(
        self,
        url: str,
        context: Dict
    ) -> Confidence:
        """
        Evaluate potential navigation route.

        Args:
            url: Target URL
            context: Current context

        Returns:
            Confidence score and reasoning
        """
        prompt = self._create_evaluation_prompt(url, context)
        
        response = await self._get_completion(prompt)
        
        try:
            evaluation = json.loads(response)
            return Confidence(
                score=float(evaluation.get("confidence", 0.0)),
                reasoning=evaluation.get("reasoning", "")
            )
        except (json.JSONDecodeError, ValueError):
            logger.error("Failed to parse evaluation response")
            return Confidence(score=0.0, reasoning="Error in evaluation")

    async def decide_interaction(
        self,
        elements: List[Dict],
        goal: str
    ) -> Optional[Dict]:
        """
        Decide which element to interact with.

        Args:
            elements: List of interactive elements
            goal: Interaction goal

        Returns:
            Selected element and action
        """
        prompt = self._create_interaction_prompt(elements, goal)
        
        response = await self._get_completion(prompt)
        
        try:
            decision = json.loads(response)
            return decision if decision.get("element") else None
        except json.JSONDecodeError:
            logger.error("Failed to parse interaction decision")
            return None

    async def summarize_results(
        self,
        results: List[Dict]
    ) -> Dict:
        """
        Summarize task results.

        Args:
            results: List of task results

        Returns:
            Summary of results
        """
        prompt = self._create_summary_prompt(results)
        
        response = await self._get_completion(prompt)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse summary response")
            return {}

    async def _get_completion(
        self,
        prompt: Union[str, List[Dict]]
    ) -> str:
        """Get completion from LLM."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt if isinstance(prompt, str) else json.dumps(prompt)}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            raise

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an intelligent browser agent assistant. 
        Your role is to help analyze web pages, plan interactions,
        and make decisions about navigation and task completion.
        Provide responses in JSON format."""

    def _create_analysis_prompt(self, page_data: PageData) -> str:
        """Create prompt for page analysis."""
        return f"""Analyze the following page content and structure:
        URL: {page_data.url}
        Title: {page_data.title}
        Content: {page_data.text[:1000]}...
        
        Provide analysis including:
        - Main content topics
        - Page structure
        - Possible interactions
        - Content importance
        
        Response format:
        {{
            "content": {{...}},
            "structure": {{...}},
            "possible_actions": [...],
            "importance": float
        }}
        """

    def _create_planning_prompt(self, task: str) -> str:
        """Create prompt for task planning."""
        return f"""Create a plan for the following task:
        {task}
        
        Break it down into steps including:
        - Navigation steps
        - Interaction steps
        - Data collection steps
        
        Response format:
        {{
            "steps": [
                {{"type": "browse", "url": "..."}},
                {{"type": "interact", "selector": "...", "action": "..."}}
            ]
        }}
        """

    def _create_evaluation_prompt(self, url: str, context: Dict) -> str:
        """Create prompt for route evaluation."""
        return f"""Evaluate the following navigation option:
        URL: {url}
        Context: {json.dumps(context)}
        
        Provide confidence score and reasoning.
        
        Response format:
        {{
            "confidence": float,
            "reasoning": "..."
        }}
        """

    def _create_interaction_prompt(self, elements: List[Dict], goal: str) -> str:
        """Create prompt for interaction decision."""
        return f"""Decide which element to interact with:
        Goal: {goal}
        Elements: {json.dumps(elements)}
        
        Select the best element and action.
        
        Response format:
        {{
            "element": {{...}},
            "action": "...",
            "reason": "..."
        }}
        """

    def _create_summary_prompt(self, results: List[Dict]) -> str:
        """Create prompt for results summary."""
        return f"""Summarize the following task results:
        {json.dumps(results)}
        
        Provide overview of success, findings, and next steps.
        
        Response format:
        {{
            "success": bool,
            "summary": "...",
            "findings": [...],
            "next_steps": [...]
        }}
        """
