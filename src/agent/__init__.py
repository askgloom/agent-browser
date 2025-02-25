"""
Agent-Browser: An intelligent browser automation agent
powered by large language models and computer vision.

This module provides the core agent functionality for web browsing,
interaction, and task completion.
"""

from typing import Dict, List, Optional, Union
import logging

from .browser import BrowserAgent
from .memory import Memory
from .reasoning import Reasoning

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Agent:
    """Main agent class that coordinates browsing, memory, and reasoning."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        memory_size: int = 1000,
        headless: bool = False,
        debug: bool = False
    ):
        """
        Initialize the agent with specified configuration.

        Args:
            model: The LLM model to use for reasoning
            memory_size: Maximum number of items to keep in memory
            headless: Whether to run browser in headless mode
            debug: Enable debug logging
        """
        self.browser = BrowserAgent(headless=headless)
        self.memory = Memory(max_size=memory_size)
        self.reasoning = Reasoning(model=model)
        
        if debug:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
            
        logger.info(f"Agent initialized with model: {model}")

    async def browse(self, url: str) -> Dict:
        """
        Navigate to a URL and analyze the page.

        Args:
            url: The URL to navigate to

        Returns:
            Dict containing page analysis and extracted information
        """
        logger.debug(f"Browsing to {url}")
        page_data = await self.browser.navigate(url)
        analysis = await self.reasoning.analyze(page_data)
        self.memory.add(url, analysis)
        return analysis

    async def interact(
        self,
        selector: str,
        action: str,
        value: Optional[str] = None
    ) -> bool:
        """
        Interact with an element on the page.

        Args:
            selector: CSS selector for the element
            action: Type of interaction (click, type, etc.)
            value: Optional value for interaction (e.g. text to type)

        Returns:
            Success status of the interaction
        """
        logger.debug(f"Interacting with {selector}: {action}")
        result = await self.browser.interact(selector, action, value)
        self.memory.add_interaction(selector, action, result)
        return result

    async def complete_task(self, task: str) -> Dict:
        """
        Complete a specified task using browsing and interaction.

        Args:
            task: Natural language description of the task

        Returns:
            Dict containing task results and steps taken
        """
        logger.info(f"Starting task: {task}")
        
        # Generate plan
        plan = await self.reasoning.plan_task(task)
        
        # Execute steps
        results = []
        for step in plan:
            if step["type"] == "browse":
                result = await self.browse(step["url"])
            elif step["type"] == "interact":
                result = await self.interact(
                    step["selector"],
                    step["action"],
                    step.get("value")
                )
            results.append(result)
            
        # Analyze results
        summary = await self.reasoning.summarize_results(results)
        self.memory.add_task(task, summary)
        
        return {
            "task": task,
            "steps": len(results),
            "success": summary["success"],
            "results": summary
        }

    def reset(self) -> None:
        """Reset agent state and clear memory."""
        self.browser.reset()
        self.memory.clear()
        logger.info("Agent reset complete")

    @property
    def state(self) -> Dict:
        """Get current agent state."""
        return {
            "browser": self.browser.state,
            "memory": self.memory.stats,
            "model": self.reasoning.model
        }

__all__ = ["Agent", "BrowserAgent", "Memory", "Reasoning"]