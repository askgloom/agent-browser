"""
Browser agent module for web automation and interaction.
Handles browser control, navigation, and element interaction.
"""

from typing import Dict, List, Optional, Union
import asyncio
import logging
from playwright.async_api import async_playwright, Browser, Page, ElementHandle
from bs4 import BeautifulSoup
import cv2
import numpy as np

from ..core.types import InteractionResult, PageData
from ..utils.logger import get_logger
from ..utils.parser import extract_text, parse_html

logger = get_logger(__name__)

class BrowserAgent:
    """Browser automation and control agent."""

    def __init__(
        self,
        headless: bool = False,
        viewport: Dict[str, int] = {"width": 1920, "height": 1080},
        timeout: int = 30000,
        user_agent: Optional[str] = None
    ):
        """
        Initialize browser agent.

        Args:
            headless: Run browser in headless mode
            viewport: Browser viewport dimensions
            timeout: Default timeout in milliseconds
            user_agent: Custom user agent string
        """
        self.headless = headless
        self.viewport = viewport
        self.timeout = timeout
        self.user_agent = user_agent
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self._setup_complete = False

    async def setup(self) -> None:
        """Initialize browser instance and configuration."""
        if self._setup_complete:
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless
        )
        self.context = await self.browser.new_context(
            viewport=self.viewport,
            user_agent=self.user_agent
        )
        self.page = await self.context.new_page()
        
        # Setup handlers
        await self._setup_handlers()
        self._setup_complete = True
        logger.info("Browser setup complete")

    async def navigate(self, url: str) -> PageData:
        """
        Navigate to URL and analyze page content.

        Args:
            url: Target URL

        Returns:
            PageData containing analysis and content
        """
        if not self._setup_complete:
            await self.setup()

        try:
            # Navigate and wait for network idle
            response = await self.page.goto(
                url,
                wait_until="networkidle",
                timeout=self.timeout
            )
            
            # Capture page state
            screenshot = await self.page.screenshot(
                type="jpeg",
                quality=80
            )
            html = await self.page.content()
            
            # Extract and analyze content
            text = extract_text(html)
            elements = await self._get_interactive_elements()
            
            return PageData(
                url=url,
                status=response.status if response else 0,
                title=await self.page.title(),
                html=html,
                text=text,
                screenshot=screenshot,
                elements=elements,
                timestamp=asyncio.get_event_loop().time()
            )

        except Exception as e:
            logger.error(f"Navigation failed: {str(e)}")
            raise

    async def interact(
        self,
        selector: str,
        action: str,
        value: Optional[str] = None
    ) -> InteractionResult:
        """
        Interact with page element.

        Args:
            selector: Element selector
            action: Interaction type (click, type, etc)
            value: Optional value for interaction

        Returns:
            Result of interaction
        """
        try:
            element = await self.page.wait_for_selector(
                selector,
                timeout=self.timeout
            )
            
            if not element:
                raise ValueError(f"Element not found: {selector}")

            # Perform interaction
            if action == "click":
                await element.click()
            elif action == "type":
                await element.type(value or "")
            elif action == "select":
                await element.select_option(value or "")
            else:
                raise ValueError(f"Unknown action: {action}")

            # Capture result
            screenshot = await self.page.screenshot(
                type="jpeg",
                quality=80
            )

            return InteractionResult(
                success=True,
                selector=selector,
                action=action,
                value=value,
                screenshot=screenshot,
                timestamp=asyncio.get_event_loop().time()
            )

        except Exception as e:
            logger.error(f"Interaction failed: {str(e)}")
            return InteractionResult(
                success=False,
                selector=selector,
                action=action,
                value=value,
                error=str(e),
                timestamp=asyncio.get_event_loop().time()
            )

    async def evaluate(self, script: str) -> any:
        """
        Evaluate JavaScript in page context.

        Args:
            script: JavaScript code to evaluate

        Returns:
            Result of evaluation
        """
        return await self.page.evaluate(script)

    async def find_element(
        self,
        text: str,
        tag: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Optional[ElementHandle]:
        """
        Find element by text content.

        Args:
            text: Text to search for
            tag: Optional HTML tag to filter by
            timeout: Custom timeout

        Returns:
            Found element or None
        """
        selector = f"//*[contains(text(), '{text}')]"
        if tag:
            selector = f"//{tag}[contains(text(), '{text}')]"
            
        try:
            return await self.page.wait_for_selector(
                selector,
                timeout=timeout or self.timeout
            )
        except:
            return None

    async def get_element_screenshot(
        self,
        selector: str
    ) -> Optional[bytes]:
        """
        Take screenshot of specific element.

        Args:
            selector: Element selector

        Returns:
            Screenshot bytes or None
        """
        try:
            element = await self.page.wait_for_selector(selector)
            if element:
                return await element.screenshot()
        except:
            return None

    async def reset(self) -> None:
        """Reset browser state and clear data."""
        if self.context:
            await self.context.clear_cookies()
            await self.context.clear_permissions()
        if self.page:
            await self.page.close()
        self.page = await self.context.new_page()
        logger.info("Browser state reset")

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._setup_complete = False
        logger.info("Browser closed")

    async def _setup_handlers(self) -> None:
        """Setup page event handlers."""
        await self.page.set_viewport_size(self.viewport)
        
        # Handle dialogs automatically
        self.page.on("dialog", lambda dialog: dialog.accept())
        
        # Setup request interception
        await self.page.route("**/*", self._handle_route)
        
        # Handle console messages
        self.page.on("console", self._handle_console)

    async def _handle_route(self, route: any) -> None:
        """Handle network requests."""
        if route.request.resource_type in ["image", "media"]:
            await route.abort()
        else:
            await route.continue_()

    def _handle_console(self, msg: any) -> None:
        """Handle console messages."""
        logger.debug(f"Console: {msg.text}")

    async def _get_interactive_elements(self) -> List[Dict]:
        """Get all interactive elements on page."""
        elements = []
        for selector in ["button", "a", "input", "select"]:
            items = await self.page.query_selector_all(selector)
            for item in items:
                elements.append({
                    "tag": selector,
                    "text": await item.text_content(),
                    "visible": await item.is_visible(),
                    "enabled": await item.is_enabled()
                })
        return elements

    @property
    def state(self) -> Dict:
        """Get current browser state."""
        return {
            "url": self.page.url if self.page else None,
            "ready": self._setup_complete,
            "headless": self.headless
        }