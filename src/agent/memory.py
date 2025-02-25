"""
Memory system for browser agent.
Handles storage and retrieval of browsing history, interactions,
and learned patterns.
"""

from typing import Dict, List, Optional, Union
import json
import time
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

from ..core.types import PageData, InteractionResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MemoryItem:
    """Single memory item structure."""
    type: str  # 'page', 'interaction', or 'pattern'
    data: Dict
    timestamp: float
    importance: float
    tags: List[str]
    embedding: Optional[np.ndarray] = None

class Memory:
    """Memory system for storing and analyzing browsing history."""

    def __init__(
        self,
        max_size: int = 1000,
        embedding_dim: int = 384,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize memory system.

        Args:
            max_size: Maximum number of items to store
            embedding_dim: Dimension of memory embeddings
            similarity_threshold: Threshold for pattern recognition
        """
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Memory storage
        self.items: deque = deque(maxlen=max_size)
        self.patterns: Dict = {}
        self.index: Dict = {}  # URL to memory mapping
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "pages_visited": 0,
            "interactions": 0,
            "patterns_found": 0
        }
        
        logger.info(f"Memory system initialized with capacity: {max_size}")

    def add(
        self,
        url: str,
        data: Union[PageData, Dict],
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Add page visit to memory.

        Args:
            url: Page URL
            data: Page data or analysis
            tags: Optional tags for categorization
        """
        # Create memory item
        item = MemoryItem(
            type="page",
            data=asdict(data) if isinstance(data, PageData) else data,
            timestamp=time.time(),
            importance=self._calculate_importance(data),
            tags=tags or [],
            embedding=self._generate_embedding(data)
        )
        
        # Add to memory
        self.items.append(item)
        self.index[url] = item
        
        # Update stats
        self.stats["total_items"] += 1
        self.stats["pages_visited"] += 1
        
        # Analyze for patterns
        self._analyze_patterns()
        
        logger.debug(f"Added memory for URL: {url}")

    def add_interaction(
        self,
        selector: str,
        action: str,
        result: InteractionResult
    ) -> None:
        """
        Add interaction to memory.

        Args:
            selector: Element selector
            action: Interaction type
            result: Interaction result
        """
        item = MemoryItem(
            type="interaction",
            data={
                "selector": selector,
                "action": action,
                "result": asdict(result)
            },
            timestamp=time.time(),
            importance=0.5 if result.success else 0.8,
            tags=[action, "interaction"],
            embedding=self._generate_embedding(result)
        )
        
        self.items.append(item)
        self.stats["interactions"] += 1
        
        logger.debug(f"Added interaction memory: {action} on {selector}")

    def get_similar(
        self,
        query: Union[str, Dict],
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        Find similar memories.

        Args:
            query: Search query or data
            limit: Maximum number of results

        Returns:
            List of similar memory items
        """
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for item in self.items:
            if item.embedding is not None:
                similarity = self._calculate_similarity(
                    query_embedding,
                    item.embedding
                )
                similarities.append((similarity, item))
        
        # Sort and return top matches
        similarities.sort(reverse=True)
        return [item for _, item in similarities[:limit]]

    def get_pattern(self, context: str) -> Optional[Dict]:
        """
        Get learned pattern for context.

        Args:
            context: Pattern context

        Returns:
            Pattern data if found
        """
        return self.patterns.get(context)

    def clear(self) -> None:
        """Clear all memory."""
        self.items.clear()
        self.patterns.clear()
        self.index.clear()
        self.stats = {k: 0 for k in self.stats}
        logger.info("Memory cleared")

    def save(self, path: str) -> None:
        """
        Save memory to file.

        Args:
            path: File path
        """
        data = {
            "items": [asdict(item) for item in self.items],
            "patterns": self.patterns,
            "stats": self.stats
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Memory saved to: {path}")

    def load(self, path: str) -> None:
        """
        Load memory from file.

        Args:
            path: File path
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.items = deque(
            [MemoryItem(**item) for item in data["items"]],
            maxlen=self.max_size
        )
        self.patterns = data["patterns"]
        self.stats = data["stats"]
        
        logger.info(f"Memory loaded from: {path}")

    def _calculate_importance(self, data: Union[PageData, Dict]) -> float:
        """Calculate importance score for memory item."""
        # Implement importance scoring logic
        return 0.5

    def _generate_embedding(
        self,
        data: Union[str, Dict, PageData, InteractionResult]
    ) -> np.ndarray:
        """Generate embedding for memory item."""
        # Implement embedding generation
        return np.random.randn(self.embedding_dim)

    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        return float(np.dot(embedding1, embedding2) / 
               (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def _analyze_patterns(self) -> None:
        """Analyze memory for patterns."""
        # Implement pattern recognition logic
        pass

    @property
    def size(self) -> int:
        """Get current memory size."""
        return len(self.items)

    def __len__(self) -> int:
        return self.size