"""
Built-in tools for the agent framework.

Provides essential tools for memory management, calculation,
and other common agent tasks.
"""

import json
import logging
import math
import operator
import re
from typing import Any, Dict, Optional

from sglang.srt.agents.tool_registry import Tool, ToolParameter, ToolParameterType

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Management Tools
# ============================================================================


class MemoryStoreTool:
    """
    Tool for storing information to conversation memory.

    This tool integrates with the snapshot/tier system to persist
    important facts, user preferences, and conversation context.
    """

    @staticmethod
    def create_tool(tier_manager=None) -> Tool:
        """
        Create memory store tool.

        Args:
            tier_manager: Optional TierManager instance for integration

        Returns:
            Tool instance
        """

        def memory_store(
            key: str,
            value: str,
            category: str = "general",
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Store information to memory.

            Args:
                key: Memory key (e.g., "user_name", "favorite_color")
                value: Value to store
                category: Category for organization (general, user_info, preferences, etc.)
                _conversation_context: Injected conversation context

            Returns:
                Status dict
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            # Store in conversation context metadata
            if _conversation_context is not None:
                if "memory" not in _conversation_context:
                    _conversation_context["memory"] = {}

                if category not in _conversation_context["memory"]:
                    _conversation_context["memory"][category] = {}

                _conversation_context["memory"][category][key] = value

            logger.info(
                f"Memory stored: conversation={conversation_id}, "
                f"category={category}, key={key}"
            )

            return {
                "status": "success",
                "message": f"Stored '{key}' in category '{category}'",
                "conversation_id": conversation_id,
            }

        return Tool(
            name="memory_store",
            description="Store information to conversation memory for later recall",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ToolParameterType.STRING,
                    description="Memory key (e.g., 'user_name', 'project_goal')",
                    required=True,
                ),
                ToolParameter(
                    name="value",
                    type=ToolParameterType.STRING,
                    description="Value to store",
                    required=True,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Category for organization (default: general)",
                    required=False,
                    default="general",
                    enum=["general", "user_info", "preferences", "facts", "tasks"],
                ),
            ],
            function=memory_store,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "storage"]},
        )


class MemoryRecallTool:
    """
    Tool for recalling information from conversation memory.
    """

    @staticmethod
    def create_tool(tier_manager=None) -> Tool:
        """
        Create memory recall tool.

        Args:
            tier_manager: Optional TierManager instance for integration

        Returns:
            Tool instance
        """

        def memory_recall(
            key: Optional[str] = None,
            category: Optional[str] = None,
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Recall information from memory.

            Args:
                key: Specific key to recall (None = list all in category)
                category: Category to search (None = search all)
                _conversation_context: Injected conversation context

            Returns:
                Memory data
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            if _conversation_context is None or "memory" not in _conversation_context:
                return {
                    "status": "not_found",
                    "message": "No memory data available",
                    "conversation_id": conversation_id,
                }

            memory = _conversation_context["memory"]

            # Recall specific key in category
            if key is not None and category is not None:
                if category in memory and key in memory[category]:
                    return {
                        "status": "success",
                        "key": key,
                        "value": memory[category][key],
                        "category": category,
                        "conversation_id": conversation_id,
                    }
                else:
                    return {
                        "status": "not_found",
                        "message": f"Key '{key}' not found in category '{category}'",
                        "conversation_id": conversation_id,
                    }

            # List all in category
            elif category is not None:
                if category in memory:
                    return {
                        "status": "success",
                        "category": category,
                        "data": memory[category],
                        "conversation_id": conversation_id,
                    }
                else:
                    return {
                        "status": "not_found",
                        "message": f"Category '{category}' not found",
                        "conversation_id": conversation_id,
                    }

            # List all memory
            else:
                return {
                    "status": "success",
                    "data": memory,
                    "conversation_id": conversation_id,
                }

        return Tool(
            name="memory_recall",
            description="Recall information from conversation memory",
            parameters=[
                ToolParameter(
                    name="key",
                    type=ToolParameterType.STRING,
                    description="Specific key to recall (None = list all)",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Category to search (None = search all)",
                    required=False,
                ),
            ],
            function=memory_recall,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "retrieval"]},
        )


class MemorySearchTool:
    """
    Tool for searching memory by keyword or pattern.
    """

    @staticmethod
    def create_tool() -> Tool:
        """Create memory search tool."""

        def memory_search(
            query: str,
            category: Optional[str] = None,
            _conversation_context: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            Search memory by keyword or pattern.

            Args:
                query: Search query (substring match)
                category: Optional category filter
                _conversation_context: Injected conversation context

            Returns:
                Search results
            """
            conversation_id = (
                _conversation_context.get("conversation_id")
                if _conversation_context
                else "unknown"
            )

            if _conversation_context is None or "memory" not in _conversation_context:
                return {
                    "status": "not_found",
                    "message": "No memory data available",
                    "results": [],
                }

            memory = _conversation_context["memory"]
            results = []

            # Search in specified category or all categories
            categories_to_search = (
                [category] if category else list(memory.keys())
            )

            for cat in categories_to_search:
                if cat not in memory:
                    continue

                for key, value in memory[cat].items():
                    # Search in key or value
                    if query.lower() in key.lower() or query.lower() in str(
                        value
                    ).lower():
                        results.append(
                            {
                                "category": cat,
                                "key": key,
                                "value": value,
                            }
                        )

            return {
                "status": "success",
                "query": query,
                "results": results,
                "count": len(results),
                "conversation_id": conversation_id,
            }

        return Tool(
            name="memory_search",
            description="Search conversation memory by keyword or pattern",
            parameters=[
                ToolParameter(
                    name="query",
                    type=ToolParameterType.STRING,
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="category",
                    type=ToolParameterType.STRING,
                    description="Optional category filter",
                    required=False,
                ),
            ],
            function=memory_search,
            requires_conversation_context=True,
            metadata={"tags": ["memory", "search"]},
        )


# ============================================================================
# Utility Tools
# ============================================================================


class CalculatorTool:
    """
    Tool for performing mathematical calculations.
    """

    @staticmethod
    def create_tool() -> Tool:
        """Create calculator tool."""

        def calculator(expression: str) -> Dict[str, Any]:
            """
            Evaluate a mathematical expression.

            Args:
                expression: Math expression (e.g., "2 + 2", "sqrt(16)")

            Returns:
                Calculation result
            """
            # Allowed operations and functions
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                # Math functions
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "floor": math.floor,
                "ceil": math.ceil,
                # Constants
                "pi": math.pi,
                "e": math.e,
            }

            try:
                # Use eval with restricted namespace for safety
                result = eval(expression, {"__builtins__": {}}, allowed_names)

                return {
                    "status": "success",
                    "expression": expression,
                    "result": result,
                }

            except Exception as e:
                return {
                    "status": "error",
                    "expression": expression,
                    "error": str(e),
                }

        return Tool(
            name="calculator",
            description="Perform mathematical calculations (basic arithmetic, sqrt, sin, cos, etc.)",
            parameters=[
                ToolParameter(
                    name="expression",
                    type=ToolParameterType.STRING,
                    description="Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')",
                    required=True,
                ),
            ],
            function=calculator,
            metadata={"tags": ["utility", "math"]},
        )


# ============================================================================
# Helper Functions
# ============================================================================


def register_builtin_tools(tool_registry, tier_manager=None):
    """
    Register all built-in tools to a registry.

    Args:
        tool_registry: ToolRegistry instance
        tier_manager: Optional TierManager for memory tools

    Returns:
        Number of tools registered
    """
    tools = [
        MemoryStoreTool.create_tool(tier_manager),
        MemoryRecallTool.create_tool(tier_manager),
        MemorySearchTool.create_tool(),
        CalculatorTool.create_tool(),
    ]

    for tool in tools:
        tool_registry.register(tool)

    logger.info(f"Registered {len(tools)} built-in tools")

    return len(tools)
