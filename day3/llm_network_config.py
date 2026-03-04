"""
TE 751 - Day 3 Lab: LLM-Powered Network Configuration Interface
================================================================
This script demonstrates how to use an LLM to translate natural-language
operator intent into structured network configuration commands.

Run with: uv run python day3/llm_network_config.py
"""

import json
import re
from dataclasses import dataclass

# ============================================================
# STEP 1: Define the network configuration schema
# ============================================================


@dataclass
class NetworkCommand:
    """Represents a structured network configuration command."""
    action: str          # e.g., "set", "get", "delete", "create"
    target: str          # e.g., "cell", "slice", "bearer"
    parameters: dict     # e.g., {"cell_id": "A1", "tx_power": 40}
    priority: str        # "low", "medium", "high", "critical"
    requires_approval: bool  # whether human must approve

    def to_cli(self) -> str:
        """Convert to CLI-style command string."""
        params = " ".join(f"--{k}={v}" for k, v in self.parameters.items())
        return f"nw-cli {self.action} {self.target} {params}"


# ============================================================
# STEP 2: Intent parser (simulates LLM output parsing)
# ============================================================

# In production, this would call an actual LLM API like:
#   from langchain_anthropic import ChatAnthropic
#   llm = ChatAnthropic(model="claude-sonnet-4-20250514")
#
# For this lab, we use pattern matching to demonstrate the concept.

INTENT_PATTERNS = {
    r"(?:increase|raise|boost)\s+(?:the\s+)?(?:tx\s*)?power\s+(?:of|on|for)\s+(?:cell\s+)?(\S+)\s+(?:to|by)\s+(\d+)": {
        "action": "set",
        "target": "cell",
        "param_key": "tx_power",
        "priority": "medium",
        "approval": False,
    },
    r"(?:decrease|reduce|lower)\s+(?:the\s+)?(?:tx\s*)?power\s+(?:of|on|for)\s+(?:cell\s+)?(\S+)\s+(?:to|by)\s+(\d+)": {
        "action": "set",
        "target": "cell",
        "param_key": "tx_power",
        "priority": "medium",
        "approval": False,
    },
    r"(?:create|provision|add)\s+(?:a\s+)?(?:network\s+)?slice\s+(?:for|named|called)\s+(\S+)\s+with\s+(\d+)\s*(?:mbps|Mbps)": {
        "action": "create",
        "target": "slice",
        "param_key": "guaranteed_mbps",
        "priority": "high",
        "approval": True,
    },
    r"(?:shut\s*down|disable|deactivate)\s+(?:cell\s+)?(\S+)": {
        "action": "delete",
        "target": "cell",
        "param_key": None,
        "priority": "critical",
        "approval": True,
    },
    r"(?:show|get|display|check)\s+(?:the\s+)?(?:status|kpis?|metrics?)\s+(?:of|for)\s+(?:cell\s+)?(\S+)": {
        "action": "get",
        "target": "cell",
        "param_key": None,
        "priority": "low",
        "approval": False,
    },
    r"(?:set|change|modify)\s+(?:the\s+)?tilt\s+(?:of|on|for)\s+(?:cell\s+)?(\S+)\s+to\s+(\d+)": {
        "action": "set",
        "target": "cell",
        "param_key": "tilt",
        "priority": "medium",
        "approval": False,
    },
}


def parse_intent(user_input: str) -> NetworkCommand | None:
    """
    Parse natural language intent into a structured NetworkCommand.

    In production, this function would:
    1. Send the user input to an LLM with a system prompt
    2. Parse the LLM's structured output (JSON)
    3. Validate against the schema
    """
    user_input = user_input.lower().strip()

    for pattern, config in INTENT_PATTERNS.items():
        match = re.search(pattern, user_input)
        if match:
            groups = match.groups()
            params = {}

            if config["target"] == "cell":
                params["cell_id"] = groups[0].upper()
            elif config["target"] == "slice":
                params["slice_name"] = groups[0]

            if config["param_key"] and len(groups) > 1:
                params[config["param_key"]] = float(groups[1])

            return NetworkCommand(
                action=config["action"],
                target=config["target"],
                parameters=params,
                priority=config["priority"],
                requires_approval=config["approval"],
            )

    return None


# ============================================================
# STEP 3: Command executor (simulated)
# ============================================================


def execute_command(cmd: NetworkCommand) -> dict:
    """Simulate executing a network command."""
    if cmd.requires_approval:
        return {
            "status": "PENDING_APPROVAL",
            "command": cmd.to_cli(),
            "message": f"This {cmd.priority}-priority action requires human approval.",
        }

    # Simulate execution
    return {
        "status": "SUCCESS",
        "command": cmd.to_cli(),
        "message": f"Executed: {cmd.action} on {cmd.target}",
        "result": cmd.parameters,
    }


# ============================================================
# STEP 4: The LLM prompt template (for reference)
# ============================================================

SYSTEM_PROMPT = """You are a network configuration assistant for a 6G
telecommunications network. Convert the operator's natural language
request into a structured JSON command.

Output format:
{
    "action": "set|get|create|delete",
    "target": "cell|slice|bearer|policy",
    "parameters": {...},
    "priority": "low|medium|high|critical",
    "requires_approval": true|false
}

Rules:
- Shutdown/delete operations ALWAYS require approval
- Create operations on slices require approval
- Get/query operations never require approval
- Always validate cell IDs match pattern [A-Z][0-9]-[0-9]{3}
"""


# ============================================================
# STEP 5: Interactive demo
# ============================================================

def demo():
    """Run the intent-to-command demo."""
    test_inputs = [
        "Increase the power of cell A1-001 to 43",
        "Show the status of cell B2-047",
        "Create a network slice for eMBB with 500 Mbps",
        "Shutdown cell C3-102",
        "Set the tilt of cell A1-001 to 6",
        "Reduce power of cell B2-047 to 35",
    ]

    print("=" * 60)
    print("  LLM-Powered Network Configuration Interface")
    print("=" * 60)

    for user_input in test_inputs:
        print(f"\n  Operator: \"{user_input}\"")
        cmd = parse_intent(user_input)

        if cmd:
            print(f"  Parsed:   {cmd.action} {cmd.target} {cmd.parameters}")
            print(f"  CLI:      {cmd.to_cli()}")
            print(f"  Priority: {cmd.priority}")

            result = execute_command(cmd)
            print(f"  Status:   {result['status']}")
            if result["status"] == "PENDING_APPROVAL":
                print(f"  ⚠ {result['message']}")
        else:
            print("  ✗ Could not parse intent. Needs LLM fallback.")

    print(f"\n{'=' * 60}")
    print("  In production, replace parse_intent() with an LLM call.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    demo()
