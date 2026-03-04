"""
TE 751 - Day 2 Lab: Building a Simple ReAct Agent for Network Monitoring
=========================================================================
This script demonstrates a ReAct (Reasoning + Acting) agent that can
query simulated network metrics and suggest optimization actions.

Run with: uv run python day2/react_agent.py
"""

import json
import random
from datetime import datetime

# ============================================================
# STEP 1: Define simulated network tools
# ============================================================


def get_cell_kpis(cell_id: str) -> dict:
    """Simulate retrieving KPIs for a given cell."""
    # In production, this would query an OSS/BSS API
    random.seed(hash(cell_id) % 100)
    return {
        "cell_id": cell_id,
        "timestamp": datetime.now().isoformat(),
        "throughput_mbps": round(random.uniform(50, 500), 1),
        "latency_ms": round(random.uniform(1, 50), 2),
        "packet_loss_pct": round(random.uniform(0, 5), 3),
        "connected_users": random.randint(10, 500),
        "prb_utilization_pct": round(random.uniform(20, 95), 1),
        "cqi_average": round(random.uniform(5, 15), 1),
    }


def get_active_alarms(cell_id: str) -> list:
    """Simulate retrieving active alarms for a cell."""
    possible_alarms = [
        {"severity": "CRITICAL", "description": "High PRB utilization (>90%)"},
        {"severity": "MAJOR", "description": "CQI degradation detected"},
        {"severity": "MINOR", "description": "Increased handover failures"},
        {"severity": "WARNING", "description": "Throughput below baseline"},
    ]
    random.seed(hash(cell_id) % 50)
    n_alarms = random.randint(0, 3)
    return random.sample(possible_alarms, min(n_alarms, len(possible_alarms)))


def adjust_cell_parameter(cell_id: str, parameter: str, value: float) -> dict:
    """Simulate adjusting a cell parameter."""
    valid_params = ["tx_power", "tilt", "bandwidth", "scheduling_weight"]
    if parameter not in valid_params:
        return {"status": "ERROR", "message": f"Invalid parameter. Valid: {valid_params}"}
    return {
        "status": "SUCCESS",
        "cell_id": cell_id,
        "parameter": parameter,
        "new_value": value,
        "message": f"Parameter {parameter} set to {value} for cell {cell_id}",
    }


# ============================================================
# STEP 2: Define the tool registry
# ============================================================

TOOLS = {
    "get_cell_kpis": {
        "function": get_cell_kpis,
        "description": "Get current KPIs for a cell. Args: cell_id (str)",
    },
    "get_active_alarms": {
        "function": get_active_alarms,
        "description": "Get active alarms for a cell. Args: cell_id (str)",
    },
    "adjust_cell_parameter": {
        "function": adjust_cell_parameter,
        "description": (
            "Adjust a cell parameter. "
            "Args: cell_id (str), parameter (str), value (float). "
            "Valid parameters: tx_power, tilt, bandwidth, scheduling_weight"
        ),
    },
}


# ============================================================
# STEP 3: Simple ReAct agent loop (no LLM needed for demo)
# ============================================================


class SimpleReActAgent:
    """
    A rule-based ReAct agent that demonstrates the Thought-Action-Observation
    loop pattern. In production, the 'Thought' step would use an LLM.
    """

    def __init__(self, tools: dict):
        self.tools = tools
        self.memory: list[dict] = []

    def think(self, observation: dict | None = None) -> str:
        """Generate a thought based on the current observation."""
        if observation is None:
            return "I should start by checking the cell KPIs to understand the current state."

        # Simple rule-based reasoning (replace with LLM in production)
        if "throughput_mbps" in str(observation):
            kpis = observation
            issues = []
            if kpis.get("prb_utilization_pct", 0) > 85:
                issues.append("PRB utilization is high")
            if kpis.get("latency_ms", 0) > 30:
                issues.append("latency is elevated")
            if kpis.get("packet_loss_pct", 0) > 2:
                issues.append("packet loss is concerning")
            if kpis.get("cqi_average", 15) < 8:
                issues.append("CQI is low, indicating poor channel quality")

            if issues:
                return f"Issues detected: {', '.join(issues)}. Let me check for active alarms."
            return "KPIs look healthy. Let me verify there are no active alarms."

        if isinstance(observation, list):  # Alarms
            if len(observation) > 0:
                critical = [a for a in observation if a["severity"] == "CRITICAL"]
                if critical:
                    return (
                        "Critical alarm detected! I should take corrective action "
                        "by adjusting cell parameters."
                    )
                return "Non-critical alarms present. Monitoring recommended. No further action needed."
            return "No active alarms. The cell is operating normally."

        if isinstance(observation, dict) and "status" in observation:
            return f"Action completed: {observation.get('message', 'Done')}. Task finished."

        return "Analysis complete."

    def act(self, thought: str, cell_id: str) -> tuple[str, dict]:
        """Decide which action to take based on the thought."""
        if "check" in thought.lower() and "kpi" in thought.lower():
            result = self.tools["get_cell_kpis"]["function"](cell_id)
            return "get_cell_kpis", result

        if "adjust" in thought.lower() or "corrective" in thought.lower():
            result = self.tools["adjust_cell_parameter"]["function"](
                cell_id, "tx_power", 43.0
            )
            return "adjust_cell_parameter", result

        if "alarm" in thought.lower():
            result = self.tools["get_active_alarms"]["function"](cell_id)
            return "get_active_alarms", result

        return "none", {}

    def run(self, cell_id: str, max_steps: int = 5):
        """Execute the ReAct loop."""
        print(f"\n{'='*60}")
        print(f"  ReAct Agent - Analyzing Cell: {cell_id}")
        print(f"{'='*60}\n")

        observation = None
        for step in range(1, max_steps + 1):
            # THOUGHT
            thought = self.think(observation)
            print(f"  Step {step} - THOUGHT: {thought}")

            # Check if done
            if any(phrase in thought.lower() for phrase in [
                "task finished", "operating normally", "no further action"
            ]):
                print(f"\n{'='*60}")
                print("  Agent completed analysis.")
                print(f"{'='*60}\n")
                break

            # ACTION
            action_name, observation = self.act(thought, cell_id)
            print(f"  Step {step} - ACTION:  {action_name}")
            print(f"  Step {step} - OBSERVATION: {json.dumps(observation, indent=2, default=str)}")
            print()

            # Store in memory
            self.memory.append({
                "step": step,
                "thought": thought,
                "action": action_name,
                "observation": observation,
            })


# ============================================================
# STEP 4: Run the agent
# ============================================================

if __name__ == "__main__":
    agent = SimpleReActAgent(tools=TOOLS)

    # Analyze different cells
    for cell in ["CELL-A1-001", "CELL-B2-047", "CELL-C3-102"]:
        agent.run(cell)

    # Show agent memory
    print("\n--- Agent Memory ---")
    print(f"Total actions taken: {len(agent.memory)}")
    for entry in agent.memory:
        print(f"  Step {entry['step']}: {entry['action']}")
