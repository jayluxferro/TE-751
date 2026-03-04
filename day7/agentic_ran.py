"""
TE 751 - Day 7 Workshop: End-to-End Agentic AI for RAN Management
==================================================================
A complete guided project building an AI agent with tools, planning,
memory, and testing on simulated network scenarios.

Run with: uv run python day7/agentic_ran.py
"""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# PART 1: Tool Functions
# ============================================================

class NetworkSimulator:
    """Simulates a RAN environment for agent interaction."""

    def __init__(self, n_cells=5, seed=42):
        random.seed(seed)
        self.cells = {}
        for i in range(n_cells):
            cid = f"gNB-{i+1:03d}"
            self.cells[cid] = {
                "cell_id": cid,
                "throughput_mbps": round(random.uniform(100, 600), 1),
                "latency_ms": round(random.uniform(2, 40), 1),
                "prb_util_pct": round(random.uniform(20, 95), 1),
                "connected_ues": random.randint(10, 400),
                "tx_power_dbm": round(random.uniform(33, 46), 1),
                "cqi_avg": round(random.uniform(5, 15), 1),
                "handover_fail_rate": round(random.uniform(0, 0.15), 3),
                "alarms": [],
            }
            # Generate alarms for problematic cells
            if self.cells[cid]["prb_util_pct"] > 85:
                self.cells[cid]["alarms"].append({
                    "id": f"ALM-{random.randint(1000,9999)}",
                    "severity": "CRITICAL",
                    "type": "HIGH_PRB_UTILIZATION",
                    "message": f"PRB utilization at {self.cells[cid]['prb_util_pct']}%",
                })
            if self.cells[cid]["cqi_avg"] < 7:
                self.cells[cid]["alarms"].append({
                    "id": f"ALM-{random.randint(1000,9999)}",
                    "severity": "MAJOR",
                    "type": "LOW_CQI",
                    "message": f"Average CQI dropped to {self.cells[cid]['cqi_avg']}",
                })
        self.action_history = []

    def get_kpis(self, cell_id: str) -> dict:
        """Tool: Retrieve KPIs for a cell."""
        if cell_id not in self.cells:
            return {"error": f"Cell {cell_id} not found"}
        cell = self.cells[cell_id].copy()
        cell.pop("alarms", None)
        return cell

    def get_alarms(self, cell_id: str) -> list:
        """Tool: Get active alarms for a cell."""
        if cell_id not in self.cells:
            return [{"error": f"Cell {cell_id} not found"}]
        return self.cells[cell_id]["alarms"]

    def get_all_cells(self) -> list:
        """Tool: List all cell IDs with summary status."""
        return [
            {
                "cell_id": cid,
                "status": "ALARM" if cell["alarms"] else "OK",
                "prb_util": cell["prb_util_pct"],
                "users": cell["connected_ues"],
            }
            for cid, cell in self.cells.items()
        ]

    def adjust_parameter(self, cell_id: str, param: str, value: float) -> dict:
        """Tool: Adjust a cell parameter."""
        valid = {"tx_power_dbm": (20, 46), "tilt_deg": (0, 15), "bandwidth_mhz": (5, 100)}
        if cell_id not in self.cells:
            return {"error": f"Cell {cell_id} not found"}
        if param not in valid:
            return {"error": f"Invalid param. Valid: {list(valid.keys())}"}
        lo, hi = valid[param]
        if not (lo <= value <= hi):
            return {"error": f"{param} must be between {lo} and {hi}"}

        old_val = self.cells[cell_id].get(param, "N/A")
        self.cells[cell_id][param] = value
        result = {
            "status": "SUCCESS",
            "cell_id": cell_id,
            "parameter": param,
            "old_value": old_val,
            "new_value": value,
        }
        self.action_history.append(result)
        return result

    def acknowledge_alarm(self, cell_id: str, alarm_id: str) -> dict:
        """Tool: Acknowledge and clear an alarm."""
        if cell_id not in self.cells:
            return {"error": f"Cell {cell_id} not found"}
        alarms = self.cells[cell_id]["alarms"]
        for i, a in enumerate(alarms):
            if a["id"] == alarm_id:
                removed = alarms.pop(i)
                return {"status": "ACKNOWLEDGED", "alarm": removed}
        return {"error": f"Alarm {alarm_id} not found on {cell_id}"}


# ============================================================
# PART 2: Agent with Planning and Reasoning
# ============================================================

@dataclass
class AgentMemory:
    """Persistent memory for the agent."""
    observations: list = field(default_factory=list)
    actions_taken: list = field(default_factory=list)
    lessons_learned: list = field(default_factory=list)

    def add_observation(self, obs: dict):
        self.observations.append({
            "timestamp": datetime.now().isoformat(),
            **obs,
        })

    def add_action(self, action: dict):
        self.actions_taken.append({
            "timestamp": datetime.now().isoformat(),
            **action,
        })

    def add_lesson(self, lesson: str):
        self.lessons_learned.append({
            "timestamp": datetime.now().isoformat(),
            "lesson": lesson,
        })

    def get_past_actions_for_cell(self, cell_id: str) -> list:
        return [a for a in self.actions_taken if a.get("cell_id") == cell_id]


class RANManagementAgent:
    """
    An agentic AI system for RAN management.
    Implements: Perceive -> Plan -> Act -> Learn
    """

    def __init__(self, simulator: NetworkSimulator):
        self.sim = simulator
        self.memory = AgentMemory()
        self.tools = {
            "get_all_cells": self.sim.get_all_cells,
            "get_kpis": self.sim.get_kpis,
            "get_alarms": self.sim.get_alarms,
            "adjust_parameter": self.sim.adjust_parameter,
            "acknowledge_alarm": self.sim.acknowledge_alarm,
        }

    def perceive(self) -> dict:
        """Step 1: Gather information about the network."""
        cells = self.sim.get_all_cells()
        self.memory.add_observation({"type": "network_scan", "cells": cells})

        problematic = [c for c in cells if c["status"] == "ALARM"]
        return {
            "total_cells": len(cells),
            "problematic_cells": problematic,
            "healthy_cells": len(cells) - len(problematic),
        }

    def plan(self, perception: dict) -> list:
        """Step 2: Create an action plan based on perception."""
        plan = []
        for cell in perception["problematic_cells"]:
            cell_id = cell["cell_id"]

            # Check if we've already acted on this cell recently
            past = self.memory.get_past_actions_for_cell(cell_id)
            if len(past) >= 2:
                plan.append({
                    "action": "SKIP",
                    "cell_id": cell_id,
                    "reason": "Already attempted remediation twice",
                })
                continue

            plan.append({
                "action": "INVESTIGATE",
                "cell_id": cell_id,
                "steps": [
                    f"Get detailed KPIs for {cell_id}",
                    f"Check alarms on {cell_id}",
                    "Determine root cause",
                    "Take corrective action",
                    "Verify improvement",
                ],
            })
        return plan

    def act(self, plan: list) -> list:
        """Step 3: Execute the plan."""
        results = []
        for item in plan:
            if item["action"] == "SKIP":
                results.append({"status": "SKIPPED", **item})
                continue

            cell_id = item["cell_id"]

            # Investigate
            kpis = self.sim.get_kpis(cell_id)
            alarms = self.sim.get_alarms(cell_id)

            # Determine action based on conditions
            actions_taken = []
            for alarm in alarms:
                if alarm["type"] == "HIGH_PRB_UTILIZATION":
                    # Try reducing power to shed load
                    result = self.sim.adjust_parameter(
                        cell_id, "tx_power_dbm",
                        max(33, kpis.get("tx_power_dbm", 43) - 3)
                    )
                    actions_taken.append({
                        "action": "reduce_power",
                        "cell_id": cell_id,
                        "result": result,
                    })
                    self.memory.add_action({
                        "cell_id": cell_id,
                        "action": "reduce_power",
                        "alarm": alarm["type"],
                    })

                    # Acknowledge the alarm
                    self.sim.acknowledge_alarm(cell_id, alarm["id"])

                elif alarm["type"] == "LOW_CQI":
                    # Increase power to improve signal quality
                    result = self.sim.adjust_parameter(
                        cell_id, "tx_power_dbm",
                        min(46, kpis.get("tx_power_dbm", 40) + 2)
                    )
                    actions_taken.append({
                        "action": "increase_power",
                        "cell_id": cell_id,
                        "result": result,
                    })
                    self.memory.add_action({
                        "cell_id": cell_id,
                        "action": "increase_power",
                        "alarm": alarm["type"],
                    })
                    self.sim.acknowledge_alarm(cell_id, alarm["id"])

            results.append({
                "cell_id": cell_id,
                "kpis": kpis,
                "alarms_found": len(alarms),
                "actions": actions_taken,
            })

        return results

    def learn(self, results: list):
        """Step 4: Extract lessons from the results."""
        for r in results:
            if r.get("status") == "SKIPPED":
                self.memory.add_lesson(
                    f"Cell {r['cell_id']}: Skipped (max retries reached). "
                    "Need escalation procedure."
                )
            elif r.get("actions"):
                for a in r["actions"]:
                    if a["result"].get("status") == "SUCCESS":
                        self.memory.add_lesson(
                            f"Cell {r['cell_id']}: {a['action']} succeeded. "
                            f"Changed {a['result']['parameter']} from "
                            f"{a['result']['old_value']} to {a['result']['new_value']}."
                        )

    def run_cycle(self, cycle_num: int):
        """Execute one full agent cycle."""
        print(f"\n{'='*55}")
        print(f"  Agent Cycle {cycle_num}")
        print(f"{'='*55}")

        # Perceive
        perception = self.perceive()
        print(f"\n  [PERCEIVE] Scanned {perception['total_cells']} cells: "
              f"{len(perception['problematic_cells'])} problematic, "
              f"{perception['healthy_cells']} healthy")

        if not perception["problematic_cells"]:
            print("  All cells healthy. No action needed.")
            return

        # Plan
        plan = self.plan(perception)
        print(f"  [PLAN] Created {len(plan)} action items:")
        for p in plan:
            print(f"    - {p['action']}: {p['cell_id']}")

        # Act
        results = self.act(plan)
        print(f"  [ACT] Executed actions:")
        for r in results:
            if r.get("status") == "SKIPPED":
                print(f"    {r['cell_id']}: SKIPPED ({r['reason']})")
            else:
                print(f"    {r['cell_id']}: {len(r.get('actions', []))} actions taken")
                for a in r.get("actions", []):
                    print(f"      -> {a['action']}: {a['result'].get('status', 'N/A')}")

        # Learn
        self.learn(results)
        print(f"  [LEARN] Lessons recorded: {len(self.memory.lessons_learned)}")


# ============================================================
# PART 4: Run the simulation
# ============================================================

def main():
    print("=" * 55)
    print("  Agentic AI for RAN Management - Workshop")
    print("=" * 55)

    sim = NetworkSimulator(n_cells=5, seed=42)
    agent = RANManagementAgent(sim)

    # Run 3 cycles
    for cycle in range(1, 4):
        agent.run_cycle(cycle)

    # Summary
    print(f"\n{'='*55}")
    print("  Agent Summary")
    print(f"{'='*55}")
    print(f"  Total observations: {len(agent.memory.observations)}")
    print(f"  Total actions taken: {len(agent.memory.actions_taken)}")
    print(f"  Lessons learned: {len(agent.memory.lessons_learned)}")
    for lesson in agent.memory.lessons_learned:
        print(f"    - {lesson['lesson']}")


if __name__ == "__main__":
    main()
