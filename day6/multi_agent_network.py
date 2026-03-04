"""
TE 751 - Day 6 Lab: Multi-Agent Network Management System
==========================================================
A multi-agent system with specialized agents for traffic monitoring,
resource allocation, energy efficiency, and an orchestrator.

Run with: uv run python day6/multi_agent_network.py
"""

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# STEP 1: Define network state
# ============================================================

class CellStatus(Enum):
    NORMAL = "NORMAL"
    CONGESTED = "CONGESTED"
    UNDERUTILIZED = "UNDERUTILIZED"
    ALARM = "ALARM"


@dataclass
class CellState:
    cell_id: str
    throughput_mbps: float
    prb_utilization: float
    connected_users: int
    power_dbm: float
    energy_kwh: float
    status: CellStatus = CellStatus.NORMAL

    def update_status(self):
        if self.prb_utilization > 85:
            self.status = CellStatus.CONGESTED
        elif self.prb_utilization < 20:
            self.status = CellStatus.UNDERUTILIZED
        else:
            self.status = CellStatus.NORMAL


@dataclass
class NetworkState:
    cells: dict = field(default_factory=dict)
    timestamp: int = 0

    def generate(self, n_cells=6):
        """Generate a simulated network state."""
        random.seed(self.timestamp)
        for i in range(n_cells):
            cid = f"CELL-{chr(65 + i)}{i+1}"
            cell = CellState(
                cell_id=cid,
                throughput_mbps=round(random.uniform(50, 500), 1),
                prb_utilization=round(random.uniform(10, 98), 1),
                connected_users=random.randint(5, 300),
                power_dbm=round(random.uniform(30, 46), 1),
                energy_kwh=round(random.uniform(1, 10), 2),
            )
            cell.update_status()
            self.cells[cid] = cell
        return self


# ============================================================
# STEP 2: Define specialized agents
# ============================================================

@dataclass
class AgentAction:
    agent: str
    action: str
    target: str
    details: dict
    priority: int  # 1=low, 5=critical


class TrafficMonitorAgent:
    """Monitors traffic patterns and detects anomalies."""
    name = "TrafficMonitor"

    def analyze(self, state: NetworkState) -> list[AgentAction]:
        actions = []
        for cid, cell in state.cells.items():
            if cell.status == CellStatus.CONGESTED:
                actions.append(AgentAction(
                    agent=self.name,
                    action="ALERT_CONGESTION",
                    target=cid,
                    details={
                        "prb_utilization": cell.prb_utilization,
                        "users": cell.connected_users,
                        "recommendation": "offload_traffic",
                    },
                    priority=4,
                ))
            if cell.throughput_mbps < 100 and cell.connected_users > 100:
                actions.append(AgentAction(
                    agent=self.name,
                    action="ALERT_LOW_THROUGHPUT",
                    target=cid,
                    details={
                        "throughput": cell.throughput_mbps,
                        "users": cell.connected_users,
                    },
                    priority=3,
                ))
        return actions


class ResourceAllocatorAgent:
    """Optimizes resource allocation across cells."""
    name = "ResourceAllocator"

    def analyze(self, state: NetworkState) -> list[AgentAction]:
        actions = []
        congested = [c for c in state.cells.values() if c.status == CellStatus.CONGESTED]
        underutil = [c for c in state.cells.values() if c.status == CellStatus.UNDERUTILIZED]

        # Load balancing: shift from congested to underutilized
        for cong in congested:
            if underutil:
                target = underutil[0]
                actions.append(AgentAction(
                    agent=self.name,
                    action="LOAD_BALANCE",
                    target=cong.cell_id,
                    details={
                        "from_cell": cong.cell_id,
                        "to_cell": target.cell_id,
                        "users_to_move": min(30, cong.connected_users // 4),
                        "from_util": cong.prb_utilization,
                        "to_util": target.prb_utilization,
                    },
                    priority=4,
                ))
        return actions


class EnergyEfficiencyAgent:
    """Manages energy consumption across the network."""
    name = "EnergyManager"

    def analyze(self, state: NetworkState) -> list[AgentAction]:
        actions = []
        for cid, cell in state.cells.items():
            # Sleep underutilized cells during low-traffic periods
            if cell.status == CellStatus.UNDERUTILIZED and cell.connected_users < 15:
                actions.append(AgentAction(
                    agent=self.name,
                    action="CELL_SLEEP",
                    target=cid,
                    details={
                        "current_users": cell.connected_users,
                        "current_power": cell.power_dbm,
                        "energy_saved_kwh": round(cell.energy_kwh * 0.7, 2),
                    },
                    priority=2,
                ))
            # Reduce power on medium-load cells
            elif cell.prb_utilization < 50 and cell.power_dbm > 38:
                actions.append(AgentAction(
                    agent=self.name,
                    action="REDUCE_POWER",
                    target=cid,
                    details={
                        "current_power": cell.power_dbm,
                        "suggested_power": round(cell.power_dbm - 3, 1),
                        "energy_saved_kwh": round(cell.energy_kwh * 0.2, 2),
                    },
                    priority=2,
                ))
        return actions


# ============================================================
# STEP 3: Orchestrator agent
# ============================================================

class OrchestratorAgent:
    """
    Coordinates decisions from all specialized agents.
    Resolves conflicts and prioritizes actions.
    """
    name = "Orchestrator"

    def __init__(self):
        self.agents = [
            TrafficMonitorAgent(),
            ResourceAllocatorAgent(),
            EnergyEfficiencyAgent(),
        ]
        self.action_log: list[dict] = []

    def collect_proposals(self, state: NetworkState) -> list[AgentAction]:
        """Gather proposed actions from all agents."""
        all_actions = []
        for agent in self.agents:
            proposals = agent.analyze(state)
            all_actions.extend(proposals)
        return all_actions

    def resolve_conflicts(self, actions: list[AgentAction]) -> list[AgentAction]:
        """
        Resolve conflicting proposals.
        Rule: higher priority wins. If equal, traffic > resource > energy.
        """
        # Group by target cell
        by_target: dict[str, list[AgentAction]] = {}
        for a in actions:
            by_target.setdefault(a.target, []).append(a)

        resolved = []
        for target, cell_actions in by_target.items():
            # Check for conflicts
            has_sleep = any(a.action == "CELL_SLEEP" for a in cell_actions)
            has_load = any(a.action == "LOAD_BALANCE" for a in cell_actions)

            if has_sleep and has_load:
                # Conflict: can't sleep a cell that's receiving load
                # Keep load balance, drop sleep
                cell_actions = [a for a in cell_actions if a.action != "CELL_SLEEP"]

            # Sort by priority (highest first)
            cell_actions.sort(key=lambda a: a.priority, reverse=True)
            resolved.extend(cell_actions)

        return resolved

    def execute(self, state: NetworkState) -> list[dict]:
        """Full orchestration cycle."""
        # 1. Collect proposals
        proposals = self.collect_proposals(state)

        # 2. Resolve conflicts
        resolved = self.resolve_conflicts(proposals)

        # 3. Execute (simulated)
        results = []
        for action in resolved:
            result = {
                "agent": action.agent,
                "action": action.action,
                "target": action.target,
                "priority": action.priority,
                "details": action.details,
                "status": "EXECUTED",
            }
            results.append(result)
            self.action_log.append(result)

        return results


# ============================================================
# STEP 4: Run the simulation
# ============================================================

def main():
    print("=" * 60)
    print("  Multi-Agent Network Management System")
    print("=" * 60)

    orchestrator = OrchestratorAgent()

    for t in range(3):
        state = NetworkState(timestamp=t * 42 + 7).generate(n_cells=6)

        print(f"\n--- Time Step {t+1} ---")
        print(f"\n  Network State:")
        for cid, cell in state.cells.items():
            status_color = {
                CellStatus.NORMAL: "",
                CellStatus.CONGESTED: " [!CONGESTED]",
                CellStatus.UNDERUTILIZED: " [~UNDERUTIL]",
                CellStatus.ALARM: " [!!ALARM]",
            }
            print(f"    {cid}: PRB={cell.prb_utilization:.0f}%, "
                  f"Users={cell.connected_users}, "
                  f"Power={cell.power_dbm}dBm"
                  f"{status_color[cell.status]}")

        results = orchestrator.execute(state)

        if results:
            print(f"\n  Agent Actions ({len(results)} total):")
            for r in results:
                print(f"    [{r['agent']}] {r['action']} -> {r['target']} "
                      f"(priority={r['priority']})")
                for k, v in r["details"].items():
                    print(f"      {k}: {v}")
        else:
            print("\n  No actions needed. Network is healthy.")

    print(f"\n{'='*60}")
    print(f"  Total actions taken: {len(orchestrator.action_log)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
