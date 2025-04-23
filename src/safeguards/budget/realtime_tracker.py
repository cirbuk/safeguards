"""Real-time budget tracking module with websocket support."""

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal

import websockets
from websockets.server import WebSocketServerProtocol

from safeguards.budget.cost_tracker import CostTracker
from safeguards.budget.token_tracker import TokenTracker


@dataclass
class BudgetAlert:
    """Budget alert notification."""

    agent_id: str
    alert_type: str  # "threshold", "violation", "emergency"
    message: str
    current_usage: Decimal
    budget_limit: Decimal
    usage_percentage: float
    timestamp: datetime


@dataclass
class BudgetUpdate:
    """Real-time budget update."""

    agent_id: str
    current_usage: Decimal
    remaining_budget: Decimal
    usage_percentage: float
    alert: BudgetAlert | None = None


class RealtimeBudgetTracker:
    """Tracks budget usage in real-time with websocket notifications."""

    def __init__(
        self,
        cost_tracker: CostTracker,
        token_tracker: TokenTracker,
        websocket_port: int = 8765,
        alert_thresholds: dict[str, float] | None = None,
    ):
        """Initialize realtime budget tracker.

        Args:
            cost_tracker: Cost tracking instance
            token_tracker: Token tracking instance
            websocket_port: Port for websocket server
            alert_thresholds: Dict of alert levels and their thresholds
                            e.g. {"warning": 0.7, "critical": 0.9}
        """
        self.cost_tracker = cost_tracker
        self.token_tracker = token_tracker
        self.websocket_port = websocket_port
        self.alert_thresholds = alert_thresholds or {
            "warning": 0.7,
            "critical": 0.9,
            "emergency": 0.95,
        }

        self.connected_clients: set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False

    async def start(self):
        """Start the websocket server."""
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            "localhost",
            self.websocket_port,
        )
        await self._server.wait_closed()

    async def stop(self):
        """Stop the websocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new websocket client connection.

        Args:
            websocket: Websocket client connection
            path: Connection path
        """
        try:
            self.connected_clients.add(websocket)
            # Send initial budget status
            await self._send_budget_status(websocket)

            # Keep connection alive and handle incoming messages
            while self._running:
                try:
                    await websocket.recv()
                    # Handle client messages if needed
                    await asyncio.sleep(0.1)
                except websockets.exceptions.ConnectionClosed:
                    break
        finally:
            self.connected_clients.remove(websocket)

    async def _send_budget_status(self, websocket: WebSocketServerProtocol):
        """Send current budget status to client.

        Args:
            websocket: Client websocket connection
        """
        status = self.cost_tracker.get_budget_status()
        await websocket.send(json.dumps(status, default=str))

    async def broadcast_update(self, update: BudgetUpdate):
        """Broadcast budget update to all connected clients.

        Args:
            update: Budget update to broadcast
        """
        message = json.dumps(asdict(update), default=str)
        await asyncio.gather(
            *[client.send(message) for client in self.connected_clients],
        )

    def check_alerts(
        self,
        agent_id: str,
        usage: Decimal,
        budget: Decimal,
    ) -> BudgetAlert | None:
        """Check for budget alerts based on usage.

        Args:
            agent_id: Agent ID
            usage: Current usage
            budget: Budget limit

        Returns:
            BudgetAlert if threshold exceeded, None otherwise
        """
        usage_percentage = float(usage / budget)

        # Check thresholds from highest to lowest
        if usage_percentage >= self.alert_thresholds["emergency"]:
            return BudgetAlert(
                agent_id=agent_id,
                alert_type="emergency",
                message=f"Emergency: Budget usage at {usage_percentage:.1%}",
                current_usage=usage,
                budget_limit=budget,
                usage_percentage=usage_percentage,
                timestamp=datetime.now(),
            )
        if usage_percentage >= self.alert_thresholds["critical"]:
            return BudgetAlert(
                agent_id=agent_id,
                alert_type="critical",
                message=f"Critical: Budget usage at {usage_percentage:.1%}",
                current_usage=usage,
                budget_limit=budget,
                usage_percentage=usage_percentage,
                timestamp=datetime.now(),
            )
        if usage_percentage >= self.alert_thresholds["warning"]:
            return BudgetAlert(
                agent_id=agent_id,
                alert_type="warning",
                message=f"Warning: Budget usage at {usage_percentage:.1%}",
                current_usage=usage,
                budget_limit=budget,
                usage_percentage=usage_percentage,
                timestamp=datetime.now(),
            )
        return None

    async def update_budget_usage(self, agent_id: str, usage: Decimal, budget: Decimal):
        """Update budget usage and send notifications.

        Args:
            agent_id: Agent ID
            usage: Current usage amount
            budget: Budget limit
        """
        # Calculate metrics
        remaining = budget - usage
        usage_percentage = float(usage / budget)

        # Check for alerts
        alert = self.check_alerts(agent_id, usage, budget)

        # Create update
        update = BudgetUpdate(
            agent_id=agent_id,
            current_usage=usage,
            remaining_budget=remaining,
            usage_percentage=usage_percentage,
            alert=alert,
        )

        # Broadcast to all clients
        await self.broadcast_update(update)
