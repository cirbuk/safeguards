"""Slack interaction handlers for budget override approvals."""

import hashlib
import hmac
import json
import os
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from ..core.budget_override import BudgetOverrideManager

router = APIRouter(prefix="/api/v1/slack", tags=["slack"])

# Load from environment
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET", "")  # Required in production


class SlackInteraction(BaseModel):
    """Model for Slack interaction payload."""

    type: str
    actions: list
    user: dict[str, str]
    response_url: str
    trigger_id: str
    token: str


def verify_slack_signature(request: Request) -> bool:
    """Verify that the request came from Slack.

    Args:
        request: FastAPI request object

    Returns:
        True if signature is valid
    """
    if not SLACK_SIGNING_SECRET:
        return True  # Skip verification in development

    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    # Compute expected signature
    base_string = f"v0:{timestamp}:{request.body.decode()}"
    expected_signature = (
        "v0="
        + hmac.new(
            SLACK_SIGNING_SECRET.encode(),
            base_string.encode(),
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(signature, expected_signature)


@router.post("/interaction")
async def handle_slack_interaction(
    request: Request,
    override_manager: BudgetOverrideManager = Depends(),
) -> dict:
    """Handle Slack interaction callbacks.

    Args:
        request: FastAPI request object
        override_manager: Budget override manager instance

    Returns:
        Response to send back to Slack
    """
    if not verify_slack_signature(request):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Slack signature",
        )

    # Parse payload
    form_data = await request.form()
    payload = json.loads(form_data.get("payload", "{}"))
    interaction = SlackInteraction(**payload)

    if not interaction.actions:
        return {"text": "No action specified"}

    action = interaction.actions[0]
    action_id = action.get("action_id")
    value = action.get("value", "")

    if action_id not in ["approve_override", "reject_override"]:
        return {"text": "Invalid action"}

    # Extract override ID from value (format: "approve_UUID" or "reject_UUID")
    override_id = UUID(value.split("_")[1])
    user = interaction.user["username"]

    try:
        if action_id == "approve_override":
            override_manager.approve_override(
                override_id=override_id,
                approver=user,
            )
            return {
                "text": f"Override request {override_id} approved successfully",
                "replace_original": True,
            }
        # reject_override
        override_manager.reject_override(
            override_id=override_id,
            rejector=user,
            reason="Rejected via Slack",
        )
        return {
            "text": f"Override request {override_id} rejected",
            "replace_original": True,
        }

    except ValueError as e:
        return {
            "text": f"Error processing override: {e!s}",
            "replace_original": False,
        }
