"""API routes for budget override management."""

from datetime import timedelta
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..core.budget_override import (
    BudgetOverrideManager,
    OverrideRequest,
    OverrideType,
)
from ..core.dynamic_budget import AgentPriority
from ..utils.auth import get_current_user

router = APIRouter(prefix="/api/v1/budget-overrides", tags=["budget-overrides"])


class OverrideRequestModel(BaseModel):
    """Model for override request submission."""

    agent_id: str = Field(..., description="ID of the agent requesting override")
    requested_amount: Decimal = Field(..., description="Requested budget amount")
    override_type: OverrideType = Field(..., description="Type of override")
    justification: str = Field(..., description="Reason for override request")
    duration_hours: int | None = Field(
        None,
        description="Duration in hours for temporary overrides",
    )
    priority_override: AgentPriority | None = Field(
        None,
        description="Optional priority change",
    )
    metadata: dict | None = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class OverrideActionModel(BaseModel):
    """Model for override approval/rejection actions."""

    reason: str = Field(..., description="Reason for approval/rejection")


@router.post(
    "/request",
    response_model=dict[str, UUID],
    status_code=status.HTTP_201_CREATED,
)
async def create_override_request(
    request: OverrideRequestModel,
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> dict[str, UUID]:
    """Create a new budget override request."""
    try:
        duration = timedelta(hours=request.duration_hours) if request.duration_hours else None
        request_id = override_manager.request_override(
            agent_id=request.agent_id,
            requested_amount=request.requested_amount,
            override_type=request.override_type,
            justification=request.justification,
            requester=current_user,
            duration=duration,
            priority_override=request.priority_override,
            metadata=request.metadata,
        )
        return {"request_id": request_id}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{override_id}/approve")
async def approve_override(
    override_id: UUID,
    action: OverrideActionModel,
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> dict[str, str]:
    """Approve a budget override request."""
    try:
        override_manager.approve_override(
            override_id=override_id,
            approver=current_user,
        )
        return {
            "status": "approved",
            "message": "Override request approved successfully",
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{override_id}/reject")
async def reject_override(
    override_id: UUID,
    action: OverrideActionModel,
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> dict[str, str]:
    """Reject a budget override request."""
    try:
        override_manager.reject_override(
            override_id=override_id,
            rejector=current_user,
            reason=action.reason,
        )
        return {
            "status": "rejected",
            "message": "Override request rejected successfully",
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{override_id}", response_model=OverrideRequest)
async def get_override_status(
    override_id: UUID,
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> OverrideRequest:
    """Get the current status of an override request."""
    try:
        return override_manager.get_override_status(override_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/agent/{agent_id}", response_model=list[OverrideRequest])
async def get_agent_overrides(
    agent_id: str,
    include_inactive: bool = False,
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> list[OverrideRequest]:
    """Get all override requests for an agent."""
    return override_manager.get_agent_overrides(agent_id, include_inactive)


@router.get("/active", response_model=list[OverrideRequest])
async def get_active_overrides(
    override_manager: BudgetOverrideManager = Depends(),
    current_user: str = Depends(get_current_user),
) -> list[OverrideRequest]:
    """Get all currently active overrides."""
    active_overrides = []
    for agent_id in override_manager._active_overrides:
        overrides = override_manager.get_agent_overrides(
            agent_id,
            include_inactive=False,
        )
        active_overrides.extend(overrides)
    return active_overrides
