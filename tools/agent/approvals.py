from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ApprovalItem:
    approval_id: str
    session_id: str
    tool_name: str
    tool_args: dict[str, Any]
    allowed_decisions: list[str]
    created_at: float
    status: str = "pending"  # pending（待审批）| approved（已批准）| rejected（已拒绝）
    decision: dict[str, Any] | None = None
    result: str | None = None


_STORE: dict[str, ApprovalItem] = {}


def create_approval(
    *,
    session_id: str | None,
    tool_name: str,
    tool_args: dict[str, Any],
    allowed_decisions: list[str] | None = None,
) -> ApprovalItem:
    aid = str(uuid.uuid4())
    item = ApprovalItem(
        approval_id=aid,
        session_id=(session_id or "default"),
        tool_name=tool_name,
        tool_args=dict(tool_args or {}),
        allowed_decisions=list(allowed_decisions or ["approve", "edit", "reject"]),
        created_at=time.time(),
    )
    _STORE[aid] = item
    return item


def get_approval(approval_id: str) -> ApprovalItem | None:
    return _STORE.get(approval_id)


def list_pending(session_id: str | None = None) -> list[dict[str, Any]]:
    sid = (session_id or "").strip() or None
    out: list[dict[str, Any]] = []
    for it in _STORE.values():
        if it.status != "pending":
            continue
        if sid and it.session_id != sid:
            continue
        out.append(asdict(it))
    out.sort(key=lambda x: x.get("created_at", 0.0), reverse=True)
    return out


def decide(
    *,
    approval_id: str,
    decision_type: str,
    edited_args: dict[str, Any] | None = None,
    note: str | None = None,
) -> ApprovalItem:
    it = _STORE.get(approval_id)
    if not it:
        raise KeyError("approval_id not found")
    if it.status != "pending":
        return it
    d = (decision_type or "").strip().lower()
    if d not in it.allowed_decisions:
        raise ValueError(f"decision {d} not allowed; allowed={it.allowed_decisions}")
    it.decision = {"type": d, "edited_args": edited_args, "note": note}
    if d == "reject":
        it.status = "rejected"
    else:
        it.status = "approved"
        if d == "edit" and edited_args is not None:
            it.tool_args = dict(edited_args)
    return it

