from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
import uvicorn
import base64
import hashlib
from fastapi.responses import Response
from report_generator import generate_report

from database import (
    save_alert, get_all_alerts, get_alerts_by_session, get_alert_summary,
    register_student, get_student, get_all_students,
    update_student_status, delete_student,
    remove_student_with_notice, get_removal_notice,
    students_col, alerts_col
)

from fastapi import WebSocket, WebSocketDisconnect
import json

# ── WebSocket Connection Manager ───────────────────────────────────────────
class ConnectionManager:
    """
    Manages all active WebSocket connections.
    When an alert fires, it broadcasts to every connected dashboard.
    """
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Dashboard connected. "
              f"Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"[WS] Dashboard disconnected. "
              f"Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Sends alert to every connected dashboard instantly."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception:
                disconnected.append(connection)
        # Clean up dead connections
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# ── FastAPI App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="ExamProctor API",
    description="Real-time exam proctoring system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────
class AlertModel(BaseModel):
    session_id:     str
    candidate_id:   str
    station_id:     str
    violation_type: str
    objects:        list[str]      = []
    confidence:     Optional[float] = None
    timestamp:      Optional[str]   = None
    evidence_frame: Optional[str]   = None  # base64 encoded jpeg

class AdminLogin(BaseModel):
    username: str
    password: str

class LoginModel(BaseModel):
    registration_number: str
    password:            str

class StatusUpdate(BaseModel):
    status: str

class ExamAssign(BaseModel):
    exam_url: str

class DeletionNotice(BaseModel):
    note: str


# ═══════════════════════════════════════════════════════════════════════════
# ROOT
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {
        "system":  "ExamProctor API",
        "version": "2.0.0",
        "status":  "running"
    }


# ═══════════════════════════════════════════════════════════════════════════
# ALERT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/alerts")
async def receive_alert(alert: AlertModel):
    from datetime import timezone
    alert_data = {
        "session_id":     alert.session_id,
        "candidate_id":   alert.candidate_id,
        "station_id":     alert.station_id,
        "violation_type": alert.violation_type,
        "objects":        alert.objects,
        "confidence":     alert.confidence,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "evidence_frame": alert.evidence_frame,
    }
    alert_id = await save_alert(alert_data)

    print(f"[ALERT SAVED] {alert.violation_type} | "
          f"Candidate={alert.candidate_id} | ID={alert_id} | "
          f"Evidence={'YES' if alert.evidence_frame else 'NO'}")


    # Broadcast to WebSocket connections
    await manager.broadcast({
        "type":           "new_alert",
        "alert_id":       alert_id,
        "session_id":     alert.session_id,
        "candidate_id":   alert.candidate_id,
        "station_id":     alert.station_id,
        "violation_type": alert.violation_type,
        "objects":        alert.objects,
        "timestamp":      alert_data["timestamp"],
        "evidence_frame": alert.evidence_frame,
    })

    return {"status": "saved", "alert_id": alert_id}

   


@app.get("/alerts")
async def list_alerts():
    alerts = await get_all_alerts()
    return {"total": len(alerts), "alerts": alerts}


@app.get("/alerts/session/{session_id}")
async def alerts_by_session(session_id: str):
    alerts = await get_alerts_by_session(session_id)
    return {"session_id": session_id, "total": len(alerts), "alerts": alerts}


@app.get("/alerts/summary")
async def alert_summary():
    summary = await get_alert_summary()
    return {"summary": summary}


@app.delete("/alerts")
async def clear_alerts():
    result = await alerts_col.delete_many({})
    return {"status": "cleared", "deleted": result.deleted_count}


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alert streaming.
    React dashboard connects here and receives alerts instantly.
    """
    await manager.connect(websocket)
    try:
        # Send last 10 alerts immediately on connect
        # so dashboard is not empty when supervisor opens it
        recent = await get_all_alerts()
        await websocket.send_text(json.dumps({
            "type":   "history",
            "alerts": recent[:10]
        }))
        # Keep connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ═══════════════════════════════════════════════════════════════════════════
# ADMIN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/admin/login")
async def admin_login(credentials: AdminLogin):
    if credentials.username == "admin" and credentials.password == "admin123":
        return {
            "status":   "success",
            "username": "admin",
            "role":     "admin"
        }
    raise HTTPException(status_code=401, detail="Invalid admin credentials")


# ═══════════════════════════════════════════════════════════════════════════
# STUDENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/students/register")
async def register_new_student(
    full_name:           str        = Form(...),
    registration_number: str        = Form(...),
    password:            str        = Form(...),
    department:          str        = Form(...),
    passport_photo:      UploadFile = File(...)
):
    """
    Student self-registration from the web portal.
    Admin assigns exam_id and exam_url after approval.
    """
    photo_bytes  = await passport_photo.read()
    photo_base64 = base64.b64encode(photo_bytes).decode("utf-8")
    photo_mime   = passport_photo.content_type

    hashed_password = hashlib.sha256(password.encode()).hexdigest()

    student_data = {
        "full_name":           full_name,
        "registration_number": registration_number.upper().strip(),
        "password":            hashed_password,
        "department":          department,
        "exam_id":             None,
        "exam_url":            None,
        "passport_photo":      photo_base64,
        "passport_mime":       photo_mime,
        "face_encoding":       None,
    }

    student_id, message = await register_student(student_data)

    if not student_id:
        raise HTTPException(status_code=400, detail=message)

    print(f"[STUDENT REGISTERED] {full_name} | {registration_number}")
    return {
        "status":     "registered",
        "student_id": student_id,
        "message":    message
    }


@app.post("/students/login")
async def student_login(login: LoginModel):
    """
    Validates student credentials and returns their record.
    """
    student = await get_student(login.registration_number.upper().strip())

    if not student:
        # If this student was deleted by an admin, surface the note
        # they left explaining what happened and what to do next,
        # instead of a plain "not found".
        notice = await get_removal_notice(
            login.registration_number.upper().strip()
        )
        if notice:
            raise HTTPException(
                status_code=410,
                detail=notice.get(
                    "note",
                    "Your registration was removed. "
                    "Please contact your institution."
                )
            )
        raise HTTPException(status_code=404, detail="Registration number not found")

    hashed_input = hashlib.sha256(login.password.encode()).hexdigest()
    if student["password"] != hashed_input:
        raise HTTPException(status_code=401, detail="Incorrect password")

    if student["status"] == "pending":
        raise HTTPException(
            status_code=403,
            detail="Your registration is pending admin approval. Please check back later."
        )

    if student["status"] == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Your registration was not approved. Please contact your institution."
        )

    await update_student_status(
        login.registration_number.upper().strip(), "verified"
    )

    print(f"[LOGIN SUCCESS] {student['full_name']} | {student['registration_number']}")

    return {
        "status":              "success",
        "full_name":           student["full_name"],
        "registration_number": student["registration_number"],
        "department":          student["department"],
        "exam_id":             student.get("exam_id"),
        "exam_url":            student.get("exam_url"),
        "passport_photo":      student["passport_photo"],
        "passport_mime":       student["passport_mime"],
        "student_status":      student["status"],
    }


@app.get("/students")
async def list_students():
    students = await get_all_students()
    for s in students:
        s.pop("password",      None)
        s.pop("face_encoding", None)
    return {"total": len(students), "students": students}


@app.patch("/students/{registration_number}/status")
async def update_status(registration_number: str, body: StatusUpdate):
    await update_student_status(
        registration_number.upper().strip(), body.status
    )
    return {"status": "updated"}


@app.patch("/students/{registration_number}/exam")
async def assign_exam_url(registration_number: str, body: ExamAssign):
    await students_col.update_one(
        {"registration_number": registration_number.upper().strip()},
        {"$set": {"exam_url": body.exam_url}}
    )
    return {"status": "updated"}


@app.delete("/students/{registration_number}")
async def remove_student(registration_number: str, body: DeletionNotice):
    """
    Deletes a student, but first records the admin's note explaining
    the removal. If the student (or anyone using that registration
    number) tries to log in afterward, they'll see this note instead
    of a generic error — so they know what happened and what to do next.
    """
    reg  = registration_number.upper().strip()
    note = (body.note or "").strip()

    if not note:
        raise HTTPException(
            status_code=400,
            detail="A note explaining the removal is required."
        )

    deleted = await remove_student_with_notice(reg, note)
    if not deleted:
        raise HTTPException(status_code=404, detail="Student not found")

    print(f"[STUDENT REMOVED] {reg} | Note: {note}")
    return {"status": "deleted", "note": note}


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run(
        "main_api:app",
        host="127.0.0.1",
        port=8000,
        reload= False
    )

# ═══════════════════════════════════════════════════════════════════════════
# REPORT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/reports/{registration_number}")
async def download_report(registration_number: str):
    """
    Generates and returns a PDF integrity report for a student.
    Called when admin clicks Download Report on the dashboard.
    """
    # Fetch student record
    student = await get_student(registration_number.upper().strip())
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Remove sensitive fields
    student.pop("password",       None)
    student.pop("face_encoding",  None)
    student.pop("passport_photo", None)

    # Fetch all alerts for this student
    alerts = await get_alerts_by_session(
        student.get("exam_id", registration_number)
    )

    # Also fetch by candidate_id in case session_id differs
    extra_alerts = []
    cursor = alerts_col.find(
        {"candidate_id": registration_number.upper().strip()}
    ).sort("timestamp", 1)
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        extra_alerts.append(doc)

    # Merge and deduplicate
    all_ids    = {a["_id"] for a in alerts}
    for a in extra_alerts:
        if a["_id"] not in all_ids:
            alerts.append(a)
            all_ids.add(a["_id"])

    # Sort by timestamp
    alerts.sort(key=lambda x: x.get("timestamp", ""))

    # Generate PDF
    pdf_bytes = generate_report(student, alerts)

    filename = (
        f"ExamProctor_Report_"
        f"{registration_number.replace('/', '_')}.pdf"
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )