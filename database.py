from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# ── MongoDB Connection ─────────────────────────────────────────────────────
MONGO_URL        = "mongodb://localhost:27017"
DATABASE_NAME    = "aiproctor"

client       = AsyncIOMotorClient(MONGO_URL)
db           = client[DATABASE_NAME]
alerts_col   = db["alerts"]
students_col = db["students"]
removal_notices_col = db["removal_notices"]


# ═══════════════════════════════════════════════════════════════════════════
# ALERT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def save_alert(alert_data: dict):
    alert_data["saved_at"] = datetime.utcnow()
    result = await alerts_col.insert_one(alert_data)
    return str(result.inserted_id)


async def get_all_alerts():
    alerts = []
    cursor = alerts_col.find().sort("saved_at", -1)
    async for document in cursor:
        document["_id"] = str(document["_id"])
        alerts.append(document)
    return alerts


async def get_alerts_by_session(session_id: str):
    alerts = []
    cursor = alerts_col.find(
        {"session_id": session_id}
    ).sort("saved_at", -1)
    async for document in cursor:
        document["_id"] = str(document["_id"])
        alerts.append(document)
    return alerts


async def get_alert_summary():
    pipeline = [
        {"$group": {
            "_id":   "$violation_type",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}}
    ]
    summary = []
    async for doc in alerts_col.aggregate(pipeline):
        summary.append(doc)
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# STUDENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

async def register_student(student_data: dict):
    """
    Saves a new student record to MongoDB.
    Checks for duplicate registration numbers first.
    """
    existing = await students_col.find_one(
        {"registration_number": student_data["registration_number"]}
    )
    if existing:
        return None, "Registration number already exists"

    student_data["registered_at"] = datetime.utcnow()
    student_data["status"]        = "pending"
    result = await students_col.insert_one(student_data)
    return str(result.inserted_id), "Student registered successfully"


async def get_student(registration_number: str):
    """
    Fetches a student record by registration number.
    """
    student = await students_col.find_one(
        {"registration_number": registration_number}
    )
    if student:
        student["_id"] = str(student["_id"])
    return student


async def get_all_students():
    """
    Returns all registered students newest first.
    """
    students = []
    cursor   = students_col.find().sort("registered_at", -1)
    async for document in cursor:
        document["_id"] = str(document["_id"])
        students.append(document)
    return students


async def update_student_status(registration_number: str, status: str):
    """
    Updates student exam status.
    Options: pending | approved | rejected | in_exam | completed | flagged
    """
    await students_col.update_one(
        {"registration_number": registration_number},
        {"$set": {
            "status":     status,
            "updated_at": datetime.utcnow()
        }}
    )


async def delete_student(registration_number: str):
    """
    Removes a student record from the database.
    """
    result = await students_col.delete_one(
        {"registration_number": registration_number}
    )
    return result.deleted_count > 0


# ═══════════════════════════════════════════════════════════════════════════
# REMOVAL NOTICE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
# Kept in a separate collection (keyed by registration_number, not by the
# student's _id) so the note survives even after the student document
# itself is deleted — it's what lets a removed student still see *why*
# and *what to do next* if they try to log in again.

async def save_removal_notice(registration_number: str, note: str, full_name: str = None):
    await removal_notices_col.update_one(
        {"registration_number": registration_number},
        {"$set": {
            "registration_number": registration_number,
            "full_name":           full_name,
            "note":                note,
            "created_at":          datetime.utcnow(),
        }},
        upsert=True
    )


async def get_removal_notice(registration_number: str):
    return await removal_notices_col.find_one(
        {"registration_number": registration_number}
    )


async def remove_student_with_notice(registration_number: str, note: str):
    """
    Records a note for the student being removed, then deletes their
    record. Returns False if there was no such student to delete.
    """
    student   = await students_col.find_one(
        {"registration_number": registration_number}
    )
    full_name = student.get("full_name") if student else None

    await save_removal_notice(registration_number, note, full_name)

    result = await students_col.delete_one(
        {"registration_number": registration_number}
    )
    return result.deleted_count > 0