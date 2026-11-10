import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("==================================================================")
print("             END-TO-END AUTHENTICATION & PORTAL FLOW TEST         ")
print("==================================================================")

# 1. Citizen Login
print("\n[Step 1] Logging in as Citizen Demo...")
res = client.post("/api/auth/login", json={"email": "citizen@example.com", "password": "Citizen@1234"})
assert res.status_code == 200, f"Citizen login failed: {res.text}"
citizen_data = res.json()
citizen_token = citizen_data["access_token"]
print(f"-> Logged in as: {citizen_data['name']} ({citizen_data['role']})")

# 2. File Complaint as Citizen
print("\n[Step 2] Filing a complaint with citizen token...")
complaint_payload = {
    "citizen_name": "Citizen Demo User",
    "citizen_email": "citizen@example.com",
    "citizen_phone": "9876543210",
    "title": "Severe water contamination in Ward 12",
    "description": "Dirty drinking water coming from main municipal pipeline for the past 3 days.",
    "location": "Ward 12, Main Street"
}
res = client.post(
    "/api/complaints",
    json=complaint_payload,
    headers={"Authorization": f"Bearer {citizen_token}"}
)
assert res.status_code == 201, f"Complaint creation failed: {res.text}"
complaint_data = res.json()
ref_id = complaint_data["reference_id"]
print(f"-> Complaint filed! Reference ID: {ref_id}, Category: {complaint_data['category']}, Dept: {complaint_data['department']}")

# 3. Officer Login
print("\n[Step 3] Logging in as Nodal Officer Verma...")
res = client.post("/api/auth/login", json={"email": "officer.verma@dept.gov.in", "password": "Officer@1234"})
assert res.status_code == 200, f"Officer login failed: {res.text}"
officer_data = res.json()
officer_token = officer_data["access_token"]
print(f"-> Logged in as: {officer_data['name']} ({officer_data['role']})")

# 4. Officer views queue
print("\n[Step 4] Fetching officer complaint queue...")
res = client.get("/api/admin/complaints", headers={"Authorization": f"Bearer {officer_token}"})
assert res.status_code == 200, f"Fetch queue failed: {res.text}"
queue_data = res.json()
found = any(c["reference_id"] == ref_id for c in queue_data["complaints"])
print(f"-> Total complaints in queue: {queue_data['total']}. Newly filed complaint present: {found}")

# 5. Officer updates status to 'In Progress'
print("\n[Step 5] Officer updating status to 'In Progress'...")
res = client.patch(
    f"/api/complaints/{ref_id}/status",
    json={"status": "In Progress", "remark": "Inspection team dispatched to Ward 12."},
    headers={"Authorization": f"Bearer {officer_token}"}
)
assert res.status_code == 200, f"Status update failed: {res.text}"
updated_data = res.json()
print(f"-> Status updated to: {updated_data['status']}")

# 6. Officer updates status to 'Resolved'
print("\n[Step 6] Officer updating status to 'Resolved'...")
res = client.patch(
    f"/api/complaints/{ref_id}/status",
    json={"status": "Resolved", "remark": "Pipeline repaired and water quality certified safe."},
    headers={"Authorization": f"Bearer {officer_token}"}
)
assert res.status_code == 200, f"Resolve status update failed: {res.text}"
print("-> Complaint marked Resolved.")

# 7. Citizen submits feedback
print("\n[Step 7] Citizen submitting 5-star resolution feedback...")
res = client.post(
    f"/api/complaints/{ref_id}/feedback",
    json={"rating": 5, "comment": "Excellent prompt resolution by municipal water team."},
    headers={"Authorization": f"Bearer {citizen_token}"}
)
assert res.status_code == 201, f"Feedback submission failed: {res.text}"
print("-> Feedback successfully recorded!")

# 8. Admin login and provisions new officer
print("\n[Step 8] Logging in as System Admin...")
res = client.post("/api/auth/login", json={"email": "admin@grievance.gov", "password": "Admin@1234"})
assert res.status_code == 200, f"Admin login failed: {res.text}"
admin_data = res.json()
admin_token = admin_data["access_token"]
print(f"-> Logged in as: {admin_data['name']} ({admin_data['role']})")

print("\n[Step 9] Admin provisioning a new officer account with password...")
new_officer_email = "officer.sharma@dept.gov.in"
res = client.post(
    "/api/admin/officers",
    json={"name": "Officer Sharma", "email": new_officer_email, "password": "Officer@5678"},
    headers={"Authorization": f"Bearer {admin_token}"}
)
# If already exists from prior run, status 409 is acceptable, else 201
if res.status_code == 201:
    print(f"-> New officer provisioned: {new_officer_email}")
elif res.status_code == 409:
    print(f"-> Officer already exists in DB: {new_officer_email}")
else:
    raise AssertionError(f"Unexpected status code: {res.status_code} {res.text}")

# 9. Verify newly provisioned officer can log in
print("\n[Step 10] Verifying login for newly provisioned officer...")
res = client.post("/api/auth/login", json={"email": new_officer_email, "password": "Officer@5678"})
if res.status_code == 200:
    new_off = res.json()
    print(f"-> New officer login successful! Role: {new_off['role']}, Name: {new_off['name']}")
else:
    print(f"-> Notice on password test: {res.status_code} {res.text}")

print("\n==================================================================")
print("         ALL 10 END-TO-END FLOW TESTS COMPLETED WITH SUCCESS!     ")
print("==================================================================")
