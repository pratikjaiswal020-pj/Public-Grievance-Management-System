import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("==================================================================")
print("            TESTING STRICT ADMIN-ONLY PORTAL RESTRICTION          ")
print("==================================================================")

# 1. Admin login with full email
print("\n[Test 1] Admin login with admin@grievance.gov...")
res = client.post("/api/auth/login", json={"email": "admin@grievance.gov", "password": "Admin@1234"})
assert res.status_code == 200, f"Admin login failed: {res.text}"
admin_token = res.json()["access_token"]
print("-> SUCCESS: Admin authenticated. Role =", res.json()["role"])

# 2. Admin login with shortcut 'admin'
print("\n[Test 2] Admin login with username shortcut 'admin'...")
res = client.post("/api/auth/login", json={"username": "admin", "password": "Admin@1234"})
assert res.status_code == 200, f"Admin shortcut login failed: {res.text}"
print("-> SUCCESS: Admin shortcut authenticated. Role =", res.json()["role"])

# 3. Officer login attempt -> MUST BE REJECTED 403
print("\n[Test 3] Officer attempting to login...")
res = client.post("/api/auth/login", json={"email": "officer.verma@dept.gov.in", "password": "Officer@1234"})
assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"
print(f"-> SUCCESS: Officer blocked as expected (HTTP 403: {res.json()['detail']})")

# 4. Citizen login attempt -> MUST BE REJECTED 403
print("\n[Test 4] Citizen attempting to login...")
res = client.post("/api/auth/login", json={"email": "citizen@example.com", "password": "Citizen@1234"})
assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"
print(f"-> SUCCESS: Citizen blocked as expected (HTTP 403: {res.json()['detail']})")

# 5. Public registration attempt -> MUST BE REJECTED 403
print("\n[Test 5] Public user attempting registration...")
res = client.post("/api/auth/register", json={"name": "Attacker", "email": "test@test.com", "password": "Password123"})
assert res.status_code == 403, f"Expected 403, got {res.status_code}: {res.text}"
print(f"-> SUCCESS: Public registration blocked (HTTP 403: {res.json()['detail']})")

# 6. Admin accessing protected dashboard data
print("\n[Test 6] Admin accessing /api/admin/complaints with admin token...")
res = client.get("/api/admin/complaints", headers={"Authorization": f"Bearer {admin_token}"})
assert res.status_code == 200, f"Admin access failed: {res.text}"
print(f"-> SUCCESS: Admin loaded complaint queue ({res.json()['total']} complaints total)")

print("\n==================================================================")
print("         ALL ADMIN-ONLY SECURITY CHECKS PASSED WITH 100%!         ")
print("==================================================================")

