"""
One-off helper to create a login user for local testing.
Usage: python seed_user.py <username> <password> <doctor|medical_student>
"""
import sys

from auth import create_user

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python seed_user.py <username> <password> <doctor|medical_student>")
        sys.exit(1)
    _, username, password, role = sys.argv
    create_user(username, password, role)
    print(f"Created user '{username}' with role '{role}'.")
