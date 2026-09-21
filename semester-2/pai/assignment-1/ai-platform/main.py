from role import *
from decorators import *
from model import *

def main():
    # Create users
    admin_user = User("Alice", AdminRole(), quota=500)
    standard_user = User("Bob", StandardRole(), quota=25) # Low quota for testing

    model = ImageModel("VisionPro-v2")
                    
    # TEST 1: Successful Call (Bob)
    set_current_user(standard_user)
    print(f"\n--- Test 1: {standard_user.get_username()} ---")
    print(model.classify("cat.png", image_size_kb=50))
  
    # TEST 2: Rate Limiting (Bob's 2nd call should fail)
    print(f"\n--- Test 2: {standard_user.get_username()} (Trigger Limit) ---")
    try:
        model.classify("bird.png") # 2nd call
    except Exception as e:
        print(f"Caught expected error!")
  
    # TEST 3: Admin action (Switch to Alice, then back to Bob)
    set_current_user(admin_user)
    print(f"\n--- Test 3: {admin_user.get_username()} (Admin Action) ---")
    # Admins can do things viewers can't (if we had a @require_permission("train") method)
    model.train("Bird")
    print("Trained a model to classify birds!")

    set_current_user(standard_user)
    try:
        model.train("Cat")
    except Exception as e:
        print(f"Caught expected error!")
   
    # TEST 4: Quota check
    set_current_user(standard_user)
    print(f"\n--- Test 4: {standard_user.get_username()} (Quota Check) ---")
    # Bob has used 15 (1st) + 20 (2nd) = 35? No, wait—he started with 25.
    # His 2nd call likely depleted him.
    try:
        model.classify("fish.png")
    except Exception as e:
        print(f"Caught expected error!")

if __name__ == "__main__":
    main()
