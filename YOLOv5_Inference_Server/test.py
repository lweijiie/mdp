import os

file_path = r"C:\Users\DELL G15\Desktop\SCHOOL\MDP\mdp\Weights\best_yolo5_2024_10_18.pt"

# Verify if the file exists
if os.path.exists(file_path):
    print("✅ File found:", file_path)
else:
    print("❌ ERROR: File NOT found at:", file_path)
