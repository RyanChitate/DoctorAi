import json

feedback_json = '''
[
    {"image_path": "bone 2.jpg", "correct_label": "Unhealthy", "correct_body_part": "leg", "body_disease": "Bone"},
    {"image_path": "bone 3.jpg", "correct_label": "Healthy", "correct_body_part": "arm", "body_disease": "Bone"},
    {"image_path": "bone.jpg", "correct_label": "Healthy", "correct_body_part": "leg", "body_disease": "Bone"},
    {"image_path": "brain 2.jpg", "correct_label": "Healthy", "correct_body_part": "brain", "body_disease": "Brain"},
    {"image_path": "brain 3.jpg", "correct_label": "Healthy", "correct_body_part": "brain", "body_disease": "Brain"},
    {"image_path": "brain.jpg", "correct_label": "Healthy", "correct_body_part": "brain", "body_disease": "Brain"},
    {"image_path": "liver 2.jpg", "correct_label": "Healthy", "correct_body_part": "liver", "body_disease": "Liver"},
    {"image_path": "liver 2.jpg", "correct_label": "Unhealthy", "correct_body_part": "liver", "body_disease": "Liver"},
    {"image_path": "liver 2.jpg", "correct_label": "Unhealthy", "correct_body_part": "liver", "body_disease": "Liver"},
    {"image_path": "liver.jpg", "correct_label": "Unhealthy", "correct_body_part": "liver", "body_disease": "Liver"},
    {"image_path": "spine 2.jpg", "correct_label": "Unhealthy", "correct_body_part": "spine", "body_disease": "Spinal"},
    {"image_path": "spine 3.jpg", "correct_label": "Healthy", "correct_body_part": "spine", "body_disease": "Spinal"},
    {"image_path": "spine 4.jpg", "correct_label": "Healthy", "correct_body_part": "spine", "body_disease": "Spinal"},
    {"image_path": "spine.jpg", "correct_label": "Unhealthy", "correct_body_part": "spine", "body_disease": "Spinal"}
]
'''

try:
    feedback_data = json.loads(feedback_json)
    print("JSON is valid.")
    for entry in feedback_data:
        print(entry)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
