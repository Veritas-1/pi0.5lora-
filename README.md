## pi0.5lora-

## 🎥 Demos

### Piper Single-Arm Red-on-Green Block Stacking

**Policy:** pi0.5 lora 
**Robot:** Piper Single Arm  
**Dataset:** 60 Episodes  
**Observation:** Head Camera + Left Wrist Camera  
**Action Space:** 7D (6 Joints + 1 Gripper)

### ✅ Successful Cases

| Demo | Description | Success | Demo | Description | Success |
|:---:|:---|:---:|:---:|:---|:---:|
| [![Success 1](GIF_SUCCESS_1_URL)](https://github.com/user-attachments/assets/3bb0e602-279a-4649-b31e-0c9269b26b7a) | Successful red-on-green stacking | True | [![Success 2](GIF_SUCCESS_2_URL)](https://github.com/user-attachments/assets/25357222-19a9-4888-bb79-ba36ac393458) | Successful red-on-green stacking | True |

### ❌ Failure Cases

| Demo | Description | Success | Demo | Description | Success |
|:---:|:---|:---:|:---:|:---|:---:|
| [![Failure 1](GIF_FAILURE_1_URL)](https://github.com/user-attachments/assets/daff4f67-0c4e-4b30-a7dd-dc2d0c8dc9e5) |先夹了蓝色方块，再将红色置于蓝色之上 ，未判定成功，之后也不知道干嘛（任务语义理解呢？）| False | [![Failure 2](GIF_FAILURE_2_URL)](https://github.com/user-attachments/assets/bd7921a5-8bec-4a58-9257-96dc227b179a) | 绿色放置成功，之后夹红色方块，未夹住，之后也没有张开夹臂 | False |

### Failure Case 3

| Demo | Description | Success |
|:---:|:---|:---:|
| [![Failure 3](GIF_FAILURE_3_URL)](https://github.com/user-attachments/assets/d979b49f-1261-405f-b8d2-354eeba783ee) | 准备夹绿色，不知为何，又偏移了 | False |

---










