import platform
import launch

# Check for TorchVision package
if not launch.is_installed("torchvision"):
    print("[FacePop Debug] TorchVision is not installed. Installing...")
    launch.run_pip("install torchvision", "requirements for FacePop (TorchVision)")

# Check for OpenCV package
if not launch.is_installed("opencv-python"):
    print("[FacePop Debug] OpenCV is not installed. Installing...")
    launch.run_pip("install opencv-python", "requirements for FacePop (OpenCV)")
    



