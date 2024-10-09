import os
import gradio as gr
from PIL import Image, ImageOps, ImageChops, ImageFilter
import numpy as np
import cv2  # Added import for OpenCV
from modules import scripts, script_callbacks, processing, shared
from modules.processing import process_images, StableDiffusionProcessingImg2Img

class OverlayPNG(scripts.Script):
    def title(self):
        return "Overlay"

    # Show in Img2Img only
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        if not is_img2img:
            return None
        
        with gr.Accordion("Overlay", open=False):
            with gr.Group():
                # Enable Overlay checkbox
                enable = gr.Checkbox(label="Enable Overlay", value=False)

                # Ensure the image is handled as an RGBA image
                overlay_image = gr.Image(label="Overlay PNG", type="pil", image_mode="RGBA", tool="editor", elem_id="overlay_img")

                # Add sliders for Overlay Strength and Mask Strength on the same line
                with gr.Row():
                    overlay_strength = gr.Slider(label="Overlay Strength", minimum=0, maximum=1, step=0.01, value=1)
                    mask_strength = gr.Slider(label="Mask Strength", minimum=0, maximum=1, step=0.01, value=1)

                # Add checkboxes for Apply in Process and Apply in Postprocess
                with gr.Row():
                    apply_proc = gr.Checkbox(label="Apply in Process", value=False)
                    apply_postprocess = gr.Checkbox(label="Apply in Postprocess", value=False)

                # Add dropdown for overlay/mask mode
                overlay_mode = gr.Dropdown(label="Overlay Mode", choices=["Image + Mask", "Image Only", "Mask Only"], value="Image + Mask")

                # Add Hard Mask checkbox
                hard_mask = gr.Checkbox(label="Hard Mask", value=False)

                # Add Mask Erode/Dilate slider and Blur slider on the same row
                with gr.Row():
                    mask_erode_dilate = gr.Slider(label="Mask Erode/Dilate", minimum=-100, maximum=100, step=1, value=0)
                    blur_amount = gr.Slider(label="Blur", minimum=0, maximum=100, step=1, value=0)

        return [enable, overlay_image, overlay_strength, mask_strength, apply_proc, apply_postprocess, overlay_mode, hard_mask, mask_erode_dilate, blur_amount]

    def process(self, p, enable=False, overlay_image=None, overlay_strength=1.0, mask_strength=1.0, apply_proc=False, apply_postprocess=False, overlay_mode="Image + Mask", hard_mask=False, mask_erode_dilate=0, blur_amount=0):
        # Return early if the overlay is not enabled or there's no overlay image
        if not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        if not apply_proc:
            return
        if not enable or overlay_image is None or not p.init_images:
            return  # No overlay or no initial images

        try:
            # The overlay_image is now directly in RGBA mode
            overlay = overlay_image

            # Extract the alpha channel from the overlay image
            overlay_alpha = np.array(overlay.split()[3])  # Extract the alpha channel as a NumPy array

            # Normalize the alpha channel to create a mask (0-255 range for transparency)
            mask = np.clip(overlay_alpha, 0, 255).astype(np.uint8)

            # Apply Erode/Dilate if mask_erode_dilate is not zero
            if mask_erode_dilate != 0:
                kernel_size = int(abs(mask_erode_dilate))
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure the kernel size is odd
                if kernel_size < 3:
                    kernel_size = 3  # Minimum kernel size

                kernel = np.ones((kernel_size, kernel_size), np.uint8)

                if mask_erode_dilate > 0:
                    # Dilate the mask
                    mask = cv2.dilate(mask, kernel, iterations=1)
                else:
                    # Erode the mask
                    mask = cv2.erode(mask, kernel, iterations=1)
                    
            # Apply Hard Mask if enabled
            if hard_mask:
                mask[mask != 0] = 255  # Set all non-opaque pixels to transparent

            # Convert the NumPy mask back to an Image object (grayscale 'L' mode)
            mask_image = Image.fromarray(mask, mode="L")

            # Handle different modes from the dropdown
            if overlay_mode == "Mask Only":
                print("Mask Only mode: Skipping overlay of the image.")
            elif overlay_mode == "Image Only":
                # Process p.init_images using the mask but do not apply the mask to p.image_mask
                for i in range(len(p.init_images)):
                    # Convert the current init image to RGBA to handle transparency
                    base = p.init_images[i].convert("RGBA")

                    overlay_resized = overlay.resize(base.size, Image.ANTIALIAS)
                    mask_image_resized = mask_image.resize(base.size, Image.ANTIALIAS)

                    # Apply Gaussian Blur if blur_amount > 0
                    if blur_amount > 0:
                        overlay_resized = overlay_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
                        mask_image_resized = mask_image_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
                        # Ensure mask_image_resized is in 'L' mode after blur
                        if mask_image_resized.mode != 'L':
                            mask_image_resized = mask_image_resized.convert('L')

                    # Adjust mask_image_resized by overlay_strength
                    mask_np = np.array(mask_image_resized, dtype=np.float32)
                    mask_np *= overlay_strength
                    mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
                    mask_image_resized = Image.fromarray(mask_np, mode='L')

                    # Apply the overlay at the specified position using the mask for transparency
                    position = (0, 0)
                    base.paste(overlay_resized, position, mask=mask_image_resized)  # Use mask here

                    # Convert back to RGB and update the init_images
                    p.init_images[i] = base.convert("RGB")
            else:  # "Image + Mask" mode
                # Process p.init_images (with mask)
                for i in range(len(p.init_images)):
                    # Convert the current init image to RGBA to handle transparency
                    base = p.init_images[i].convert("RGBA")

                    overlay_resized = overlay.resize(base.size, Image.ANTIALIAS)
                    mask_image_resized = mask_image.resize(base.size, Image.ANTIALIAS)

                    # Apply Gaussian Blur if blur_amount > 0
                    if blur_amount > 0:
                        overlay_resized = overlay_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
                        mask_image_resized = mask_image_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
                        # Ensure mask_image_resized is in 'L' mode after blur
                        if mask_image_resized.mode != 'L':
                            mask_image_resized = mask_image_resized.convert('L')
                            
                    # Ensure mask_image_resized is in 'L' mode before converting to NumPy array
                    mask_image_resized = mask_image_resized.convert('L')

                    # Adjust mask_image_resized by overlay_strength
                    mask_np = np.array(mask_image_resized, dtype=np.float32)
                    mask_np *= overlay_strength
                    mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
                    mask_image_resized = Image.fromarray(mask_np, mode='L')

                    # Apply the overlay at the specified position using the mask for transparency
                    position = (0, 0)
                    base.paste(overlay_resized, position, mask=mask_image_resized)

                    # Convert back to RGB and update the init_images
                    p.init_images[i] = base.convert("RGB")

            # Handle p.image_mask combination after processing p.init_images
            if overlay_mode != "Image Only":  # Skip if "Image Only" is selected
                if p.image_mask is not None:
                    print("Combining with p.image_mask...")

                    # Invert our mask (convert black to white and white to black)
                    inverted_mask = ImageOps.invert(mask_image)

                    # Resize inverted mask to match p.image_mask if necessary
                    if inverted_mask.size != p.image_mask.size:
                        inverted_mask = inverted_mask.resize(p.image_mask.size, Image.ANTIALIAS)

                    # Apply mask_strength: blend p.image_mask and our inverted mask based on mask_strength
                    combined_mask = ImageChops.blend(p.image_mask.convert("L"), inverted_mask, mask_strength)
                    
                    # Ensure that black pixels from p.image_mask remain black in combined_mask
                    # Convert p.image_mask and combined_mask to NumPy arrays
                    p_image_mask_L = p.image_mask.convert("L")
                    p_image_mask_array = np.array(p_image_mask_L)
                    combined_mask_array = np.array(combined_mask)
                    
                    # Set pixels to 0 in combined_mask_array where p_image_mask_array is 0 (black)
                    combined_mask_array[p_image_mask_array == 0] = 0
                    
                    # Convert the modified array back to an Image
                    combined_mask = Image.fromarray(combined_mask_array, mode='L')
                    
                    # Replace p.image_mask with the combined version
                    p.image_mask = combined_mask

                else:
                    # If p.image_mask is empty, assume a white background
                    white_background = Image.new("L", mask_image.size, 255)  # A fully white mask
                    p.image_mask = ImageChops.blend(white_background, ImageOps.invert(mask_image), mask_strength)

            return

        except Exception as e:
            print(f"Error in OverlayPNG extension: {e}")
            return

    def postprocess_image(self, p, processed, enable=False, overlay_image=None, overlay_strength=1.0, mask_strength=1.0, apply_proc=False, apply_postprocess=False, overlay_mode="Image + Mask", hard_mask=False, mask_erode_dilate=0, blur_amount=0):
        if not isinstance(p, StableDiffusionProcessingImg2Img):
            return
        if not apply_postprocess:
            return
        if not enable or overlay_image is None:
            return  # No overlay or no initial images


        # The overlay_image is now directly in RGBA mode
        overlay = overlay_image

        # **New Code Start: Check for fp_x_start, fp_y_start, fp_x_end, fp_y_end**
        # Ensure that all four parameters exist and are not None
        if all(hasattr(p, attr) and getattr(p, attr) is not None for attr in ['fp_x_start', 'fp_y_start', 'fp_x_end', 'fp_y_end']):
            # Extract the cropping coordinates
            x_start = p.fp_x_start
            y_start = p.fp_y_start
            x_end = p.fp_x_end
            y_end = p.fp_y_end
            angle = p.fp_angle

            # Crop the overlay image using the provided coordinates
            overlay = overlay.crop((x_start, y_start, x_end, y_end))

            # Also crop the mask_image accordingly
            overlay_alpha = np.array(overlay.split()[3])  # Extract the alpha channel as a NumPy array

            # Normalize the alpha channel to create a mask (0-255 range for transparency)
            mask = np.clip(overlay_alpha, 0, 255).astype(np.uint8)

            # Convert the NumPy mask back to an Image object (grayscale 'L' mode)
            mask_image = Image.fromarray(mask, mode="L")

            if angle != 0:
                overlay = self.rotate_image(overlay, angle, True)
                mask_image = self.rotate_image(mask_image, angle)

            print(f"Cropped overlay and mask using coordinates: ({x_start}, {y_start}, {x_end}, {y_end})")
        else:
            # Extract the alpha channel from the overlay image
            overlay_alpha = np.array(overlay.split()[3])  # Extract the alpha channel as a NumPy array

            # Normalize the alpha channel to create a mask (0-255 range for transparency)
            mask = np.clip(overlay_alpha, 0, 255).astype(np.uint8)

            # Convert the NumPy mask back to an Image object (grayscale 'L' mode)
            mask_image = Image.fromarray(mask, mode="L")
        # **New Code End**

        # Resize overlay and mask to match the dimensions of the processed image
        base = processed.image.convert("RGBA")
        if overlay.size != base.size:
            overlay_resized = overlay.resize(base.size, Image.ANTIALIAS)
            mask_image_resized = mask_image.resize(base.size, Image.ANTIALIAS)
        else:
            overlay_resized = overlay
            mask_image_resized = mask_image

        # Apply Gaussian Blur if blur_amount > 0
        if blur_amount > 0:
            overlay_resized = overlay_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
            mask_image_resized = mask_image_resized.filter(ImageFilter.GaussianBlur(radius=blur_amount))
            # Ensure mask_image_resized is in 'L' mode after blur
            if mask_image_resized.mode != 'L':
                mask_image_resized = mask_image_resized.convert('L')

        # Ensure mask_image_resized is in 'L' mode before converting to NumPy array
        mask_image_resized = mask_image_resized.convert('L')

        # Adjust mask_image_resized by overlay_strength
        mask_np = np.array(mask_image_resized, dtype=np.float32)
        mask_np *= overlay_strength
        mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
        mask_image_resized = Image.fromarray(mask_np, mode='L')

        # Handle different modes from the dropdown
        if overlay_mode == "Mask Only":
            print("Mask Only mode: Skipping overlay of the image.")
        elif overlay_mode == "Image Only":
            # Apply the overlay at the specified position using the mask for transparency
            position = (0, 0)
            base.paste(overlay_resized, position, mask=mask_image_resized)  # Use mask here

            # Convert back to RGB and update processed.image
            processed.image = base.convert("RGB")
        else:  # "Image + Mask" mode
            # Apply the overlay at the specified position using the mask for transparency
            position = (0, 0)
            base.paste(overlay_resized, position, mask=mask_image_resized)

            # Convert back to RGB and update processed.image
            processed.image = base.convert("RGB")

        # Handle p.image_mask combination after processing if not in "Image Only" mode
        if overlay_mode != "Image Only":
            if p.image_mask is not None:
                print("Combining with p.image_mask...")

                # Invert our mask (convert black to white and white to black)
                inverted_mask = ImageOps.invert(mask_image_resized)

                # Resize inverted mask to match p.image_mask if necessary
                if inverted_mask.size != p.image_mask.size:
                    inverted_mask = inverted_mask.resize(p.image_mask.size, Image.ANTIALIAS)

                # Apply mask_strength: blend p.image_mask and our inverted mask based on mask_strength
                combined_mask = ImageChops.blend(p.image_mask.convert("L"), inverted_mask, mask_strength)
                
                # Ensure that black pixels from p.image_mask remain black in combined_mask
                # Convert p.image_mask and combined_mask to NumPy arrays
                p_image_mask_L = p.image_mask.convert("L")
                p_image_mask_array = np.array(p_image_mask_L)
                combined_mask_array = np.array(combined_mask)
                
                # Set pixels to 0 in combined_mask_array where p_image_mask_array is 0 (black)
                combined_mask_array[p_image_mask_array == 0] = 0
                
                # Convert the modified array back to an Image
                combined_mask = Image.fromarray(combined_mask_array, mode='L')
                
                # Replace p.image_mask with the combined version
                p.image_mask = combined_mask

            else:
                # If p.image_mask is empty, assume a white background
                white_background = Image.new("L", mask_image_resized.size, 255)  # A fully white mask

                # Invert our resized mask
                inverted_mask = ImageOps.invert(mask_image_resized)
            
                # Apply mask_strength: blend white background and inverted mask
                p.image_mask = ImageChops.blend(white_background, inverted_mask, mask_strength)

        return

                
    def rotate_image(self, image, angle, use_alpha=False):
        """
        Rotates an image (NumPy array or PIL Image) by the given angle using Pillow.
        If the input is a NumPy array, it will be converted to a PIL Image first.
    
        Parameters:
        - image: The input image (NumPy array or PIL Image).
        - angle: The angle by which to rotate the image.
        - use_alpha: If True, the image will maintain transparency (RGBA).
                     If False, the image will be converted to RGB with no alpha.
        
        Returns:
        - The rotated image with transparent (alpha) borders if use_alpha is True,
          or with a white background if use_alpha is False.
        """
        # Check if the image is a NumPy array and convert it to a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if use_alpha:
            # Ensure the image is in RGBA mode for transparency support
            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            # Rotate with transparent background (0, 0, 0, 0) for alpha support
            rotated_image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0, 0))  # Transparent fill
        else:
            # Ensure the image is in RGB mode for non-alpha support
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Rotate with white background (255, 255, 255) for no transparency
            rotated_image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255))  # White fill

        # Return the rotated image
        return rotated_image  # Return as PIL Image

### EOF