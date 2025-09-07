from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import torch
import os   
import time
from datetime import datetime
from typing import Optional, Tuple

class ImageGenerator:
    def __init__(self, model_id: str = "dreamlike-art/dreamlike-diffusion-1.0"):
        self.setup_device()
        self.setup_pipeline(model_id)
        self.output_dir = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)

    def setup_device(self):
        """Setup CUDA device if available"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Using device: {self.device}")

    def setup_pipeline(self, model_id: str):
        """Initialize the Stable Diffusion pipeline"""
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                use_auth_token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
            )
            self.pipe.enable_attention_slicing()
            self.pipe = self.pipe.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_image(
        self,
        prompt: str,
        num_steps: int = 20,
        size: Tuple[int, int] = (512, 512),
        guidance_scale: float = 7.5
    ) -> Optional[str]:
        """Generate image from prompt and save it"""
        try:
            print(f"\nGenerating image for prompt: '{prompt}'")
            print(f"Steps: {num_steps}, Size: {size}")
            
            start_time = time.time()
            
            image = self.pipe(
                prompt,
                num_inference_steps=num_steps,
                height=size[1],
                width=size[0],
                guidance_scale=guidance_scale
            ).images[0]
            
            elapsed = time.time() - start_time
            print(f"Generation time: {elapsed:.2f} seconds")
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{prompt[:30]}.png".replace(" ", "_")
            filepath = os.path.join(self.output_dir, filename)
            
            image.save(filepath)
            print(f"Saved image to: {filepath}")
            
            # Display image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.axis("off")
            plt.show()
            
            return filepath
        
        except Exception as e:
            print(f"Error generating image: {e}")
            return None

def main():
    generator = ImageGenerator()
    
    while True:
        print("\n=== Text to Image Generator ===")
        print("1. Generate image")
        print("2. Change settings")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            prompt = input("\nEnter your prompt: ")
            if prompt.strip():
                generator.generate_image(prompt)
            else:
                print("Prompt cannot be empty!")
                
        elif choice == "2":
            try:
                steps = int(input("Enter number of steps (10-50): "))
                width = int(input("Enter image width (256-1024): "))
                height = int(input("Enter image height (256-1024): "))
                guidance = float(input("Enter guidance scale (1-20): "))
                
                prompt = input("\nEnter your prompt: ")
                if prompt.strip():
                    generator.generate_image(
                        prompt,
                        num_steps=steps,
                        size=(width, height),
                        guidance_scale=guidance
                    )
            except ValueError:
                print("Invalid input! Using default values.")
                
        elif choice == "3":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please try again.")

if __name__ == "__main__":
    main()