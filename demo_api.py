#!/usr/bin/env python3
"""
Demo script for the Academic Multimodal LLM Experiment System API.

This script demonstrates how to use the REST API endpoints for image/video generation,
model management, and experiment tracking.
"""

import requests
import time
import json
from typing import Dict, Any


class APIDemo:
    """Demo client for the Academic Multimodal LLM API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        print("🔍 Checking API health...")
        response = requests.get(f"{self.base_url}/health")
        result = response.json()
        print(f"✅ API Status: {result['status']}")
        return result
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and VRAM information."""
        print("\n📊 Getting model status...")
        response = requests.get(f"{self.base_url}/models/status")
        
        if response.status_code == 200:
            result = response.json()
            print(f"🖥️  Hardware: {result['hardware_info']['gpu_model']}")
            print(f"💾 VRAM: {result['vram_info']['used_mb']}MB / {result['vram_info']['total_mb']}MB")
            print(f"🎨 Current Image Model: {result['current_models']['image']}")
            print(f"🎬 Current Video Model: {result['current_models']['video']}")
            return result
        else:
            print(f"❌ Error getting model status: {response.status_code}")
            return {}
    
    def list_available_models(self, pipeline_type: str = "image") -> Dict[str, Any]:
        """List available models for a pipeline type."""
        print(f"\n📋 Listing available {pipeline_type} models...")
        response = requests.get(f"{self.base_url}/models/list/{pipeline_type}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"Available {pipeline_type} models:")
            for model in models:
                status = "🟢 Loaded" if model['is_loaded'] else "⚪ Available"
                print(f"  {status} {model['name']} (VRAM: {model['vram_requirement_mb']}MB)")
            return models
        else:
            print(f"❌ Error listing models: {response.status_code}")
            return []
    
    def generate_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        print(f"\n🎨 Generating image: '{prompt[:50]}...'")
        
        request_data = {
            "prompt": prompt,
            "compliance_mode": "research_safe",
            **kwargs
        }
        
        response = requests.post(f"{self.base_url}/generate/image", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result['task_id']
            print(f"📝 Task ID: {task_id}")
            print(f"⏱️  Estimated time: {result.get('estimated_time', 'Unknown')}s")
            
            # Poll for completion
            return self._wait_for_task(task_id)
        else:
            print(f"❌ Error starting image generation: {response.status_code}")
            return {}
    
    def generate_video(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a video from a text prompt."""
        print(f"\n🎬 Generating video: '{prompt[:50]}...'")
        
        request_data = {
            "prompt": prompt,
            "compliance_mode": "research_safe",
            **kwargs
        }
        
        response = requests.post(f"{self.base_url}/generate/video", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result['task_id']
            print(f"📝 Task ID: {task_id}")
            print(f"⏱️  Estimated time: {result.get('estimated_time', 'Unknown')}s")
            
            # Poll for completion
            return self._wait_for_task(task_id)
        else:
            print(f"❌ Error starting video generation: {response.status_code}")
            return {}
    
    def _wait_for_task(self, task_id: str, max_wait: int = 300) -> Dict[str, Any]:
        """Wait for a task to complete and return the result."""
        print("⏳ Waiting for task to complete...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = requests.get(f"{self.base_url}/tasks/{task_id}")
            
            if response.status_code == 200:
                status = response.json()
                
                if status['status'] == 'completed':
                    result = status.get('result', {})
                    if result.get('success'):
                        print(f"✅ Generation completed in {result['generation_time']:.1f}s")
                        print(f"📁 Output: {result.get('output_path', 'N/A')}")
                        print(f"🤖 Model used: {result['model_used']}")
                        return result
                    else:
                        print(f"❌ Generation failed: {result.get('error_message', 'Unknown error')}")
                        return result
                
                elif status['status'] == 'failed':
                    print(f"❌ Task failed: {status.get('error', 'Unknown error')}")
                    return status
                
                elif status['status'] == 'processing':
                    progress = status.get('progress', 0)
                    print(f"🔄 Processing... {progress:.1f}%")
                
                time.sleep(2)
            else:
                print(f"❌ Error checking task status: {response.status_code}")
                break
        
        print("⏰ Task timed out")
        return {}
    
    def switch_model(self, model_name: str, pipeline_type: str = "image") -> Dict[str, Any]:
        """Switch to a different model."""
        print(f"\n🔄 Switching {pipeline_type} model to {model_name}...")
        
        request_data = {
            "model_name": model_name,
            "pipeline_type": pipeline_type
        }
        
        response = requests.post(f"{self.base_url}/models/switch", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Model switched successfully in {result['switch_time_seconds']:.1f}s")
            print(f"💾 VRAM change: {result['vram_usage']['vram_change_mb']:+d}MB")
            return result
        else:
            print(f"❌ Error switching model: {response.status_code}")
            return {}
    
    def save_experiment(self, name: str, description: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Save experiment results."""
        print(f"\n💾 Saving experiment: {name}")
        
        request_data = {
            "experiment_name": name,
            "description": description,
            "tags": ["demo", "api-test"],
            "results": results,
            "compliance_mode": "research_safe"
        }
        
        response = requests.post(f"{self.base_url}/models/experiment/save", json=request_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Experiment saved: {result['experiment_id']}")
            return result
        else:
            print(f"❌ Error saving experiment: {response.status_code}")
            return {}


def main():
    """Run the API demonstration."""
    print("🚀 Academic Multimodal LLM Experiment System - API Demo")
    print("=" * 60)
    
    # Initialize demo client
    demo = APIDemo()
    
    try:
        # 1. Health check
        demo.health_check()
        
        # 2. Get system status
        status = demo.get_model_status()
        
        # 3. List available models
        image_models = demo.list_available_models("image")
        video_models = demo.list_available_models("video")
        
        # 4. Generate a simple image (this will likely fail without actual models)
        print("\n" + "=" * 60)
        print("🎨 IMAGE GENERATION DEMO")
        print("=" * 60)
        
        image_result = demo.generate_image(
            "A serene mountain landscape with a crystal clear lake",
            width=512,
            height=512,
            num_inference_steps=20
        )
        
        # 5. Generate a simple video (this will likely fail without actual models)
        print("\n" + "=" * 60)
        print("🎬 VIDEO GENERATION DEMO")
        print("=" * 60)
        
        video_result = demo.generate_video(
            "Ocean waves gently lapping on a sandy beach at sunset",
            width=512,
            height=512,
            num_frames=14,
            fps=7
        )
        
        # 6. Save experiment results
        print("\n" + "=" * 60)
        print("💾 EXPERIMENT TRACKING DEMO")
        print("=" * 60)
        
        experiment_results = {
            "image_generation": image_result,
            "video_generation": video_result,
            "demo_timestamp": time.time(),
            "api_version": "1.0.0"
        }
        
        demo.save_experiment(
            "API Demo Test",
            "Demonstration of API functionality with sample generations",
            experiment_results
        )
        
        print("\n✅ Demo completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("💡 Make sure to start the server first:")
        print("   python -c \"from src.api.server import run_server; run_server()\"")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")


if __name__ == "__main__":
    main()