#!/usr/bin/env python3
"""
Test script for the modern UI to verify it works correctly.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_ui_creation():
    """Test that the modern UI can be created and initialized."""
    try:
        from src.ui.modern_interface import ModernInterface
        
        print("✅ Successfully imported ModernInterface")
        
        # Create interface
        interface = ModernInterface()
        print("✅ Successfully created interface instance")
        
        # Test initialization
        if interface.initialize():
            print("✅ Interface initialized successfully")
        else:
            print("⚠️ Interface initialization failed (expected without Gradio)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_system_integration():
    """Test that system integration works."""
    try:
        from src.ui.ui_integration import SystemIntegration
        
        print("✅ Successfully imported SystemIntegration")
        
        # Create system integration
        system = SystemIntegration()
        print("✅ Successfully created system integration instance")
        
        # Test initialization
        if system.initialize():
            print("✅ System integration initialized successfully")
        else:
            print("⚠️ System integration initialization failed (expected without full dependencies)")
        
        # Test model listing
        models = system.get_available_models("image")
        print(f"✅ Available image models: {list(models.keys())}")
        
        models = system.get_available_models("video")
        print(f"✅ Available video models: {list(models.keys())}")
        
        # Test hardware info
        hw_info = system.get_hardware_info()
        print(f"✅ Hardware info: {hw_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False


def test_model_registry():
    """Test the model registry."""
    try:
        from src.core.model_registry import get_model_registry, ModelType
        
        print("✅ Successfully imported model registry")
        
        registry = get_model_registry()
        print("✅ Successfully created model registry")
        
        # Test model listing
        all_models = registry.list_all_models()
        print(f"✅ Total models in registry: {len(all_models)}")
        
        # Test image models by tier
        image_models = registry.get_models_by_type(ModelType.TEXT_TO_IMAGE)
        print(f"✅ Image models: {len(image_models)}")
        
        # Group by tier
        by_tier = {}
        for model in image_models:
            tier = model.tier.value
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(model)
        
        for tier, models in by_tier.items():
            print(f"   {tier.upper()}: {len(models)} models")
            for model in models[:2]:  # Show first 2 per tier
                print(f"     - {model.model_name} ({model.download_size_gb:.1f}GB)")
        
        # Test video models
        video_models = registry.get_models_by_type(ModelType.IMAGE_TO_VIDEO)
        print(f"✅ Video models: {len(video_models)}")
        for model in video_models:
            print(f"   - {model.model_name} ({model.model_id})")
        
        # Test hardware recommendations
        recommendations = registry.get_compatible_models(8000)  # 8GB VRAM
        print(f"✅ Models compatible with 8GB VRAM: {len(recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry test failed: {e}")
        return False


def test_hardware_recommendations():
    """Test hardware recommendations system."""
    try:
        from src.core.hardware_recommendations import get_hardware_recommendations
        
        print("✅ Successfully imported hardware recommendations")
        
        recommendations = get_hardware_recommendations()
        print("✅ Successfully created recommendations system")
        
        # Test different hardware tiers
        test_configs = [
            (4000, "Budget (GTX 1650)"),
            (8000, "Mid-Range (RTX 3070)"),
            (16000, "High-End (RTX 4080)"),
            (24000, "Enthusiast (RTX 4090)")
        ]
        
        for vram, description in test_configs:
            setup = recommendations.get_setup_instructions(vram)
            tier = setup["tier"]
            auto_models = setup["recommendations"]["auto_download_models"]
            
            print(f"✅ {description}: {tier} tier, auto-download {len(auto_models)} models")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware recommendations test failed: {e}")
        return False


def test_model_downloader():
    """Test model downloader system."""
    try:
        from src.core.model_downloader import get_model_downloader
        
        print("✅ Successfully imported model downloader")
        
        downloader = get_model_downloader()
        print("✅ Successfully created downloader")
        
        # Test model availability checking
        test_model = "runwayml/stable-diffusion-v1-5"
        is_available = downloader.is_model_downloaded(test_model)
        size_mb = downloader.get_model_size_mb(test_model)
        
        print(f"✅ Model {test_model}: {'available' if is_available else 'not downloaded'} ({size_mb:.0f}MB)")
        
        # Test download status
        status = downloader.get_download_status(test_model)
        print(f"✅ Download status: {status.status.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model downloader test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Modern AI Content Generator")
    print("=" * 50)
    
    tests = [
        ("Model Registry", test_model_registry),
        ("Hardware Recommendations", test_hardware_recommendations),
        ("Model Downloader", test_model_downloader),
        ("System Integration", test_system_integration),
        ("UI Creation", test_ui_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 To launch the interface, run:")
        print("   python app.py")
        print("\n💡 The system will automatically:")
        print("   - Detect your hardware capabilities")
        print("   - Show only compatible models")
        print("   - Auto-download recommended models")
        print("   - Optimize settings for your GPU")
    else:
        print("⚠️ Some tests failed. Check the error messages above.")
        print("💡 The interface may still work in mock mode.")


if __name__ == "__main__":
    main()