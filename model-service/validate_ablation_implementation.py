#!/usr/bin/env python
"""
Quick validation script for modality ablation study implementation.

Validates that all components are correctly implemented and integrated.
"""

import sys
from pathlib import Path

def validate_files():
    """Check that all required files exist."""
    print("📋 Validating file structure...")
    
    required_files = {
        'config/config.yaml': 'Configuration file',
        'src/models/multimodal_model.py': 'Multimodal model (modified)',
        'src/train/multimodal_train.py': 'Training script (modified)',
        'src/eval/ablation_study.py': 'Ablation study evaluation (NEW)',
        'tests/test_modality_ablation.py': 'Unit tests (NEW)',
        'MODALITY_ABLATION_GUIDE.md': 'User guide (NEW)',
        'sample_ablation_results.json': 'Example results (NEW)',
        'MODALITY_ABLATION_IMPLEMENTATION_SUMMARY.md': 'Implementation summary (NEW)',
    }
    
    all_exist = True
    for file_path, description in required_files.items():
        full_path = Path(file_path)
        if full_path.exists():
            print(f"  ✅ {file_path:<40} - {description}")
        else:
            print(f"  ❌ {file_path:<40} - MISSING")
            all_exist = False
    
    return all_exist


def validate_config():
    """Check config has modality flags."""
    print("\n⚙️  Validating configuration...")
    
    try:
        import yaml
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        model_cfg = config.get('model', {})
        enable_audio = model_cfg.get('enable_audio')
        enable_video = model_cfg.get('enable_video')
        
        if enable_audio is not None and enable_video is not None:
            print(f"  ✅ enable_audio: {enable_audio}")
            print(f"  ✅ enable_video: {enable_video}")
            return True
        else:
            print(f"  ❌ Modality flags not found in config")
            return False
    except Exception as e:
        print(f"  ❌ Error loading config: {e}")
        return False


def validate_model():
    """Check MultimodalModel has modality support."""
    print("\n🧠 Validating model...")
    
    try:
        # Check that imports work
        from src.models.multimodal_model import MultimodalModel
        import inspect
        
        # Check constructor signature
        sig = inspect.signature(MultimodalModel.__init__)
        params = list(sig.parameters.keys())
        
        if 'enable_video' in params and 'enable_audio' in params:
            print(f"  ✅ Constructor has enable_video parameter")
            print(f"  ✅ Constructor has enable_audio parameter")
        else:
            print(f"  ❌ Constructor missing modality parameters")
            return False
        
        # Check forward method
        forward_sig = inspect.signature(MultimodalModel.forward)
        forward_params = list(forward_sig.parameters.keys())
        
        if 'video' in forward_params and 'audio' in forward_params:
            print(f"  ✅ forward() accepts video parameter")
            print(f"  ✅ forward() accepts audio parameter")
        else:
            print(f"  ❌ forward() missing parameters")
            return False
        
        # Check extract_features method
        if hasattr(MultimodalModel, 'extract_features'):
            print(f"  ✅ extract_features() method exists")
        else:
            print(f"  ❌ extract_features() method missing")
            return False
        
        # Verify validation logic exists
        source = inspect.getsource(MultimodalModel.__init__)
        if 'at least one modality must be enabled' in source:
            print(f"  ✅ Modality validation logic present")
        else:
            print(f"  ⚠️  Modality validation logic may be missing")
        
        return True
    except Exception as e:
        print(f"  ❌ Model validation error: {e}")
        return False


def validate_training():
    """Check training script has modality integration."""
    print("\n🚂 Validating training script...")
    
    try:
        with open('src/train/multimodal_train.py', 'r') as f:
            content = f.read()
        
        checks = {
            'enable_audio': 'Config reads enable_audio flag',
            'enable_video': 'Config reads enable_video flag',
            'MultimodalModel(': 'Creates MultimodalModel instance',
            'enable_video=enable_video': 'Passes enable_video to model',
            'enable_audio=enable_audio': 'Passes enable_audio to model',
        }
        
        all_ok = True
        for pattern, description in checks.items():
            if pattern in content:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ {description}")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"  ❌ Training validation error: {e}")
        return False


def validate_evaluation():
    """Check ablation study exists."""
    print("\n📊 Validating evaluation...")
    
    try:
        from src.eval.ablation_study import AblationStudy
        import inspect
        
        methods = [m for m, _ in inspect.getmembers(AblationStudy, predicate=inspect.isfunction)]
        
        required_methods = [
            'run',
            '_evaluate_config',
            '_load_model',
            '_evaluate_split',
            '_compile_report',
            '_generate_comparison',
            '_generate_analysis',
        ]
        
        all_ok = True
        for method in required_methods:
            if method in methods:
                print(f"  ✅ {method}() implemented")
            else:
                print(f"  ❌ {method}() missing")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"  ❌ Evaluation validation error: {e}")
        return False


def validate_tests():
    """Check tests exist."""
    print("\n🧪 Validating tests...")
    
    try:
        with open('tests/test_modality_ablation.py', 'r') as f:
            content = f.read()
        
        test_classes = [
            'TestModalityConfiguration',
            'TestMultimodalModelInstantiation',
            'TestForwardPassWithModalities',
            'TestExtractFeaturesWithModalities',
            'TestModalityFusionDimensions',
            'TestTrainerModalitySupport',
            'TestAblationConfiguration',
        ]
        
        all_ok = True
        for test_class in test_classes:
            if f'class {test_class}' in content:
                print(f"  ✅ {test_class} implemented")
            else:
                print(f"  ❌ {test_class} missing")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"  ❌ Test validation error: {e}")
        return False


def validate_documentation():
    """Check documentation exists."""
    print("\n📚 Validating documentation...")
    
    docs = {
        'MODALITY_ABLATION_GUIDE.md': 'User guide',
        'MODALITY_ABLATION_IMPLEMENTATION_SUMMARY.md': 'Implementation summary',
        'sample_ablation_results.json': 'Example results',
    }
    
    all_ok = True
    for doc_file, description in docs.items():
        path = Path(doc_file)
        if path.exists() and path.stat().st_size > 100:
            print(f"  ✅ {doc_file:<45} ({description})")
        else:
            print(f"  ❌ {doc_file:<45} (MISSING OR EMPTY)")
            all_ok = False
    
    return all_ok


def main():
    """Run all validations."""
    print("=" * 70)
    print("[CHECK] MODALITY ABLATION STUDY - VALIDATION SUITE")
    print("=" * 70)
    
    results = {
        'Files': validate_files(),
        'Configuration': validate_config(),
        'Model': validate_model(),
        'Training': validate_training(),
        'Evaluation': validate_evaluation(),
        'Tests': validate_tests(),
        'Documentation': validate_documentation(),
    }
    
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name:<30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("\nNext steps:")
        print("  1. Run unit tests: pytest tests/test_modality_ablation.py -v")
        print("  2. Read guide: MODALITY_ABLATION_GUIDE.md")
        print("  3. Configure: Edit config/config.yaml enable_audio/enable_video")
        print("  4. Train: python -m src.train.multimodal_train --data-root <path>")
        print("  5. Evaluate: python -m src.eval.ablation_study --checkpoint <path>")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("\nPlease check the errors above and correct them.")
        sys.exit(1)
    
    print("=" * 70)


if __name__ == '__main__':
    main()
