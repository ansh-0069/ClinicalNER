"""
verify_project.py
─────────────────
Verification script to ensure all clinical trial features are working.

Run this before submitting your job application to verify everything is ready.
"""

import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_module_imports():
    """Check if all modules can be imported."""
    print("\n🔍 Checking Module Imports...")
    
    modules = [
        ("src.utils.data_loader", "DataLoader"),
        ("src.utils.eda", "ClinicalEDA"),
        ("src.pipeline.ner_pipeline", "NERPipeline"),
        ("src.pipeline.data_cleaner", "DataCleaner"),
        ("src.pipeline.audit_logger", "AuditLogger"),
        ("src.pipeline.data_quality_validator", "DataQualityValidator"),
        ("src.reports.clinical_listings", "ClinicalReportGenerator"),
        ("src.utils.sql_queries", "QUERY_CATALOG"),
    ]
    
    all_passed = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_passed = False
    
    return all_passed

def check_documentation():
    """Check if all documentation files exist."""
    print("\n📚 Checking Documentation...")
    
    docs = [
        ("README.md", "Project README"),
        ("CLINICAL_USE_CASES.md", "Clinical Use Cases"),
        ("COMPLIANCE.md", "Regulatory Compliance"),
        ("AWS_DEPLOYMENT.md", "AWS Deployment Guide"),
        ("JOB_APPLICATION_SUMMARY.md", "Job Application Summary"),
        ("PROJECT_COMPLETE.md", "Project Completion Guide"),
        ("STRUCTURE.md", "Project Structure"),
    ]
    
    all_passed = True
    for filepath, description in docs:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    return all_passed

def check_deployment_files():
    """Check if deployment files exist."""
    print("\n🚀 Checking Deployment Files...")
    
    files = [
        ("docker/Dockerfile", "Docker Configuration"),
        ("docker-compose.yml", "Docker Compose"),
        ("docker-compose.aws.yml", "AWS Docker Compose"),
        ("terraform/main.tf", "Terraform Infrastructure"),
        ("requirements.txt", "Python Dependencies"),
    ]
    
    all_passed = True
    for filepath, description in files:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    return all_passed

def check_test_files():
    """Check if test files exist."""
    print("\n🧪 Checking Test Files...")
    
    test_dir = Path("tests")
    if not test_dir.exists():
        print("❌ tests/ directory not found")
        return False
    
    test_files = list(test_dir.glob("test_*.py"))
    if test_files:
        print(f"✅ Found {len(test_files)} test files")
        for test_file in test_files:
            print(f"   - {test_file}")
        return True
    else:
        print("❌ No test files found")
        return False

def check_runner_scripts():
    """Check if phase runner scripts exist."""
    print("\n▶️  Checking Runner Scripts...")
    
    scripts = [
        ("run_phase1.py", "Phase 1: Data Ingestion + EDA"),
        ("run_phase2.py", "Phase 2: NER Pipeline"),
        ("run_phase3.py", "Phase 3: Quality + Audit"),
        ("run_phase4.py", "Phase 4: Flask API"),
    ]
    
    all_passed = True
    for filepath, description in scripts:
        if not check_file_exists(filepath, description):
            all_passed = False
    
    return all_passed

def verify_sql_queries():
    """Verify SQL queries are defined."""
    print("\n💾 Checking SQL Queries...")
    
    try:
        from src.utils.sql_queries import QUERY_CATALOG
        
        expected_queries = [
            'study_summary',
            'quality_metrics',
            'high_risk_notes',
            'regulatory_readiness',
            'phi_by_specialty',
        ]
        
        all_passed = True
        for query_name in expected_queries:
            if query_name in QUERY_CATALOG:
                print(f"✅ {query_name}")
            else:
                print(f"❌ {query_name} NOT FOUND")
                all_passed = False
        
        return all_passed
    except Exception as e:
        print(f"❌ Error loading SQL queries: {e}")
        return False

def main():
    """Run all verification checks."""
    print("="*60)
    print("  ClinicalNER Project Verification")
    print("  Checking all clinical trial features...")
    print("="*60)
    
    results = {
        "Module Imports": check_module_imports(),
        "Documentation": check_documentation(),
        "Deployment Files": check_deployment_files(),
        "Test Files": check_test_files(),
        "Runner Scripts": check_runner_scripts(),
        "SQL Queries": verify_sql_queries(),
    }
    
    print("\n" + "="*60)
    print("  Verification Summary")
    print("="*60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Your project is ready for submission.")
        print("\nNext steps:")
        print("1. Run: python run_phase1.py")
        print("2. Run: python run_phase2.py")
        print("3. Run: python run_phase3.py")
        print("4. Run: python run_phase4.py")
        print("5. Run: pytest tests/ -v")
        print("6. Review: JOB_APPLICATION_SUMMARY.md")
        print("\n✅ Ready to apply for Associate Clinical Programmer role!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
