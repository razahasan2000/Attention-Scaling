"""
Verification script for attention scaling verification suite.

Checks that all required artifacts exist and are consistent.
"""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def verify_artifacts(artifacts_dir: str = None) -> dict:
    """
    Verify that all required artifacts exist and are consistent.
    
    Args:
        artifacts_dir: Path to artifacts directory
        
    Returns:
        Dictionary with verification results
    """
    if artifacts_dir is None:
        artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    
    results = {
        'artifacts_dir': artifacts_dir,
        'exists': os.path.exists(artifacts_dir),
        'checks': [],
        'passed': 0,
        'failed': 0,
        'warnings': [],
    }
    
    # Check key directories
    required_dirs = [
        'code',
        'preregistered_plan',
        'run_records',
        'statistics',
        'figures',
        'tables',
        'manuscript',
        'claim_strengthen',
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(artifacts_dir, dir_name)
        exists = os.path.exists(dir_path)
        results['checks'].append({
            'check': f'directory_exists_{dir_name}',
            'path': dir_path,
            'exists': exists,
            'passed': exists,
        })
        if exists:
            results['passed'] += 1
        else:
            results['failed'] += 1
            results['warnings'].append(f'Missing directory: {dir_path}')
    
    # Check key files
    required_files = [
        'claim_strengthen/summary_claims_e1e4.json',
        'paper_imrad/paper.tex',
        'paper_imrad/paper.pdf',
        'paper_imrad/references.bib',
        'paper_imrad/figures/',
    ]
    
    for file_path in required_files:
        full_path = os.path.join(artifacts_dir, file_path)
        exists = os.path.exists(full_path)
        results['checks'].append({
            'check': f'file_exists_{file_path}',
            'path': full_path,
            'exists': exists,
            'passed': exists,
        })
        if exists:
            results['passed'] += 1
        else:
            results['failed'] += 1
            results['warnings'].append(f'Missing file: {full_path}')
    
    # Check summary claims JSON
    claims_path = os.path.join(artifacts_dir, 'claim_strengthen', 'summary_claims_e1e4.json')
    if os.path.exists(claims_path):
        with open(claims_path, 'r') as f:
            claims = json.load(f)
        results['claims'] = claims
        
        # Check claim flags
        claim_flags = claims.get('claim_flags', {})
        for claim_name, claim_value in claim_flags.items():
            results['checks'].append({
                'check': f'claim_{claim_name}',
                'claim': claim_name,
                'value': claim_value,
                'passed': claim_value in ['pass', 'fail', 'partial'],
            })
    
    # Calculate totals
    results['total_checks'] = len(results['checks'])
    results['all_passed'] = results['failed'] == 0
    
    return results


def print_verification_report(results: dict):
    """Print verification report to console."""
    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"Artifacts directory: {results['artifacts_dir']}")
    print(f"Directory exists: {results['exists']}")
    print(f"Total checks: {results['total_checks']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"All passed: {results['all_passed']}")
    print()
    
    if results['warnings']:
        print("WARNINGS:")
        for warning in results['warnings']:
            print(f"  - {warning}")
        print()
    
    if results.get('claims'):
        print("CLAIM FLAGS:")
        for claim_name, claim_value in results['claims'].get('claim_flags', {}).items():
            print(f"  {claim_name}: {claim_value}")
        print()
    
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify artifact pack')
    parser.add_argument('--artifacts-dir', type=str, default=None,
                        help='Path to artifacts directory')
    args = parser.parse_args()
    
    results = verify_artifacts(args.artifacts_dir)
    print_verification_report(results)
    
    # Exit with error code if verification failed
    if not results['all_passed']:
        sys.exit(1)
    else:
        print("\nAll verification checks passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()