import json
import base64
import requests
import os
from typing import Dict, List, Any

API_URL = "http://localhost:8000/api/"  # Replace with your actual running URL

def encode_image(path: str) -> str:
    """Encode image to base64 string"""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"⚠️  Image file not found: {path}")
        return None
    except Exception as e:
        print(f"❌ Error encoding image {path}: {e}")
        return None

def check_api_health() -> bool:
    """Check if API is running and healthy"""
    try:
        health_response = requests.get(f"{API_URL.replace('/api/', '/health')}", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"🏥 API Health: {health_data.get('status', 'unknown')}")
            print(f"   Knowledge Base: {health_data.get('knowledge_base', 'unknown')}")
            print(f"   AIPipe API: {health_data.get('aipipe_api', 'unknown')}")
            return health_data.get('status') in ['healthy', 'degraded']
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

test_cases = [
    {
        "name": "Model clarification with image",
        "question": "The question asks to use gpt-3.5-turbo-0125 model but the ai-proxy provided by Anand sir only supports gpt-4o-mini. So should we just use gpt-4o-mini or use the OpenAI API for gpt3.5 turbo?",
        "image": "project-tds-virtual-ta-q1.webp",
        "expected_link": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
        "rubric": ["gpt-3.5-turbo-0125", "gpt-4o-mini", "use gpt-4o-mini"]
    },
    {
        "name": "GA4 scoring dashboard",
        "question": "If a student scores 10/10 on GA4 as well as a bonus, how would it appear on the dashboard?",
        "expected_link": "https://discourse.onlinedegree.iitm.ac.in/t/ga4-data-sourcing-discussion-thread-tds-jan-2025/165959",
        "rubric": ["dashboard", "110", "bonus"]
    },
    {
        "name": "Docker vs Podman preference",
        "question": "I know Docker but have not used Podman before. Should I use Docker for this course?",
        "expected_link": "https://tds.s-anand.net/#/docker",
        "rubric": ["Podman", "Docker is acceptable", "recommended"]
    },
    {
        "name": "Future exam date (should not know)",
        "question": "When is the TDS Sep 2025 end-term exam?",
        "rubric": ["don't have", "don't know", "not available", "no information"]
    },
    {
        "name": "Course evaluation structure",
        "question": "What are the different types of evaluations in TDS and their weightages?",
        "rubric": ["GA", "15%", "Project", "20%", "Final", "25%", "ROE"]
    }
]

def evaluate_rubric(answer: str, rubric: List[str]) -> Dict[str, bool]:
    """Evaluate answer against rubric criteria"""
    results = {}
    answer_lower = answer.lower()
    
    for criterion in rubric:
        criterion_lower = criterion.lower()
        passed = criterion_lower in answer_lower
        results[criterion] = passed
    
    return results

def run_test_case(case: Dict[str, Any], case_num: int) -> Dict[str, Any]:
    """Run a single test case and return results"""
    print("="*80)
    print(f"🧪 Test Case {case_num}: {case['name']}")
    print(f"📌 Question: {case['question']}")
    
    # Prepare payload
    payload = {"question": case["question"]}
    
    # Handle image if provided
    if "image" in case:
        encoded_image = encode_image(case["image"])
        if encoded_image:
            payload["image"] = encoded_image
            print(f"🖼️  Image attached: {case['image']}")
        else:
            print(f"⚠️  Skipping image: {case['image']}")
    
    # Make API request
    try:
        print("🔄 Making API request...")
        response = requests.post(API_URL, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "case_name": case["name"]
            }
        
        data = response.json()
        answer = data.get("answer", "").strip()
        links = [link["url"] for link in data.get("links", [])]
        
        print(f"🧠 Answer ({len(answer)} chars):")
        print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}\n")
        
        print(f"🔗 Links returned ({len(links)}):")
        for link in links:
            print(f"   • {link}")
        if not links:
            print("   (No links returned)")
        print()
        
        # Evaluate rubric
        rubric_results = {}
        if "rubric" in case:
            rubric_criteria = case["rubric"] if isinstance(case["rubric"], list) else [case["rubric"]]
            rubric_results = evaluate_rubric(answer, rubric_criteria)
            
            print("📋 Rubric Evaluation:")
            passed_count = 0
            for criterion, passed in rubric_results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   {status}: '{criterion}'")
                if passed:
                    passed_count += 1
            
            print(f"   Overall: {passed_count}/{len(rubric_criteria)} criteria passed")
        
        # Check expected link
        link_found = False
        if "expected_link" in case:
            expected_link = case["expected_link"]
            link_found = any(expected_link in link for link in links)
            
            print(f"\n🔍 Expected Link Check:")
            if link_found:
                print(f"   ✅ FOUND: {expected_link}")
            else:
                print(f"   ❌ MISSING: {expected_link}")
        
        return {
            "success": True,
            "case_name": case["name"],
            "answer_length": len(answer),
            "links_count": len(links),
            "rubric_results": rubric_results,
            "expected_link_found": link_found,
            "links": links
        }
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (30s)")
        return {
            "success": False,
            "error": "Timeout",
            "case_name": case["name"]
        }
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "case_name": case["name"]
        }

def print_summary(results: List[Dict[str, Any]]):
    """Print test summary"""
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    successful_tests = [r for r in results if r.get("success", False)]
    failed_tests = [r for r in results if not r.get("success", False)]
    
    print(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed tests: {len(failed_tests)}/{len(results)}")
    
    if failed_tests:
        print("\n🔴 Failed Tests:")
        for test in failed_tests:
            print(f"   • {test['case_name']}: {test.get('error', 'Unknown error')}")
    
    if successful_tests:
        print("\n🟢 Successful Tests Details:")
        for test in successful_tests:
            rubric_passed = sum(test.get("rubric_results", {}).values())
            rubric_total = len(test.get("rubric_results", {}))
            link_status = "✅" if test.get("expected_link_found", False) else "❌" if "expected_link_found" in test else "➖"
            
            print(f"   • {test['case_name']}:")
            print(f"     - Answer: {test.get('answer_length', 0)} chars")
            print(f"     - Links: {test.get('links_count', 0)}")
            if rubric_total > 0:
                print(f"     - Rubric: {rubric_passed}/{rubric_total}")
            print(f"     - Expected Link: {link_status}")

def run_tests():
    """Run all test cases"""
    print("🚀 Starting TDS Virtual TA API Tests")
    print("="*80)
    
    # Check API health first
    if not check_api_health():
        print("❌ API health check failed. Please ensure the API is running.")
        return
    
    print(f"\n🧪 Running {len(test_cases)} test cases...\n")
    
    results = []
    for i, case in enumerate(test_cases, 1):
        result = run_test_case(case, i)
        results.append(result)
        
        # Small delay between tests
        if i < len(test_cases):
            import time
            time.sleep(1)
    
    # Print summary
    print_summary(results)
    
    # Return results for programmatic use
    return results

if __name__ == "__main__":
    results = run_tests()