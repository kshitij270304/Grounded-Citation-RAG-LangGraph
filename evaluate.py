import os
import json
from graph import app
from datetime import datetime

TEST_CASES = [
    "What is the primary purpose of the EU AI Act according to the regulation?",
    "What does the acronym TFEU stand for?",
    "What are the exact tax penalties for violating the EU AI Act?",
    "Who is the intended audience for the NIST AI Risk Management Framework?",
    "According to the NIST framework, is AI risk management a one-time activity or a continuous process?",
    "Does the NIST AI RMF mandate strict legal compliance for all organizations?",
    "If a company is deploying an AI system, which of our documents provides mandatory legal requirements, and which provides voluntary risk management guidelines?",
    "How do the EU AI Act and the NIST AI RMF differ in their fundamental approach to AI?",
    
    # New Test Cases for Comprehensive Coverage
    "What are the four core functions of the NIST AI RMF profile?",
    "Does the EU AI Act prohibit any specific types of AI systems entirely? If so, which ones?",
    "What are the requirements for transparency in generative AI models under the EU AI Act?",
    "How does the NIST AI RMF define 'AI risk' and how is it measured?",
    "What role does human oversight play in the deployment of high-risk AI systems under the EU AI Act?",
    "Are there any specific exemptions mentioned in the EU AI Act for national security?"
]

def run_evals():
    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
        
    results = []
    
    with open("evaluation/report.md", "w") as md_file:
        md_file.write("# RAG Evaluation Report\n\n")
        md_file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, query in enumerate(TEST_CASES):
            print(f"Evaluating Q{i+1}: {query}")
            final_result = app.invoke({"question": query})
            
            gen = final_result.get("generation", {})
            reasoning = gen.get("reasoning", "")
            answer = gen.get("answer", "")
            citation = gen.get("citation", "")
            
            results.append({
                "question": query,
                "reasoning": reasoning,
                "answer": answer,
                "citation": citation,
                "valid_citation": final_result.get("is_valid", False)
            })
            
            md_file.write(f"### Q{i+1}: {query}\n")
            md_file.write(f"**Reasoning:** {reasoning}\n\n")
            md_file.write(f"**Answer:** {answer}\n\n")
            md_file.write(f"**Citation:** {citation}\n\n")
            md_file.write("---\n\n")
            
    with open("evaluation/results.json", "w") as json_file:
        json.dump(results, f=json_file, indent=4)
        
    print("Evaluation complete! Check evaluation/report.md")

if __name__ == "__main__":
    run_evals()
