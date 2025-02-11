from typing import List, Dict, TypedDict

class QAPair(TypedDict):
    question: str
    answer: str
    context_sections: List[str]  # Relevant document sections
    retrieval_type: str  # Type of retrieval/reasoning required
    ragas_metrics: List[str]  # Relevant RAGAS metrics

# QA pairs for testing FRA document retrieval
qa_pairs: List[QAPair] = [
    {
        "question": "What is the current evacuation strategy for 80 Glengall Road and what factors were considered in determining this?",
        "answer": "The current evacuation strategy is Simultaneous Evacuation. This was determined based on the building being a conversion with unknown standard of compartmentation, as noted in the evacuation strategy notes section 1.3. The strategy aligns with the risk assessment rating of 'Moderate'.",
        "context_sections": ["1.3 Evacuation Strategy", "1.1 Compliance and Risk Record"],
        "retrieval_type": "multi_section_reasoning",
        "ragas_metrics": ["context_precision", "answer_relevancy"]
    },
    {
        "question": "How many days do priority 'A' recommendations have for completion, and which specific recommendations fall under this category?",
        "answer": "Priority 'A' recommendations have 6 months (180 days) for completion according to the priority timeline in section 3.4. Each recommendation with priority 'A' can be found in section 2.1 of the Action Plan.",
        "context_sections": ["3.4 Recommendation Priorities", "2.1 Recommendations Assessment"],
        "retrieval_type": "timeline_tracking",
        "ragas_metrics": ["faithfulness", "context_recall"]
    },
    {
        "question": "For the fire door recommendations, what are the completion timelines and how do they relate to the building's current risk rating?",
        "answer": "Fire door recommendations should be cross-referenced between section 2.1 (Action Plan) and section 3.4 (Priority Timeline). The building's 'Moderate' risk rating influences the urgency of these actions, particularly for issues affecting means of escape.",
        "context_sections": ["2.1 Recommendations Assessment", "3.4 Recommendation Priorities", "1.1 Compliance and Risk Record"],
        "retrieval_type": "complex_relationship",
        "ragas_metrics": ["context_relevance", "answer_relevancy"]
    },
    {
        "question": "What impact would upgrading from a 'Grade D' to a 'Grade A' fire alarm system have on the evacuation strategy?",
        "answer": "This links to both fire safety systems and evacuation strategy sections. A Grade A system would provide enhanced detection and warning capabilities, potentially affecting the evacuation strategy, but any changes would need to consider the building's conversion status and compartmentation standards.",
        "context_sections": ["1.3 Evacuation Strategy", "5. FRA Questionnaire"],
        "retrieval_type": "technical_analysis",
        "ragas_metrics": ["faithfulness", "context_precision"]
    },
    {
        "question": "What are the specific responsibilities of the named responsible person (Hyde Housing Group) regarding fire safety management?",
        "answer": "Their responsibilities are defined in the legislation section 3.2, referencing Article 3 of the Fire Safety Order, and detailed in the management sections of the questionnaire (Fire Safety Management section).",
        "context_sections": ["3.2 Legislation", "5. FRA Questionnaire"],
        "retrieval_type": "regulatory_compliance",
        "ragas_metrics": ["faithfulness", "context_recall"]
    },
    {
        "question": "Based on the building layout and occupant information, what fire safety measures are required for vulnerable residents?",
        "answer": "This requires analyzing section 4.1 (Building Layout) and 4.3 (Occupant Information), particularly the special risk subsection, to determine appropriate measures for vulnerable residents.",
        "context_sections": ["4.1 Building Layout", "4.3 Occupant Information"],
        "retrieval_type": "safety_assessment",
        "ragas_metrics": ["context_relevance", "answer_relevancy"]
    },
    {
        "question": "How do the current fire safety systems align with the property classification and risk assessment?",
        "answer": "Compare the property classification (Level 2) from section 0 with the fire safety systems inventory and risk assessment (section 1.1) to evaluate alignment with required standards.",
        "context_sections": ["0. Document Metadata", "1.1 Compliance and Risk Record"],
        "retrieval_type": "compliance_verification",
        "ragas_metrics": ["faithfulness", "context_precision"]
    },
    {
        "question": "Which recommendations have dependencies on other actions being completed first?",
        "answer": "Review section 2.1 (Recommendations Assessment) to identify linked actions, particularly those affecting compartmentation and evacuation strategy changes.",
        "context_sections": ["2.1 Recommendations Assessment"],
        "retrieval_type": "dependency_analysis",
        "ragas_metrics": ["context_recall", "answer_relevancy"]
    },
    {
        "question": "How does the current risk rating impact the frequency of future FRA assessments?",
        "answer": "Link the risk rating from section 1.1 with the reassessment date recommendations, considering any escalating factors noted in the risk assessment.",
        "context_sections": ["1.1 Compliance and Risk Record", "3.3 Risk Level Matrix"],
        "retrieval_type": "risk_assessment",
        "ragas_metrics": ["faithfulness", "context_relevance"]
    },
    {
        "question": "What are the implications of 'limited access areas' on the assessment's completeness and risk rating?",
        "answer": "Review limitations section 3.1 in context with inaccessible areas noted in building layout (4.1) and their impact on risk assessment.",
        "context_sections": ["3.1 Limitations", "4.1 Building Layout", "1.1 Compliance and Risk Record"],
        "retrieval_type": "limitation_analysis",
        "ragas_metrics": ["context_precision", "answer_relevancy"]
    } 
]

 # Additional QA pairs focusing on specific aspects...
advanced_qa_pairs: List[QAPair] = [
    {
        "question": "How do recent regulatory changes affect the current FRA recommendations?",
        "answer": "Compare legislation section 3.2 with current recommendations, considering any updated regulatory requirements.",
        "context_sections": ["3.2 Legislation", "2.1 Recommendations Assessment"],
        "retrieval_type": "regulatory_impact",
        "ragas_metrics": ["faithfulness", "context_recall"]
    },
    {
        "question": "How does the condition of compartmentation affect both the risk rating and evacuation strategy?",
        "answer": "Link compartmentation findings from the questionnaire with both risk assessment (1.1) and evacuation strategy (1.3).",
        "context_sections": ["5. FRA Questionnaire", "1.1 Compliance and Risk Record", "1.3 Evacuation Strategy"],
        "retrieval_type": "complex_relationship",
        "ragas_metrics": ["context_relevance", "answer_relevancy"]
    } 
]


def get_qa_pairs(category: str = "all") -> List[QAPair]:
    """
    Retrieve QA pairs based on category.
    
    Args:
        category: Type of QA pairs to retrieve (e.g., "basic", "advanced", "all")
    
    Returns:
        List of QA pairs matching the specified category
    """
    if category == "basic":
        return qa_pairs
    elif category == "advanced":
        return advanced_qa_pairs
    else:
        return qa_pairs + advanced_qa_pairs

def get_qa_pairs_by_metric(metric: str) -> List[QAPair]:
    """
    Retrieve QA pairs that test a specific RAGAS metric.
    
    Args:
        metric: RAGAS metric to filter by
    
    Returns:
        List of QA pairs testing the specified metric
    """
    return [
        qa for qa in get_qa_pairs()
        if metric in qa["ragas_metrics"]
    ]

def get_qa_pairs_by_retrieval_type(retrieval_type: str) -> List[QAPair]:
    """
    Retrieve QA pairs that test a specific type of retrieval.
    
    Args:
        retrieval_type: Type of retrieval to filter by
    
    Returns:
        List of QA pairs testing the specified retrieval type
    """
    return [
        qa for qa in get_qa_pairs()
        if qa["retrieval_type"] == retrieval_type
    ]

if __name__ == "__main__":
    # Example usage
    print(f"Total QA pairs: {len(get_qa_pairs())}")
    print(f"Faithfulness tests: {len(get_qa_pairs_by_metric('faithfulness'))}")
    print(f"Complex relationship tests: {len(get_qa_pairs_by_retrieval_type('complex_relationship'))}")


# Older evals
""""
    {
         "single_fact": [
        {
            "question": "When is the recommended reassessment date for UPRN-abcd1266-5?",
            "answer": "16 January 2027",
            "metric": "single_fact_accuracy"
        },
        {
            "question": "What is the current risk rating for this building?",
            "answer": "Moderate",
            "metric": "single_fact_accuracy"
        }
    ],
    "action_item": [
        {
            "question": "What is Action ID 706137 and its priority level?",
            "answer": "Action ID 706137 has Priority U (1 day)",
            "metric": "action_item_completeness"
        },
        {
            "question": "List all Priority A actions for this building",
            "answer": "Priority A - Action ID 706134: Arrangements should be put in place to ensure the common area is regularly cleaned to prevent the build up of combustible items. Due date: 16/01/2025",
            "metric": "action_item_completeness"
        }
    ],
    "faithfulness": [
        {
            "question": "Why is the building rated as moderate risk?",
            "answer": "The building is rated as moderate risk because the likelihood is Medium and consequence is Moderate Harm, where outbreak of fire could foreseeably result in injury of one or more occupants",
            "metric": "faithfulness"
        },
        {
            "question": "What is the current evacuation strategy and why?",
            "answer": "The evacuation strategy is Simultaneous Evacuation. This is considered appropriate whilst identified defects remain outstanding. Once resolved, the premises should revert to Stay Put strategy",
            "metric": "faithfulness"
        }
    ],
    "context_precision": [
        {
            "question": "List all fire safety systems installed in this building",
            "answer": "The building has: disabled evacuation aids, emergency lighting, evacuation alert system, extinguishers, fire alarm system, fire blankets, fire mains, hose reels, lifts for fire safety uses, lightning protection, smoke control system, and sprinkler system",
            "metric": "context_precision"
        },
        {
            "question": "What are the issues with gas services in the building?",
            "answer": "Gas meters and supply pipework in electrical cupboard are insufficiently separated from electrical equipment. They need to be at least 150mm from electricity meters and 25mm from electricity cables per BS 6891",
            "metric": "context_precision"
        } 
    ]
}

# Metric definitions for evaluation
METRICS = {
    "single_fact_accuracy": "Measures accuracy of retrieving specific regulatory information",
    "action_item_completeness": "Measures if recommendations and their context stay together",
    "faithfulness": "Measures if responses are factually supported by source",
    "context_precision": "Measures relevance of retrieved chunks"
} 
"""