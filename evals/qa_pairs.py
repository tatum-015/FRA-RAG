"""Evaluation QA pairs and queries for FRA document retrieval testing."""
from typing import List, Dict, Any

# Structured Q&A pairs organized by metric type
EVAL_QA_PAIRS: Dict[str, List[Dict[str, str]]] = {
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

# Additional evaluation queries for testing specific aspects
EVAL_QUERIES = {
    "regulatory": [
        "What is the property classification?",
        "When was the last assessment conducted?",
    ],
    "safety_systems": [
        "Describe the emergency lighting system",
        "What type of fire alarm system is installed?",
    ],
    "actions_required": [
        "List all urgent priority actions",
        "What fire door issues need addressing?",
    ],
    "risk_assessment": [
        "What factors contribute to the risk rating?",
        "How could the risk rating be improved?",
    ]
}

