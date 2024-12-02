import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from config import GROQ_API_KEY
from main import RAGApplication

class RAGEvaluator:
    def __init__(self, pdf_path: str, api_key: str):
        """
        Initialize RAG application for evaluation
        """
        self.rag_app = RAGApplication(
            pdf_path=pdf_path, 
            groq_api_key=api_key
        )
        
        # Ensure index is created
        if not self.rag_app.index:
            self.rag_app.create_index()
        
        # Ground truth and test cases
        self.ground_truth = self._prepare_ground_truth()

    def _prepare_ground_truth(self):
        """
        Prepare ground truth data for evaluation
        Each entry should have:
        - query: The input query
        - expected_keywords: Keywords that should be in the response
        - is_relevant: Boolean indicating if the query is expected to have a relevant answer
        """
        return [
            {
                "query": "When should I use a pie chart?",
                "expected_keywords": ["pie chart", "proportion", "small number", "categories"],
                "is_relevant": True
            },
            {
                "query": "What are the key principles of effective dashboard design?",
                "expected_keywords": ["clarity", "simplicity", "user", "focus", "design"],
                "is_relevant": True
            },
            {
                "query": "How do I choose color schemes for data visualization?",
                "expected_keywords": ["color", "perception", "accessibility", "contrast"],
                "is_relevant": True
            },
            {
                "query": "What is a BAN in dashboard design?",
                "expected_keywords": ["Big-Ass Number", "Big", "key metric", "large"],
                "is_relevant": True
            },
            {
                "query": "Best practices for displaying time series data",
                "expected_keywords": ["time series", "trend", "line chart", "data over time"],
                "is_relevant": True
            },
            {
                "query": "What sound does a cow make?", 
                "expected_keywords": [],
                "is_relevant": False
            }
        ]

    def _evaluate_response(self, response: str, test_case: dict) -> dict:
        """
        Evaluate individual response
        """
        # Check if response contains expected keywords
        keyword_hits = [
            keyword.lower() in response.lower() 
            for keyword in test_case['expected_keywords']
        ]
        
        # Calculate hit rate for keywords
        keyword_hit_rate = np.mean(keyword_hits) if test_case['expected_keywords'] else 0
        
        # Relevance check
        is_response_relevant = (
            (test_case['is_relevant'] and keyword_hit_rate > 0) or 
            (not test_case['is_relevant'] and keyword_hit_rate == 0)
        )
        
        return {
            'response': response,
            'keyword_hit_rate': keyword_hit_rate,
            'is_response_relevant': is_response_relevant
        }

    def run_evaluation(self):
        """
        Run comprehensive evaluation
        """
        # Stores evaluation results
        results = []
        
        # Predictions and ground truth for metrics
        y_true = []
        y_pred = []
        
        # Iterate through test cases
        for test_case in self.ground_truth:
            query = test_case['query']
            
            try:
                # Get RAG response
                response = self.rag_app.query(query)
                
                # Evaluate response
                eval_result = self._evaluate_response(response, test_case)
                
                # Store full evaluation result
                eval_result['query'] = query
                results.append(eval_result)
                
                # For accuracy and F1 calculation
                y_true.append(test_case['is_relevant'])
                y_pred.append(eval_result['is_response_relevant'])
                
                # Print individual result
                print(f"Query: {query}")
                print(f"Response: {response}")
                print(f"Keyword Hit Rate: {eval_result['keyword_hit_rate']:.2f}")
                print(f"Relevance: {eval_result['is_relevant']}\n")
            
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Calculate overall hit rate
        hit_rates = [r['keyword_hit_rate'] for r in results]
        avg_hit_rate = np.mean(hit_rates)
        
        # Print overall metrics
        print("\n--- Performance Metrics ---")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Average Keyword Hit Rate: {avg_hit_rate:.2f}")
        
        return {
            'results': results,
            'accuracy': accuracy,
            'f1_score': f1,
            'avg_hit_rate': avg_hit_rate
        }

def main():
    # PDF path 
    pdf_path = 'data/The Big Book of Dashboards.pdf'

    # Create evaluator
    evaluator = RAGEvaluator(
        pdf_path=pdf_path, 
        api_key=GROQ_API_KEY
    )

    # Run comprehensive evaluation
    evaluation_results = evaluator.run_evaluation()

if __name__ == "__main__":
    main()

'''
import numpy as np
from trulens.core import Feedback
from trulens.core import Select
from trulens.providers.openai import OpenAI

from config import GROQ_API_KEY, OPENAI_API_KEY
from main import RAGApplication

from trulens.core import TruSession

session = TruSession()
session.reset_database()

provider = OpenAI(api_key=OPENAI_API_KEY)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
    .aggregate(np.mean)  # choose a different aggregation method if you wish
)

from trulens.apps.custom import TruCustomApp

rag = RAGApplication(pdf_path="data/The Big Book of Dashboards.pdf",groq_api_key=GROQ_API_KEY)

tru_rag = TruCustomApp(
    rag,
    app_name="RAG",
    app_version="base",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

with tru_rag as recording:
    rag.query(
        "When should I use a pie chart?"
    )
    rag.query(
        "What are the key principles of effective dashboard design?"
    )
    rag.query(
        "How do I choose color schemes for data visualization?"
    )
    rag.query(
        "What is a BAN in dashboard design?"
    )
    rag.query(
        "Best practices for displaying time series data"
    )

session.get_leaderboard()
'''