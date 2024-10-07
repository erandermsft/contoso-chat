# %%
import os
import json
import prompty
from evaluators.custom_evals.coherence import coherence_evaluation
from evaluators.custom_evals.relevance import relevance_evaluation
from evaluators.custom_evals.fluency import fluency_evaluation
from evaluators.custom_evals.groundedness import groundedness_evaluation
from azure.ai.evaluation import evaluate
import jsonlines
import pandas as pd
from prompty.tracer import trace
from tracing import init_tracing
from contoso_chat.chat_request import get_response

# %% [markdown]
# ## Get output from data and save to results jsonl file

# %%


@trace
def load_data():
    data_path = "./evaluators/data.jsonl"

    df = pd.read_json(data_path, lines=True)
    df.head()
    return df

# %%


@trace
def create_response_data(df):
    results = []

    for index, row in df.iterrows():
        customerId = row['customerId']
        question = row['question']

        # Run contoso-chat/chat_request flow to get response
        response = get_response(customerId=customerId,
                                question=question, chat_history=[])
        print(response)

        # Add results to list
        result = {
            'question': question,
            'context': response["context"],
            'answer': response["answer"]
        }
        results.append(result)

    # Save results to a JSONL file
    with open('result.jsonl', 'w') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')
    return results

# %%


@trace
def evaluate_wrapper():
    # Evaluate results from results file
    results_path = 'result.jsonl'
    # results = []
    # with open(results_path, 'r') as file:
    #     for line in file:
    #         print(line)
    #         results.append(json.loads(line))
    azure_ai_project = {
        "subscription_id": os.environ.get("AZURE_SUBSCRIPTION_ID"),
        "resource_group_name": os.environ.get("AZURE_RESOURCE_GROUP"),
        "project_name": os.environ.get("AZURE_PROJECT_NAME"),
    }

    result_eval = evaluate(
        data=results_path,
        # target=get_response,
        evaluators={
            "relevance": relevance_evaluation,
            # "fluency": fluency_evaluator,
            # "coherence": coherence_evaluator,
            # "groundedness": groundedness_evaluator,
        },
        evaluator_config={
            "default": {
                "question": "${data.question}",
                "answer": "${data.answer}",
                "context": "${data.context}",
            },
        },
        azure_ai_project=azure_ai_project

    )
    eval_result = pd.DataFrame(result_eval["rows"])
    # for result in results:
    #     question = result['question']
    #     context = result['context']
    #     answer = result['answer']

    #     groundedness_score = groundedness_evaluation(question=question, answer=answer, context=context)
    #     fluency_score = fluency_evaluation(question=question, answer=answer, context=context)
    #     coherence_score = coherence_evaluation(question=question, answer=answer, context=context)
    #     relevance_score = relevance_evaluation(question=question, answer=answer, context=context)

    #     result['groundedness'] = groundedness_score
    #     result['fluency'] = fluency_score
    #     result['coherence'] = coherence_score
    #     result['relevance'] = relevance_score

    # Save results to a JSONL file
    # with open('result_evaluated.jsonl', 'w') as file:
    #     for result in results:
    #         file.write(json.dumps(result) + '\n')

    # with jsonlines.open('eval_results.jsonl', 'w') as writer:
    #     writer.write(results)
    # Print results

    # df = pd.read_json('result_evaluated.jsonl', lines=True)
    # df.head()
    eval_result.rename(columns={"inputs.question": "question","inputs.answer" : "answer", "inputs.context": "context"}, inplace=True)

    return eval_result

# %%


@trace
def create_summary(df):
    print("Evaluation summary:\n")
    print(df)
    # drop question, context and answer
    mean_df = df.drop(["question", "context", "answer"], axis=1).mean()
    print("\nAverage scores:")
    print(mean_df)
    df.to_markdown('eval_results.md')
    with open('eval_results.md', 'a') as file:
        file.write("\n\nAverages scores:\n\n")
    mean_df.to_markdown('eval_results.md', 'a')

    print("Results saved to result_evaluated.jsonl")


# %%
# create main funciton for python script
if __name__ == "__main__":
    tracer = init_tracing(local_tracing=True)
    test_data_df = load_data()
    response_results = create_response_data(test_data_df)
    result_evaluated = evaluate_wrapper()
    create_summary(result_evaluated)