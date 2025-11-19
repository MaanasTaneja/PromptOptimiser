from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import List, Optional
from pydantic import BaseModel, Field  # Import from pydantic, not langchain_core.pydantic_v1
from collections import deque 

from datasets import generate_sentiment_dataset, generate_qa_dataset, generate_summarization_dataset, generate_nli_dataset, generate_reasoning_dataset
from moves import GenerationMove, VerboseMove, ChainOfThoughtMove, ShortenMove, ReorderMove, AddExamplesMove, \
AddConstraintsMove, RephraseInputOutputInstructionsMove, RoleAssignmentMove, AddDefinitionsMove


load_dotenv()
evaluation_cache = {}

class OpenAILLM():
    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key
        if self.api_key is None:
            api_key = os.getenv("OPENAI_KEY")
            if not api_key:
                raise ValueError("API key must be provided either as an argument (not recommended) or through the OPENAI_KEY environment variable.")
            self.api_key = api_key

        self.llm = ChatOpenAI(model=model_name, api_key=api_key, temperature=0.7)

    def get_model(self):
        return self.llm

class CriticLM(OpenAILLM):
    #evaluate function will evaluate response using the more powerful LM
    def __init__(self, api_key: str = None):
        model_name = "gpt-5" #using more powerful language model 
        super().__init__(model_name, api_key)

    def evaluate(self, prompt: str, response: str, expected: str) -> bool:
        class Evaluation(BaseModel):
                is_correct : bool = Field(..., description="Whether the model's response is correct")
        prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict but fair evaluator. Determine whether the model's response "
     "is an acceptable match to the expected output.\n\n"
     "A response is correct IF AND ONLY IF:\n"
     "1. It contains all core meaning units found in the expected output.\n"
     "2. It does NOT introduce major unrelated ideas (minor elaboration is OK).\n"
     "3. It is not excessively longer than the expected output "
     "(no more than 3x the number of words).\n"
     "4. It matches the overall output FORMAT (sentence vs list, etc.).\n"
     "\nA response is incorrect if:\n"
     "- It contradicts the expected output\n"
     "- It adds many unrelated points\n"
     "- It is far too long or detailed compared to the expected output\n"
     "- It does not follow the expected format\n"
     "\nReturn ONLY: true or false."
    ),
    ("user",
     "Prompt: {prompt}\n\n"
     "Model Response: {response}\n\n"
     "Expected Output: {expected}\n\n"
     "Is the model's response correct? Answer with true or false."
    )
])

        #print(prompt_template.format_prompt(prompt=prompt, response=response, expected=expected).to_string())
        evaluator = self.llm.with_structured_output(Evaluation)
        try:
            result = evaluator.invoke([{"role": "user", "content": prompt_template.format_prompt(prompt=prompt, response=response, expected=expected).to_string()}])
            #print("critic claims this was: ", result)
            return result.is_correct
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return False
        



#i am just creating critic lm as global i dont care rn 

critic_lm = None

try:
    critic_lm = CriticLM()
except ValueError as e:
    print(f"Error: {e}")
    exit(1)


class ModelOutput(BaseModel):
    response : str = Field(..., description="The response from the model")

class PromptNode:
    def __init__(self, prompt_text: str, parent=None, operator=None, score: Optional[float] = None):
        self.prompt_text = prompt_text
        self.parent = parent
        self.operator = operator
        self.score = score
        self.children: List["PromptNode"] = []

    def add_child(self, child: "PromptNode"):
        self.children.append(child)
    
    def get_path(self) -> List[str]:
        """Return the sequence of operators from root to this node."""
        path = []
        node = self
        while node.parent is not None:
            path.append(node.operator)
            node = node.parent
        return list(reversed(path))

def evaluate_metric_string_match(prompt: str, dev_set: List[dict], llm: ChatOpenAI, task_type: str = "classification") -> float:
    """
    Run LLM with given prompt on dev examples.
    Score against ground truth using simple metric.
    """

    #add caching to evaluation metrics.

    key = (prompt, task_type, len(dev_set))  # can add dataset hashes later if needed

    if key in evaluation_cache:
        return evaluation_cache[key]

    correct = 0
    total = len(dev_set)
    
    if total == 0:
        return 0.0
    
    for example in dev_set:
        try:
            formatted_prompt = f"{prompt}\n\nInput: {example['input']}\n\nOutput:"
            result = llm.invoke([{"role": "user", "content": formatted_prompt}])
            pred = result.content.strip().lower()
            #make sure predicted and exptected are smaller lower case strings.
            expected = str(example["output"]).lower()
            
            # Flexible matching for classification tasks
            #huh what the ufck is this/ oh so if thr model even says this is positive it catxhes it.. stupid.
            if task_type == "classification":
                # Check if the expected label appears in the prediction
                if expected in pred or pred in expected:
                    correct += 1
            else:
                # Exact match for other tasks
                if pred == expected:
                    correct += 1
        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    evaluation_cache[key] = accuracy
    return accuracy


def evaluate_metric_critic_lm(prompt: str, dev_set: List[dict], llm: ChatOpenAI, task_type: str = "classification") -> float:
    """
    Run LLM with given prompt on dev examples.
    Score against ground truth using simple metric.
    """

    #add caching to evaluation metrics.

    key = (prompt, task_type, len(dev_set)) # can add dataset hashes later if needed

    if key in evaluation_cache:
        return evaluation_cache[key]
    
    correct = 0
    total = len(dev_set)
    
    if total == 0:
        return 0.0
    
    for example in dev_set:
        try:
            formatted_prompt = f"{prompt}\n\nInput: {example['input']}\n\nOutput:"
            result = llm.invoke([{"role": "user", "content": formatted_prompt}])
            pred = result.content.strip().lower()
            #make sure predicted and exptected are smaller lower case strings.
            expected = str(example["output"]).lower()

            if(critic_lm.evaluate(prompt=formatted_prompt, response=pred, expected=expected)):
                correct += 1


        except Exception as e:
            print(f"Error evaluating example: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    evaluation_cache[key] = accuracy
    return accuracy


def beam_search(seed_prompt: str, train_set : List[dict], dev_set: List[dict], llm: OpenAILLM, 
                moves: List[GenerationMove], beam_width: int = 3, depth: int = 3,
                task_type: str = "classification", evaluation_metric = None) -> PromptNode:
    """
    Perform beam search over prompt space.
    i want the grpah search to be genrate all moves at each node, evaluate each prompt node, and then keep the top k in this 
    level essenitally.

    bfs is what? queue = root prompt (seed prompt)
    then pop from queue, generate all children evaluadate each child, 
    """
    model = llm.get_model()
    
    print(f"\n{'='*60}")
    print(f"Starting Beam Search (beam_width={beam_width}, depth={depth})")
    print(f"{'='*60}\n")
    
    # Evaluate seed prompt
    seed_score = evaluation_metric(seed_prompt, dev_set, model, task_type)
    root = PromptNode(seed_prompt, score=seed_score)

    queue = deque([root])
    
    best = root
    
    #this is our baseline.
    print(f"Seed prompt score: {seed_score:.3f}")
    print(f"Seed: {seed_prompt[:60]}...\n")

    if(seed_score == 1.0):
        print(f"{'='*60}")
        print("Perfect score on Dev Set achieved with seed prompt! Ending search early.")
        print(f"Search complete!")
        print(f"Best score: {best.score:.3f}")
        return best

    for d in range(depth): #this replaes while queue, since we want to limit the number of levels
        print(f"--- Depth {d+1}/{depth} ---")

        #now do bfs beam search here
        queue_len = len(queue)

        for _ in range(queue_len): #and this ensures we go thorugh full level before moving to next level.
            node = queue.popleft()

            print(f"Current Node (score: {node.score:.3f}): {node.operator}")

            for move in moves:
                new_prompt = move.apply(prompt=node.prompt_text, train_set=train_set)
                child_node = PromptNode(new_prompt, parent=node, operator=move.name)

                node.add_child(child_node) #keep generating the graph structure.
                queue.append(child_node) #add to queue for next level processing.

                #lets evaluate the node as soon as its geneatted since we need to beam search (prune paths that are not 
                #leading anywhere good)

                score = evaluation_metric(child_node.prompt_text, dev_set, model, task_type)
                child_node.score = score

                if(score > best.score): #oh also this will only happen once? since it is > not = so its best to 
                    #return early anyway if we reach 1.0 on dev set, until i decide to maybe search more for more stable prompts?
                    best = child_node
                    print(f"  ⭐ New best at depth {d+1}! Score: {score:.3f}")
                    if(score == 1.0):
                        #achieved best score possible on the dev set, we can probably end here.
                        print("Perfect score on Dev Set achieved! Ending search early.")
                        #TODO : return best here?
                        #this is our goal state (reach a perfect score on the test state.)
                        #potential this is not a good idea? since we aregreedily getting out of search 
                        #maybe this priompt is not as stable as other ones deeper the search graph?
                        #possible.
                        return best

                symbol = "↑" if score >= node.score else "↓"
                print(f"  {move.name}: {score:.3f} ({symbol})")


        #now we can do beam search, sort by score
        queue = deque(sorted(list(queue), key = lambda x: x.score, reverse = True)[:beam_width])
        #and we do this for only depth times (not until we keep doing and blow the memeory to shit)

        print(f"  Queue scores: {[f'{n.score:.3f}' for n in queue]}\n")
        

    print(f"{'='*60}")
    print(f"Search complete!")
    print(f"Best score: {best.score:.3f}")
    print(f"Optimization path: {' -> '.join(best.get_path()) if best.get_path() else 'seed'}")
    print(f"{'='*60}\n")

    return best


        
def one_hop_improve(llm, seed_prompt: str):
    generation_prompt = f"""You are an expert prompt engineer. Given a prompt, can you please improve it as best as you can to maximise the probabilty of getting an expected answer.
    Prompt: {seed_prompt}
    """
    try:
        result = llm.invoke([{"role": "user", "content": generation_prompt}])
        return result.content.strip()
    except Exception as e:
        print(f"Error generating improved one hop prompt: {e}")
        return "No prompt generated"


import random
def random_walk_search(
    seed_prompt: str,
    train_set: List[dict],
    dev_set: List[dict],
    llm: OpenAILLM,
    moves: List[GenerationMove],
    steps: int = 10,
    task_type: str = "classification",
    evaluation_metric=None
) -> PromptNode:
    """
    Perform a random walk over prompt space.
    At each step:
      - choose a random move
      - apply it
      - evaluate the new prompt
    Keep track of the best prompt seen so far.
    """

    model = llm.get_model()

    print(f"\n{'='*60}")
    print(f"Starting Random Walk (steps={steps})")
    print(f"{'='*60}\n")

    # Evaluate seed prompt
    seed_score = evaluation_metric(seed_prompt, dev_set, model, task_type)
    root = PromptNode(seed_prompt, score=seed_score)

    best = root
    current = root

    print(f"Seed prompt score: {seed_score:.3f}")
    print(f"Seed: {seed_prompt[:60]}...\n")

    for step in range(1, steps + 1):
        print(f"--- Step {step}/{steps} ---")

        # Pick a random move
        move = random.choice(moves)
        print(f"Applying move: {move.name}")

        # Generate new prompt
        new_prompt = move.apply(current.prompt_text, train_set=train_set)
        child_node = PromptNode(new_prompt, parent=current, operator=move.name)

        # Evaluate
        score = evaluation_metric(child_node.prompt_text, dev_set, model, task_type)
        child_node.score = score

        # Track as child (helps reconstruct chain)
        current.add_child(child_node)
        current = child_node  # random walk moves forward only

        symbol = "↑" if score >= best.score else "↓"
        print(f"Score: {score:.3f} ({symbol})")

        # Update global best
        if score > best.score:
            best = child_node
            print(f"  ⭐ New best score found: {score:.3f}")

    print(f"\n{'='*60}")
    print("Random Walk Complete!")
    print(f"Best score: {best.score:.3f}")
    print(f"Path: {' -> '.join(best.get_path()) if best.get_path() else 'seed'}")
    print(f"{'='*60}\n")

    return best


def generate_seed_prompt(examples: List[dict], task_type: str, llm: ChatOpenAI) -> str:
    """
    Use LLM to generate a seed prompt by reverse-engineering from a single example.
    """

        # Format multiple examples for the meta-prompt
    formatted_examples = "\n\n".join(
        [
            f"Input: {ex['input']}\nOutput Format Template: {ex['output']}"
            for ex in examples
        ]
    )

    generation_prompt = f"""
You are an expert prompt engineer. Infer BOTH the task and the output format
from the example below.

Your goal is to write a high-quality task prompt that instructs an LLM to:
1. Correctly perform the same underlying task as shown in the example,
2. Produce outputs that match the FORMAT and STRUCTURE of the example output,
3. Without copying or referencing any of the example’s specific content.

Treat the example output ONLY as a FORMAT TEMPLATE, not as content to reuse.

STRICT RULES:
- Do NOT include the example output in the final prompt.
- Do NOT quote, paraphrase, or reuse any phrases from the example.
- Do NOT hint at the example content.
- Do NOT generate a task-specific answer.
- Only generate a general reusable prompt that describes the task and the required format.

You MUST infer:
- What type of task this is (classification, reasoning, comparison, summarization, etc.)
- What kind of reasoning or processing is needed to answer similar inputs
- What format the final answer should follow (single sentence, bullet list, comma-separated items, etc.)
- What style or level of detail is expected (concise? factual? structured?)

Examples:
{formatted_examples}

Task Type: {task_type}

Write a reusable, clear, and complete prompt that:
- Defines the task the model must perform,
- Describes how to analyze any future input,
- Specifies the structure and style the output must take,
- Does NOT leak or mention the example or its content.

Write ONLY the final crafted prompt below:"""
    
    try:
        result = llm.invoke([{"role": "user", "content": generation_prompt}])
        return result.content.strip()
    except Exception as e:
        print(f"Error generating seed prompt: {e}")
        return "No prompt generated"
    

def evaluate_on_test_set(
    prompt: str,
    test_set: List[dict],
    llm: OpenAILLM,
    task_type: str,
    evaluation_metric
) -> float:
    """
    Evaluate a final optimized prompt on a held-out test set.
    This should only be called AFTER beam search / optimization.

    Returns:
        test_accuracy (float)
    """

    print("\n==================== TEST SET EVALUATION ====================\n")
    model = llm.get_model()

    test_score = evaluation_metric(prompt, test_set, model, task_type)

    print(f"Test Set Score = {test_score:.3f}")
    print("\n==============================================================\n")

    return test_score



if __name__ == "__main__":
    # Choose dataset
    print("Select dataset:")
    print("1. Sentiment Classification")
    print("2. Question Answering")
    print("3. Summarization")
    print("4. Complex Reasoning (requires CriticLM)")
    print("5. Natural Language Inference (requires CriticLM)")
    
    choice = input("Enter choice (1-5): ").strip()
    
    if choice == "1":
        full_dataset = generate_sentiment_dataset()
        task_type = "classification"
        use_critic = False
    elif choice == "2":
        full_dataset = generate_qa_dataset()
        task_type = "qa"
        use_critic = False
    elif choice == "3":
        full_dataset = generate_summarization_dataset()
        task_type = "summarization"
        use_critic = True
    elif choice == "4":
        full_dataset = generate_reasoning_dataset()
        task_type = "reasoning"
        use_critic = True
    else:
        full_dataset = generate_nli_dataset()
        task_type = "nli"
        use_critic = True

    print(f"Loaded dataset with {len(full_dataset)} examples.")

    random.seed(42)
    random.shuffle(full_dataset)

    #if dataset is small, use 60/20/20 split
    n = len(full_dataset)
    seed_end = max(1, int(0.25 * n))
    dev_end  = seed_end + max(1, int(0.25 * n))

    train_set = full_dataset[:seed_end]        # 25% train set is our seed examples.
    dev_set       = full_dataset[seed_end:dev_end] # 25%
    test_set      = full_dataset[dev_end:]         # 50%

    print(
        f"Split -> Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}"
    )
    
    # Initialize LLM
    try:
        llm = OpenAILLM()
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    model = llm.get_model()

    print("Generating seed prompt from first example...")
    print(f"Example - Input: {train_set[0]['input'][:60]}...")
    print(f"Example - Output: {train_set[0]['output']}\n")
    
    #give 5 exampels at elast to the seed generator
    seed = generate_seed_prompt(train_set, task_type, model)
    print(f"Generated seed prompt:\n{seed}\n")
    print(f"{'='*60}\n")

    # Define moves - now using the LLM for transformations
    '''
    moves = [
        VerboseMove(model),
        ChainOfThoughtMove(model),
        ShortenMove(model),
        ReorderMove(model),
        AddExamplesMove(model),
        AddConstraintsMove(model),
        RephraseInputOutputInstructionsMove(model),
        RoleAssignmentMove(model), 
        AddDefinitionsMove(model)
    ]
    '''

    moves = [
    VerboseMove(model),
    ShortenMove(model),
    ReorderMove(model),
    AddExamplesMove(model),
]
    
    # Run beam search
    best_prompt = beam_search(
        seed, 
        train_set,
        dev_set, 
        llm, 
        moves, 
        beam_width=2, 
        depth=2,
        task_type=task_type,
        evaluation_metric= evaluate_metric_critic_lm if use_critic else evaluate_metric_string_match  # or evaluate_metric_critic_lm
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Best prompt:\n{best_prompt.prompt_text}\n")
    print(f"Final score: {best_prompt.score:.3f}")
    print(f"Optimization path: {' -> '.join(best_prompt.get_path()) if best_prompt.get_path() else 'seed (no optimization)'}")
    print("="*60)

    #eavlutate using test set

    test_score = evaluate_on_test_set(
    best_prompt.prompt_text,
    test_set,
    llm,
    task_type,
    evaluation_metric=evaluate_metric_critic_lm if use_critic else evaluate_metric_string_match
)

    print(f"FINAL TEST SCORE: {test_score:.3f}")
    

