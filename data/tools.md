# Tools and the ReAct Loop

AgentDesk's executor can call tools through a ReAct loop: it alternates Thought,
Action, Action Input, and Observation steps until it produces a Final Answer.

Two tools ship by default. The calculator evaluates a basic arithmetic
expression safely without using eval. The web_search tool queries the DuckDuckGo
Instant Answer API and returns a short factual snippet.

Tools are registered in a simple registry, so adding a new tool is a matter of
writing a function and adding it to the registry with a name and description.

Evaluation measures grounded faithfulness: the fraction of the answer's content
words that are supported by the retrieved context. A no-retrieval baseline is
compared against the retrieval-augmented answer to show the lift from grounding.
