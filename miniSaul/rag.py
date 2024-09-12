from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolNode
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.document import Document

import json
from typing import Any, List, Dict, Annotated, Sequence
from pydantic import BaseModel, Field

import pandas as pd
import math
import numexpr
from typing_extensions import TypedDict
import asyncio
import tempfile
import os


class ContractTerm(BaseModel):
    section: str = Field(description="The section of the contract this term belongs to")
    subsection: str = Field(description="The subsection of the contract this term belongs to, if applicable")
    term: str = Field(description="The specific term or condition")

class ContractTerms(BaseModel):
    terms: List[ContractTerm] = Field(description="List of extracted contract terms")

class TaskAnalysis(BaseModel):
    task_description: str = Field(description="Description of the task")
    cost: float = Field(description="Estimated cost of the task")
    compliant: bool = Field(description="Whether the task complies with the contract terms")
    violation_reason: str = Field(description="Reason for violation, if any")
    relevant_sections: List[str] = Field(description="Relevant contract sections for compliance or violation")

@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
        
    Function retrieved from: https://python.langchain.com/v0.2/docs/versions/migrating_chains/llm_math_chain/
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )

class LangChainGraph:
    """Class to build the LangGraph for the RAG pipeline.
    
    Args:
        model (str): The model to use for the RAG pipeline.
        temperature (float): The temperature to use for the RAG pipeline.        
    """
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.tools = [calculator]
        self.llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="any")

    class ChainState(TypedDict):
        """LangGraph state."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    async def acall_chain(self, state: ChainState, config: RunnableConfig):
        """
        Call the chain.
        
        Args:
            state (ChainState): The state of the chain.
            config (RunnableConfig): The config of the chain.
        
        Returns:
            dict: The response of the chain.
        """
        response = await self.llm_with_tools.ainvoke(state["messages"], config)
        return {"messages": [response]}

    async def acall_model(self, state: ChainState, config: RunnableConfig):
        """
        Call the model.
        
        Args:
            state (ChainState): The state of the chain.
            config (RunnableConfig): The config of the chain.
        
        Returns:
            dict: The response of the chain.
        """
        response = await self.llm.ainvoke(state["messages"], config)
        return {"messages": [response]}

    def build_graph(self):
        """
        Build the graph.
        
        Returns:
            StateGraph: The graph.
        """
        graph_builder = StateGraph(self.ChainState)
        graph_builder.add_node("call_tool", self.acall_chain)
        graph_builder.add_node("execute_tool", ToolNode(self.tools))
        graph_builder.add_node("call_model", self.acall_model)
        graph_builder.set_entry_point("call_tool")
        graph_builder.add_edge("call_tool", "execute_tool")
        graph_builder.add_edge("execute_tool", "call_model")
        graph_builder.add_edge("call_model", END)
        return graph_builder

async def check_compliance(task, chain):
    """
    Check the compliance of a task with the contract terms.
    
    Args:
        task (str): The task to check.
        chain (StateGraph): The chain to use for the RAG pipeline.
    
    Returns:
        TaskAnalysis: The analysis of the task.
    """
    prompt = f"""
    Extract the maximum allowable sum considering the task and the relevant terms of the contract. The amendment supersed the original contract. Compare the maximum allowable sum against the cost of the task. Output the results following this class:
    {TaskAnalysis.model_fields}

    Task:
    {task}
    """

    events = chain.astream(
        {"messages": [("user", prompt)]},
        stream_mode="values",
    )
    async for event in events:
        _ = event  # Keep updating with the latest event
        
    parser = PydanticOutputParser(pydantic_object=TaskAnalysis)
    return parser.parse(event['messages'][-1].content)
        
def load_contract(file_path: str) -> str:
    """
    Load the contract from the file path.
    
    Args:
        file_path (str): The file path to the contract.
    
    Returns:
        str: The content of the contract.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(file_path.getvalue())
        tmp_file_path = tmp_file.name
    loader = Docx2txtLoader(tmp_file_path)
    documents = loader.load()
    os.unlink(tmp_file_path)
    return documents[0].page_content

def extract_terms(contract_content: str) -> List[ContractTerm]:
    """
    Extract the terms from the contract content.
    
    Args:
        contract_content (str): The content of the contract.
    
    Returns:
        List[ContractTerm]: The terms extracted from the contract.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ContractTerms)
    prompt = ChatPromptTemplate.from_template(
        "Extract key terms from the following contract. Organize them by section and subsection:\n\n"
        "{contract_content}\n\n"
        "{format_instructions}"
    )
    chain = prompt | llm | parser
    result = chain.invoke({
        "contract_content": contract_content,
        "format_instructions": parser.get_format_instructions()
    })

    return result.terms

def load_tasks(file_path) -> List[Dict]:
    """
    Load the tasks from the file path.
    
    Args:
        file_path (str): The file path to the tasks.
    
    Returns:
        List[Dict]: The tasks loaded from the file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(file_path.getvalue())
        tmp_file_path = tmp_file.name
    
    df = pd.read_excel(tmp_file_path)
    tasks = []
    for _, row in df.iterrows():
        tasks.append({
            'description': row['Task Description'],
            'cost': float(row['Amount'].replace('$', '').replace(',', '')) if isinstance(row['Amount'], str) else float(row['Amount'])
    })
    os.unlink(tmp_file_path)
    return tasks

def analyze_task(task, vectorstore):
    """
    Analyze the task for compliance with the contract terms.
    
    Args:
        task (Dict): The task to analyze.
        vectorstore (FAISS): The vector store to use for the RAG pipeline.
    
    Returns:
        TaskAnalysis: The analysis of the task.
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
    relevant_terms = vectorstore.similarity_search(task['description'], k=5)
    relevant_terms_text = "\n".join([doc.page_content for doc in relevant_terms])
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following task for compliance with the contract terms:\n\n"
        "Task: {task_description}\n"
        "Cost: ${task_cost}\n\n"
        "Relevant Contract Terms:\n{relevant_terms}\n\n"
        "Note: Amendments override original contract terms. If there's ambiguity, prioritize the amendment.\n\n"
        "Analyze the task's compliance with the contract terms, considering the following aspects:\n"
        "1. Does the task align with the contracted software development services?\n"
        "2. Does the task comply with the travel expense policies?\n"
        "3. Are there any other contract terms that this task might violate?\n\n"
        "Compliant: [Yes/No]\n"
        "Provide your analysis in the following format:\n"
        "Reason: [If non-compliant, explain why in max 3 sentences. Otherwise, leave blank]\n"
        "Relevant Sections: [List the specific contract sections relevant to your analysis]"
    )
    response = llm.invoke(prompt.format(
        task_description=task['description'],
        task_cost=task['cost'],
        relevant_terms=relevant_terms_text
    ))
    
    response_text = response.content if hasattr(response, 'content') else response
    compliant = response_text.split("Compliant: ")[-1].split("\n")[0].strip().lower() == "yes"
    reason = response_text.split("Reason: ")[-1].split("Relevant Sections:")[0].strip()
    relevant_sections = response_text.split("Relevant Sections: ")[-1].strip().split(", ")
    
    return TaskAnalysis(
        task_description=task['description'],
        cost_estimate=task['cost'],
        compliant=compliant,
        violation_reason=reason if not compliant else "",
        relevant_sections=relevant_sections
    )

def load_contract_terms(file_path):
    """
    Load the contract terms from the file path.
    
    Args:
        file_path (str): The file path to the contract terms.
    
    Returns:
        List[ContractTerm]: The contract terms loaded from the file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def create_vector_store(contract_terms):
    """
    Create the vector store.
    
    Args:
        contract_terms (List[ContractTerm]): The contract terms to store.
    
    Returns:
        FAISS: The vector store.
    """
    documents = []
    for term in contract_terms:
        content = f"{term.section} {term.subsection} {term.term}"
        doc = Document(page_content=content, metadata=term)
        documents.append(doc)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def get_relevant_terms(vector_store, query, k=3):
    """
    Get the relevant terms from the vector store.
    
    Args:
        vector_store (FAISS): The vector store to use for the RAG pipeline.
        query (str): The query to search for.
        k (int): The number of results to return.
    
    Returns:
        List[ContractTerm]: The relevant terms from the vector store.
    """
    results = vector_store.similarity_search(query, k=k)
    return [doc.metadata for doc in results]

def process_tasks(tasks, vector_store, log_callback):
    """
    Process the tasks.
    
    Args:
        tasks (List[Dict]): The tasks to process.
        vector_store (FAISS): The vector store to use for the RAG pipeline.
        log_callback (function): The function to use to log the progress.
    
    Returns:
        List[Dict]: The processed tasks.
    """
    processed_tasks = []
    for task in tasks:
        log_callback(f"Processing task {len(tasks) - tasks.index(task)}/{len(tasks)}")
        query = f"{task['description']} cost: {task['cost']}"
        relevant_terms = get_relevant_terms(vector_store, query)
        processed_task = {
            'description': task['description'],
            'cost': task['cost'],
            'relevant_terms': relevant_terms
        }
        processed_tasks.append(processed_task)
    return processed_tasks

def load_and_extract_contract_terms(contract_path: str):
    """
    Load the contract and extract the terms.
    
    Args:
        contract_path (str): The file path to the contract.
    
    Returns:
        List[ContractTerm]: The contract terms loaded from the file.
    """
    contract_content = load_contract(contract_path)
    return extract_terms(contract_content)

def save_json(file_path: str, data: Any):
    """
    Save the data to the file path.
    
    Args:
        file_path (str): The file path to save the data to.
        data (Any): The data to save.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def check_tasks_compliance(processed_tasks: List[Dict]):
    """
    Check the tasks compliance.
    
    Args:
        processed_tasks (List[Dict]): The processed tasks.
    
    Returns:
        List[TaskAnalysis]: The analysis of the tasks.
    """
    lang_chain_graph = LangChainGraph()
    chain = lang_chain_graph.build_graph().compile()
    return [await check_compliance(task, chain=chain) for task in processed_tasks]

async def rag_extraction(contract_path: str, tasks_path: str, log_callback):
    """
    RAG extraction.
    
    Args:
        contract_path (str): The file path to the contract.
        tasks_path (str): The file path to the tasks.
        log_callback (function): The function to use to log the progress.
    
    Returns:
        str: The JSON string of the tasks.
    """
    log_callback("Loading and extracting contract terms...")
    contract_terms = load_and_extract_contract_terms(contract_path)

    log_callback("Loading tasks...")
    tasks = load_tasks(tasks_path)

    log_callback("Creating vector store...")
    vector_store = create_vector_store(contract_terms)

    log_callback("Processing tasks...")
    processed_tasks = process_tasks(tasks, vector_store, log_callback)


    log_callback("Checking tasks compliance...")
    checked_tasks = await check_tasks_compliance(processed_tasks)
    
    return json.dumps([task.model_dump() for task in checked_tasks])
 
def pipeline(file1, file2, log_callback):
    """
    Pipeline.
    
    Args:
        file1 (str): The file path to the contract.
        file2 (str): The file path to the tasks.
        log_callback (function): The function to use to log the progress.
    
    Returns:
        str: The JSON string of the tasks.
    """
    return asyncio.run(rag_extraction(file1, file2, log_callback))
    
if __name__ == "__main__":
    pipeline()
