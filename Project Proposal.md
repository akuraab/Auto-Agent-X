# Project Proposal: Auto-Agent-X – An Intelligent Agent-Driven R&D Platform

## 1. Group Members
*   **He Junqian**, Student ID: 3036654144

## 2. Project Objectives
This project aims to build **Auto-Agent-X**, an intelligent agent-driven R&D platform designed for the entire product research and development lifecycle. The core focus is on achieving the following three technical objectives:

### 2.1 Intelligent Conversational Assistant
*   **Multi-Dimensional Query Rewriting**: Incorporates a query rewriting mechanism that optimizes user input across multiple dimensions, including semantic expansion, keyword extraction, and time normalization.
*   **Parallelized Intent Recognition**: Designs a parallel processing pipeline that simultaneously analyzes deep user intent and matches against a high-priority FAQ database.
*   **Dynamic Strategy Orchestration**: Implements an adaptive execution flow based on recognition results. If an FAQ match is found, a standardized answer is returned immediately; for complex issues, a multi-step execution strategy (Plan-Execute) is dynamically planned to avoid ineffective reasoning.

### 2.2 Full-Link Context Engineering Optimization
*   **Structured Content Supply**: Enforces the conversion of retrieved unstructured data (code, logs, documents) into strict formats such as **JSON, Markdown, or Tables** before supplying it to the model.
*   **Pre-computation of Critical Logic**: Complex business dependencies and rule judgments ("hard logic") are pre-calculated using traditional code and injected as facts directly to the model, reducing the LLM's reasoning burden and error rate.
*   **Multi-Layer Validation Strategies**: Introduces mechanisms including **Template Validation** (ensuring output format compliance), **Subject/Time Clarification** (proactively asking questions when ambiguity exists), and **Null-Value Explanation** (explicitly stating data absence rather than fabricating information).

### 2.3 Multi-Agent Collaboration Framework: Progressive Disclosure and On-Demand Loading
*   **Progressive Disclosure**: The Portal Agent initially holds only lightweight metadata descriptions of sub-agents/scenarios. Dedicated System Prompts and toolsets for a specific scenario are injected *only* after the target scenario is confirmed.
*   **Automatic Routing**: The Portal Agent automatically routes tasks to the most suitable specialized sub-agents (e.g., Code Analysis, Log Troubleshooting, Document Retrieval, Change Impact Assessment) based on the problem type.
*   **Dynamic Loading**: Corresponding prompt resources and MCP tool definitions are loaded in real-time only when switching to a sub-agent.

## 3. Motivation and Challenges

### 3.1 Core Motivation
In large-scale distributed systems, R&D knowledge is highly fragmented. Developers spend excessive time "asking people, digging through historical documents, and guessing call chains" instead of focusing on core coding. There is a critical need for an intelligent platform capable of **autonomously understanding business logic**, **executing complex troubleshooting**, and **enabling reusability**.

### 3.2 Key Challenges
1.  **Excessive Cognitive Load**: Newcomers struggle to quickly grasp complex cross-repository call chains and business contexts.
2.  **Difficulty in Fault Localization**: Online incidents involve multi-dimensional data (code, logs, configurations, traces). Manually correlating these data points is extremely time-consuming.
3.  **LLM Hallucinations and Context Limits**: Naive RAG (Retrieval-Augmented Generation) often leads to hallucinations due to high input noise, while long contexts result in massive token consumption and slow response times.
4.  **Collaboration Barriers**: The lack of a unified knowledge sedimentation mechanism leads to high repetitive communication costs and difficulty in reusing historical experience.

## 4. Potential Solutions

### 4.1 Overall Architecture
*   **Interaction Layer**: Responsible for query rewriting, parallel intent recognition, and FAQ matching.
*   **Orchestration Layer (Portal Agent)**: The core "brain" that executes progressive disclosure, dynamic routing, and strategy orchestration.
*   **Execution Layer (Sub-Agent Cluster)**: Includes specialized agents such as Code Experts, Log Analysts, Document Assistants, and Security Scanners, which invoke underlying tools via the **MCP (Model Context Protocol)**.

### 4.2 Key Technical Implementation Details

#### A. Intelligent Query and Dynamic Strategy 
*   **Hybrid Execution Flow**: The system runs the "Intent Classifier" and "FAQ Retriever" in parallel upon startup.
    *   *FAQ Hit*: Returns a human-verified standard answer immediately with millisecond-level latency.
    *   *No Hit*: Triggers the "Plan-Execute-Feedback" engine to dynamically decompose the task (e.g., Check Code Definition → Analyze Call Chain → Inspect Logs).
*   **Query Rewriting Engine**: Utilizes small models to complete vague user queries (e.g., adding default time ranges, clarifying service names) to improve downstream retrieval precision.

#### B. Deep Context Engineering
*   **Structured Pipeline**:
    *   Retrieval Results $\rightarrow$ **Format Converter** (to JSON/Table) $\rightarrow$ **Logic Pre-computation Module** (injects dependency facts) $\rightarrow$ **Validator** (checks for null values/ambiguity) $\rightarrow$ LLM.
*   **Anti-Hallucination Mechanisms**:
    *   **Template Constraints**: Forces the LLM to output according to a predefined schema (e.g., Incident Reports must include: Symptoms, Root Cause, Evidence Links, Recommendations).
    *   **Proactive Clarification**: When ambiguous time ranges (e.g., "recently") or unclear subjects (e.g., "that service") are detected, the Agent pauses execution to ask the user for clarification rather than guessing blindly.

#### C. Efficient Multi-Agent Collaboration
*   **Lightweight Startup**: Upon initialization, the Portal Agent's context contains only metadata, e.g., `[{name: "CodeAgent", desc: "Specializes in Java code analysis"}, {name: "LogAgent", desc: "Specializes in RMS log troubleshooting"}]`.
*   **On-Demand Injection**:
    *   *User Query*: "Help me check the errors for the Order Service."
    *   *Routing*: The Portal Agent identifies the intent and routes to the `LogAgent`.
    *   *Dynamic Loading*: Only at this stage is the full prompt for the `LogAgent` (including RMS tool definitions and Log Analysis SOPs) appended to the context, and conversation control is handed over.

#### D. Underlying Support
*   **Deep Code Understanding**: Utilizes **AST (Abstract Syntax Trees)** to build code indexes for function-level precise retrieval; combines **Call Graph Analysis** to analyze cross-repository dependencies.
*   **MCP Tool Bus**: Adheres to the Model Context Protocol standard to seamlessly integrate GitLab (Code), RMS (Logs), Bizstack (Architecture), and SmartUnit (Testing).
*   **Performance Assurance**: Uses **Redis** to cache hot FAQs, session states, and pre-computed business logic, ensuring low latency under high concurrency.

## 5. Experimental Dataset and Evaluation Plan

### 5.1 Data Sources
*   **Code Repositories**: Enterprise-grade GitLab multi-language repositories (including Commit History and Merge Requests).

### 5.2 Evaluation Metrics

| Key Metrics | Expected Goals / Description |
| :--- | :--- |
| **Hallucination Rate** | Frequency of fabricating code paths, log content, or dependency relationships. |
| **Intent Recognition Accuracy** | Correctness rate of routing natural language inputs to the appropriate sub-agent/tools. |
| **FAQ Hit Rate** | Percentage of problems resolved directly via parallel FAQ matching (reducing LLM calls). |