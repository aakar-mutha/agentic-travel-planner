# Virtual Travel Agent

Virtual Travel is a Agentic AI which helps you plan a vacation.

## Motivation and Goals
* I like to travel, however I never know where to go or what to do. I can search the internet for things to do in a particular country but there is always a fear of it not being up to the mark. The itineraries that I find online are not customized to my needs. 

* The goal of this project is to make a team of AI agents that plans a vacation for me given my requirements. 

## APIs used:
1. LLM: Anthropic Claud 3 Haiku API.
   * Used to generate queries for tavily, use the information provided by tavily and give a make an itinerary. 
2. Search: Tavily API.
   * Used for getting relevant data for the user's request.

## Architecture:
1. There are 5 different prompts/agents which work together to plan your itinerary.
    * VACATION_PLANNING_SUPERVISOR_PROMPT: Gives the ouline of what the planner has to look for when planning a vacation.
    * PLANNER_ASSISTANT_PROMPT: Uses the haiku model to create prompts that will be useful to form the itinerary.
    * VACATION_PLANNER_PROMPT: Generates the itinerary in the given format using all the information gathered from the earlier steps.
    * PLANNER_CRITIQUE_PROMPT: It gives feedback on the itinerary that is generated. Suggests what else can be considered to plan the vacation.
    * PLANNER_CRITIQUE_ASSISTANT_PROMPT: Calls the tavily api for more information to gather information on suggestions.



<p align="center">
  <img src="/images/graph.png" />
</p>


## Installation

```bash
pip install -r requirements.txt
```
### Please add your Tavily and Anthropic API keys as environment variables in the format given in .env.example.

## Usage
You have 2 options
  1. Use the python notebook and run it cell by cell.
  2. Use the GUI:

```python
python3 helper.py
```
Open the link in the browser.

